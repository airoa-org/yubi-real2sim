# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free helpers for deterministic rollout cleanup."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
from dataclasses import dataclass
import fcntl
import os
import subprocess
import sys
import threading
from typing import Any


@dataclass(frozen=True)
class CleanupFailure:
    """One resource cleanup failure that did not block later cleanup steps."""

    resource: str
    error_type: str
    detail: str

    def __str__(self) -> str:
        return f"{self.resource}: {self.error_type}: {self.detail}"


class IdempotentCleanup:
    """Run named cleanup callbacks exactly once and isolate their failures.

    Rollout workers own several independent GPU, file, and video resources.
    A failure while closing one resource must not prevent the remaining ones
    from being released.  ``run`` is thread-safe and returns the same immutable
    report to every caller, which makes it safe to use from both a normal
    ``finally`` block and an ``atexit`` fallback.
    """

    def __init__(
        self,
        steps: Iterable[tuple[str, Callable[[], None]]],
    ) -> None:
        self._steps = tuple(steps)
        if not self._steps:
            raise ValueError("cleanup requires at least one step")
        names = [name for name, _callback in self._steps]
        if any(not str(name).strip() for name in names):
            raise ValueError("cleanup resource names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("cleanup resource names must be unique")
        if any(not callable(callback) for _name, callback in self._steps):
            raise TypeError("cleanup callbacks must be callable")
        self._lock = threading.RLock()
        self._finished = False
        self._failures: tuple[CleanupFailure, ...] = ()

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def run(self) -> tuple[CleanupFailure, ...]:
        """Run each callback once, returning all observed failures."""

        with self._lock:
            if self._finished:
                return self._failures
            # Mark the cleanup as consumed before invoking user callbacks.  A
            # callback that recursively triggers cleanup must not close an
            # earlier resource twice.
            self._finished = True
            failures: list[CleanupFailure] = []
            for resource, callback in self._steps:
                try:
                    callback()
                except Exception as exc:
                    failures.append(
                        CleanupFailure(
                            resource=str(resource),
                            error_type=type(exc).__name__,
                            detail=str(exc),
                        )
                    )
            self._failures = tuple(failures)
            return self._failures


@contextlib.contextmanager
def mujoco_render_lock(session):
    """Serialize GPU rendering across workers, with safe process-local re-entry."""
    if not session._RENDER_LOCK_PATH:
        yield
        return
    depth = int(getattr(session._RENDER_LOCK_STATE, "depth", 0))
    if depth:
        session._RENDER_LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            session._RENDER_LOCK_STATE.depth -= 1
        return
    os.makedirs(os.path.dirname(session._RENDER_LOCK_PATH), exist_ok=True)
    fd = os.open(session._RENDER_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        session._RENDER_LOCK_STATE.depth = 1
        yield
    finally:
        session._RENDER_LOCK_STATE.depth = 0
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def close_renderer_with_retry(resource, *, attempts=2):
    """Close one renderer before releasing the physical-GPU lifecycle lock."""
    failures = []
    for _attempt in range(max(1, int(attempts))):
        try:
            resource.close()
        except Exception as exc:
            failures.append(exc)
        else:
            return None
    return failures[-1]


def close_renderers(session, resource_names):
    """Close named session renderers, retrying inside the EGL process lock."""
    errors = []
    closed_ids = set()
    with session.render_lock():
        for resource_name in resource_names:
            resource = getattr(session.renderers, resource_name)
            if resource is None:
                continue
            resource_id = id(resource)
            if resource_id in closed_ids:
                setattr(session.renderers, resource_name, None)
                continue
            close_error = close_renderer_with_retry(resource)
            if close_error is None:
                closed_ids.add(resource_id)
                setattr(session.renderers, resource_name, None)
            else:
                errors.append(
                    f"{resource_name}: close failed after 2 attempts: {close_error}"
                )
    return errors


def cleanup_rollout_resources_at_exit(session):
    """Best-effort fallback for exceptions before the normal video ``finally``."""
    cleanup = session._ROLLOUT_CLEANUP
    if cleanup is None:
        return
    failures = cleanup.run()
    if failures:
        print(
            "[sim] rollout cleanup warning: "
            + "; ".join(str(failure) for failure in failures),
            file=sys.stderr,
            flush=True,
        )


def terminate_rollout(signum, _frame):
    """Turn scheduler termination into normal Python unwinding plus atexit."""
    raise SystemExit(128 + int(signum))


class GlobalResourceCleanup:
    """Clean resources resolved lazily from a mutable globals mapping."""

    def __init__(
        self,
        resources: dict[str, Any],
        *,
        render_lock: Callable[[], Any],
    ) -> None:
        self._resources = resources
        self._render_lock = render_lock
        self.cleanup = IdempotentCleanup(
            (
                ("simulation log", lambda: self._close_method("logf", "close")),
                (
                    "overview video spool",
                    lambda: self._close_method("_OVERVIEW_VIDEO_FRAMES", "close"),
                ),
                (
                    "wrist video spool",
                    lambda: self._close_method("_WRIST_VIDEO_FRAMES", "close"),
                ),
                (
                    "real video capture",
                    lambda: self._close_method("realcap", "release"),
                ),
                ("MuJoCo renderers", self._close_renderers),
                ("robosuite environment", self._close_environment),
                ("fallback GS server", self._terminate_process),
                (
                    "fallback GS server log",
                    lambda: self._close_method("GS_SERVER_LOG", "close"),
                ),
            )
        )

    def _close_method(self, name: str, method: str) -> None:
        resource = self._resources.get(name)
        if resource is None:
            return
        try:
            getattr(resource, method)()
        finally:
            self._resources[name] = None

    def _close_renderers(self) -> None:
        errors: list[str] = []
        closed_ids: set[int] = set()
        retry: list[tuple[str, Any, Exception]] = []
        with self._render_lock():
            # Borrowers detach first; owners then release the shared EGL
            # context. This ordering also leaves an owner handle available for
            # one retry if its underlying renderer close fails transiently.
            for name in ("wfL_obs", "wfR_obs", "wfL", "wfR", "ren"):
                resource = self._resources.get(name)
                if resource is None:
                    continue
                if id(resource) in closed_ids:
                    self._resources[name] = None
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    retry.append((name, resource, exc))
                else:
                    closed_ids.add(id(resource))
                    self._resources[name] = None
            for name, resource, first_error in retry:
                if self._resources.get(name) is not resource:
                    continue
                try:
                    resource.close()
                except Exception as retry_error:
                    errors.append(
                        f"{name}: {first_error}; retry: {retry_error}"
                    )
                else:
                    closed_ids.add(id(resource))
                    self._resources[name] = None
        if errors:
            raise RuntimeError("; ".join(errors))

    def _close_environment(self) -> None:
        resource = self._resources.get("env")
        if resource is None:
            return
        try:
            with self._render_lock():
                resource.close()
        finally:
            self._resources["env"] = None

    def _terminate_process(self) -> None:
        process = self._resources.get("GS_SERVER_PROCESS")
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            self._resources["GS_SERVER_PROCESS"] = None


__all__ = [
    "CleanupFailure",
    "GlobalResourceCleanup",
    "IdempotentCleanup",
    "cleanup_rollout_resources_at_exit",
    "close_renderer_with_retry",
    "close_renderers",
    "mujoco_render_lock",
    "terminate_rollout",
]
