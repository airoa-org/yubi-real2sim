# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Single definition of the 3DGS background-server launch and health contract.

Two callers start ``scene/gs_background_server.py`` and then refuse to use it
until its ``/health`` identity proves it loaded the exact PLY that was asked
for: the post-hoc rerender pipeline
(:mod:`robosuite_env.rerender.gs_server`) and the rollout worker's diagnostic
fallback (:mod:`robosuite_env.rollout.build.gs_server`).  Everything both
sides must agree on -- the argv, the four-part identity check, ready timeouts,
failure types and the shape of the published digest -- lives here, so they
differ only where a call site deliberately passes a different argument.

The server binds its port only after the CUDA renderer has finished loading
(``GpuRenderOwner.start`` blocks on its readiness future before
``ThreadingHTTPServer`` is constructed), so a reachable ``/health`` is always a
*ready* ``/health``.  A transport failure therefore means "not up yet" and is
retried, while an answered-but-wrong identity is a hard failure and is not.

This module imports nothing from :mod:`robosuite_env` on purpose.
``robosuite_env.rerender`` already imports ``robosuite_env.rollout`` (for
``replay_policy_insets`` and ``state.render_state``), so a shared helper placed
under ``rerender`` would turn that into an import cycle the moment the rollout
side imported it; a helper placed under ``rollout`` would invert the layering.
Importing it from ``scene/`` instead would force the rollout package to put the
repository root on ``sys.path`` at module-import time, which it deliberately
avoids today (every ``scene.*`` import inside ``robosuite_env`` is
function-local).  A stdlib-only leaf at the ``robosuite_env`` root is reachable
from both without either problem.
"""


from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


#: The server executable both callers launch.
GS_SERVER_SCRIPT = REPO_ROOT / "scene" / "gs_background_server.py"

#: Default GaussianWrapping checkout passed as ``--gw-repo``.
GS_DEFAULT_GW_REPO = REPO_ROOT / "third_party" / "GaussianWrapping"

#: Render size both callers have always requested.
GS_DEFAULT_WIDTH = 640
GS_DEFAULT_HEIGHT = 480

GS_HEALTH_HOST = "127.0.0.1"
GS_HEALTH_TOKEN_HEADER = "X-Real2Sim-GS-Token"

#: The GS server only ever listens on loopback.  ``urllib.request.urlopen``
#: honours ``http_proxy``/``HTTP_PROXY`` even for 127.0.0.1 unless ``no_proxy``
#: names it, which sends the ready token to the proxy and makes a healthy local
#: server look unreachable.  Loopback probes therefore never use a proxy.
GS_LOOPBACK_NO_PROXY = "127.0.0.1,localhost,::1"
_LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open_loopback(request: urllib.request.Request, *, timeout: float) -> Any:
    """Open one loopback GS request without any environment proxy."""

    return _LOOPBACK_OPENER.open(request, timeout=timeout)


def _with_loopback_entries(value: str) -> str:
    """Append the loopback hosts that ``value`` does not already name."""

    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    named = {entry.lstrip(".").lower() for entry in entries}
    entries.extend(
        host for host in GS_LOOPBACK_NO_PROXY.split(",") if host not in named
    )
    return ",".join(entries)


def with_loopback_no_proxy(environment: Mapping[str, str]) -> dict[str, str]:
    """Return ``environment`` whose ``no_proxy`` also bypasses loopback.

    A child that talks to the GS server through ``urllib`` (the rollout
    worker's GS client) inherits any proxy variables.  This adds loopback to
    the exclusions that child already has and changes nothing else:

    * every spelling of ``no_proxy`` that is set keeps its own entries, so a
      non-loopback host that was direct, or proxied, stays that way;
    * a value that is exactly ``*`` is left alone: urllib treats only the
      whole value ``*`` as "bypass every host", so appending to it would
      send every other host through the proxy;
    * a spelling is created only where no exclusion is in effect.  urllib
      prefers a present lowercase ``no_proxy`` over ``NO_PROXY``, so adding a
      lowercase value next to a set ``NO_PROXY`` would hide that value.
    """

    result = dict(environment)
    names = [name for name in result if name.lower() == "no_proxy"]
    for name in names:
        value = str(result[name])
        if value.strip() and value != "*":
            result[name] = _with_loopback_entries(value)
    lowercase = result.get("no_proxy")
    if lowercase is None:
        if not any(str(result[name]).strip() for name in names):
            result["no_proxy"] = GS_LOOPBACK_NO_PROXY
    elif not str(lowercase).strip():
        # urllib reads a present but empty lowercase value as "no exclusions"
        # and ignores NO_PROXY, so loopback is the only host this makes direct.
        result["no_proxy"] = GS_LOOPBACK_NO_PROXY
    return result


#: A ``/health`` identity is four short strings; anything larger is a hostile or
#: broken peer rather than this server.
GS_HEALTH_MAX_BYTES = 65536

#: Per-request socket timeout for one ``/health`` probe.
GS_HEALTH_REQUEST_TIMEOUT_SEC = 5.0

#: Total time a freshly launched server is allowed to become ready.  Loading a
#: large PLY into CUDA on a busy node takes minutes, so this is deliberately
#: generous; the loop below exits early anyway if the child dies.
GS_READY_TIMEOUT_SEC = 600.0

#: Delay between ``/health`` probes while the server is still coming up.
GS_READY_POLL_SEC = 0.25


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GsServerError(RuntimeError):
    """Raised when a GS background server cannot be launched or trusted."""


def gs_server_argv(
    *,
    python: str,
    ply: str | os.PathLike[str],
    port: int,
    ready_token: str,
    gw_repo: str | os.PathLike[str] | None = None,
    render_lock: str = "",
    host: str = GS_HEALTH_HOST,
    width: int = GS_DEFAULT_WIDTH,
    height: int = GS_DEFAULT_HEIGHT,
    server_script: str | os.PathLike[str] = GS_SERVER_SCRIPT,
) -> list[str]:
    """Build the exact argv that starts one GS background server.

    ``gw_repo`` defaults to the in-repo GaussianWrapping checkout.  It stays a
    parameter rather than reading ``GAUSSIAN_WRAPPING_REPO`` here, because the
    rerender pipeline binds the renderer's code identity and must not have its
    GaussianWrapping checkout swapped by an ambient environment variable.
    """

    return [
        str(python),
        "-u",
        str(server_script),
        "--ply",
        str(ply),
        "--gw-repo",
        str(GS_DEFAULT_GW_REPO if gw_repo is None else gw_repo),
        "--host",
        str(host),
        "--port",
        str(port),
        "--ready-token",
        str(ready_token),
        *(["--render-lock", str(render_lock)] if render_lock else []),
        "--width",
        str(width),
        "--height",
        str(height),
    ]


def fetch_gs_health(
    port: int,
    *,
    ready_token: str = "",
    timeout: float = GS_HEALTH_REQUEST_TIMEOUT_SEC,
    host: str = GS_HEALTH_HOST,
) -> Any | None:
    """Return the parsed ``/health`` identity, or ``None`` if unreachable.

    Transport failures return ``None`` so a caller can keep waiting.  A server
    that answers with an oversized or unparsable body is answering wrongly, not
    slowly, so that raises :class:`GsServerError`.
    """

    request = urllib.request.Request(
        f"http://{host}:{port}/health",
        headers=(
            {GS_HEALTH_TOKEN_HEADER: str(ready_token)} if ready_token else {}
        ),
        method="GET",
    )
    try:
        with _open_loopback(request, timeout=timeout) as response:
            raw_body = response.read(GS_HEALTH_MAX_BYTES + 1)
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ):
        return None
    if len(raw_body) > GS_HEALTH_MAX_BYTES:
        raise GsServerError("GS health response exceeds 64 KiB")
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GsServerError(f"GS health response is invalid JSON: {exc}") from exc


def verify_gs_health(
    identity: Any,
    *,
    expected_ply: str | os.PathLike[str],
    expected_token: str = "",
    expected_ply_sha256: str | None = None,
) -> str:
    """Validate one ``/health`` identity and return its verified PLY digest.

    The four parts are ``status == "ok"``, the ready token, the resolved PLY
    path, and a well-formed PLY digest.  ``expected_token`` may be empty when
    the caller is adopting the token published by a server it did not start;
    ``expected_ply_sha256`` is checked only when the caller already knows the
    digest the scene contract requires.
    """

    if not isinstance(identity, dict) or identity.get("status") != "ok":
        raise GsServerError(f"GS health response is not ready: {identity!r}")
    if expected_token and identity.get("ready_token") != expected_token:
        raise GsServerError("authenticated GS health identity token mismatch")
    expected_path = resolved_ply_path(expected_ply)
    if identity.get("ply_path") != expected_path:
        raise GsServerError(
            "GS health PLY identity mismatch: "
            f"{identity.get('ply_path')!r} != {expected_path!r}"
        )
    ply_sha256 = identity.get("ply_sha256")
    if not isinstance(ply_sha256, str) or _SHA256_RE.fullmatch(ply_sha256) is None:
        raise GsServerError("GS health has no valid PLY SHA-256")
    if expected_ply_sha256 is not None and ply_sha256 != expected_ply_sha256:
        raise GsServerError(
            "GS health PLY digest mismatch: "
            f"{ply_sha256!r} != {expected_ply_sha256!r}"
        )
    return ply_sha256


def resolved_ply_path(ply: str | os.PathLike[str]) -> str:
    """Normalize a PLY path exactly as the server normalizes its ``--ply``."""

    try:
        return str(Path(ply).expanduser().resolve())
    except (OSError, TypeError, ValueError) as exc:
        raise GsServerError(f"cannot resolve the expected GS PLY: {exc}") from exc


def wait_for_gs_health(
    process: subprocess.Popen | None,
    *,
    port: int,
    ready_token: str,
    expected_ply: str | os.PathLike[str],
    expected_ply_sha256: str | None = None,
    log_path: str | os.PathLike[str] | None = None,
    timeout_sec: float = GS_READY_TIMEOUT_SEC,
    poll_interval_sec: float = GS_READY_POLL_SEC,
    request_timeout_sec: float = GS_HEALTH_REQUEST_TIMEOUT_SEC,
    host: str = GS_HEALTH_HOST,
) -> dict[str, str]:
    """Poll ``/health`` until the server publishes the expected identity.

    Returns the verified identity.  ``process``, when given, is polled first on
    every iteration so a server that died during startup fails immediately
    instead of at the deadline.
    """

    location = "" if log_path is None else f"; see {log_path}"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise GsServerError(f"GS server exited early{location}")
        identity = fetch_gs_health(
            port,
            ready_token=ready_token,
            timeout=request_timeout_sec,
            host=host,
        )
        if identity is None:
            time.sleep(poll_interval_sec)
            continue
        digest = verify_gs_health(
            identity,
            expected_ply=expected_ply,
            expected_token=ready_token,
            expected_ply_sha256=expected_ply_sha256,
        )
        return {
            "status": "ok",
            "ready_token": str(ready_token),
            "ply_path": resolved_ply_path(expected_ply),
            "ply_sha256": digest,
        }
    raise GsServerError(f"GS server never became ready{location}")


def launch_gs_server(
    *,
    python: str,
    ply: str | os.PathLike[str],
    port: int,
    ready_token: str,
    log_path: str | os.PathLike[str],
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    gw_repo: str | os.PathLike[str] | None = None,
    render_lock: str = "",
    host: str = GS_HEALTH_HOST,
    width: int = GS_DEFAULT_WIDTH,
    height: int = GS_DEFAULT_HEIGHT,
    expected_ply_sha256: str | None = None,
    timeout_sec: float = GS_READY_TIMEOUT_SEC,
    poll_interval_sec: float = GS_READY_POLL_SEC,
    request_timeout_sec: float = GS_HEALTH_REQUEST_TIMEOUT_SEC,
    argv: Sequence[str] | None = None,
) -> tuple[subprocess.Popen, dict[str, str]]:
    """Start one GS server and return it only once it proves its identity.

    The process is stopped again on every failure path, so a caller never has
    to reap a server it was told not to trust.
    """

    command = list(
        gs_server_argv(
            python=python,
            ply=ply,
            port=port,
            ready_token=ready_token,
            gw_repo=gw_repo,
            render_lock=render_lock,
            host=host,
            width=width,
            height=height,
        )
        if argv is None
        else argv
    )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=None if env is None else dict(env),
        )
    finally:
        # Popen duplicates the descriptor for the child; the parent should not
        # keep every scene log open for the lifetime of a multi-scene replay.
        handle.close()
    try:
        health = wait_for_gs_health(
            process,
            port=port,
            ready_token=ready_token,
            expected_ply=ply,
            expected_ply_sha256=expected_ply_sha256,
            log_path=log_path,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            request_timeout_sec=request_timeout_sec,
            host=host,
        )
    except BaseException:
        stop_process(process)
        raise
    return process, health


def stop_process(process: subprocess.Popen, *, timeout: float = 30.0) -> None:
    """Stop a helper process and guarantee reaping, even if TERM is ignored."""

    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


__all__ = [
    "GS_DEFAULT_GW_REPO",
    "GS_DEFAULT_HEIGHT",
    "GS_DEFAULT_WIDTH",
    "GS_HEALTH_HOST",
    "GS_HEALTH_MAX_BYTES",
    "GS_HEALTH_REQUEST_TIMEOUT_SEC",
    "GS_HEALTH_TOKEN_HEADER",
    "GS_READY_POLL_SEC",
    "GS_READY_TIMEOUT_SEC",
    "GS_SERVER_SCRIPT",
    "GsServerError",
    "REPO_ROOT",
    "fetch_gs_health",
    "gs_server_argv",
    "launch_gs_server",
    "resolved_ply_path",
    "stop_process",
    "verify_gs_health",
    "wait_for_gs_health",
]
