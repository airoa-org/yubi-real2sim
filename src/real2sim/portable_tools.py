# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Link the locked imageio-ffmpeg binary as the environment's FFmpeg launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


FFMPEG_VERSION = "7.0.2-static"


class PortableToolError(RuntimeError):
    """Raised when a portable tool cannot be installed or verified."""


def _atomic_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(relative_source)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _install_ffmpeg(*, environment_bin: Path) -> Path:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise PortableToolError("locked imageio-ffmpeg package is not installed") from exc
    version = imageio_ffmpeg.get_ffmpeg_version()
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if version != FFMPEG_VERSION or not executable.is_file():
        raise PortableToolError(
            f"expected FFmpeg {FFMPEG_VERSION}, found {version!r} at {executable}"
        )
    _atomic_symlink(executable, environment_bin / "ffmpeg")
    return executable


def install_portable_tools(*, environment_bin: Path) -> dict[str, str]:
    ffmpeg = _install_ffmpeg(environment_bin=environment_bin)
    return {"ffmpeg": str(ffmpeg)}


def _verify(executable: Path, arguments: tuple[str, ...], expected: str) -> None:
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PortableToolError(f"could not execute {executable}: {exc}") from exc
    detail = completed.stdout + completed.stderr
    if completed.returncode != 0 or expected not in detail:
        raise PortableToolError(
            f"unexpected version output from {executable}: {detail.strip()}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-bin", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        installed = install_portable_tools(
            environment_bin=args.environment_bin.resolve(),
        )
        _verify(
            args.environment_bin / "ffmpeg",
            ("-version",),
            f"ffmpeg version {FFMPEG_VERSION}",
        )
    except PortableToolError as exc:
        print(f"portable tool install error: {exc}", file=sys.stderr)
        return 1
    for name, path in sorted(installed.items()):
        print(f"[portable-tools] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
