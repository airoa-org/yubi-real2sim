# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Authenticated Gaussian-splat server discovery and lifecycle setup."""

import os
import secrets
import sys

from robosuite_env.gs_launch import (
    GS_DEFAULT_GW_REPO,
    GsServerError,
    fetch_gs_health,
    gs_server_argv,
    launch_gs_server,
    verify_gs_health,
)


def _gs_server_identity(port, ready_token=""):
    """Probe ``/health``, returning ``None`` for anything unusable."""

    try:
        return fetch_gs_health(port, ready_token=ready_token)
    except GsServerError:
        return None


def _expected_ply_sha256(state):
    report = state.SCENE_CONTRACT_REPORT
    return None if report is None else report["ply_sha256"]


def _gs_server_matches(state, identity):
    """Report whether one ``/health`` identity is the server this run needs."""

    try:
        verify_gs_health(
            identity,
            expected_ply=state.A.gs,
            # Empty until a pre-existing server publishes its token below.
            expected_token=state.A.gs_server_token,
            expected_ply_sha256=_expected_ply_sha256(state),
        )
    except GsServerError:
        return False
    return True


def setup_gs_server(state):
    """Connect to the requested GS server or start its diagnostic fallback."""

    state.GS_URL = ""
    state.GS_SERVER_IDENTITY = None

    if state.A.gs:
        state.GS_URL = f"http://127.0.0.1:{state.A.gs_port}"
        state.GS_SERVER_IDENTITY = _gs_server_identity(
            state.A.gs_port, state.A.gs_server_token
        )
        if not _gs_server_matches(state, state.GS_SERVER_IDENTITY):
            if state.A.strict_assets:
                raise SystemExit(
                    "[sim] official GS server token/PLY identity mismatch @ "
                    f"{state.GS_URL}"
                )
            state.A.gs_server_token = secrets.token_hex(32)
            print(
                f"[sim] starting GS server on {state.A.gs_port} for "
                f"{os.path.basename(state.A.gs)} ...",
                flush=True,
            )
            state._gs_server_command = gs_server_argv(
                python=sys.executable,
                ply=state.A.gs,
                port=state.A.gs_port,
                ready_token=state.A.gs_server_token,
                gw_repo=os.environ.get(
                    "GAUSSIAN_WRAPPING_REPO", GS_DEFAULT_GW_REPO
                ),
                render_lock=state._RENDER_LOCK_PATH,
            )
            # launch_gs_server closes the parent's copy of the log handle as
            # soon as Popen has duplicated it for the child, so there is no
            # spooled writer left for the resource cleanup to close.
            state.GS_SERVER_LOG = None
            try:
                state.GS_SERVER_PROCESS, state.GS_SERVER_IDENTITY = (
                    launch_gs_server(
                        python=sys.executable,
                        ply=state.A.gs,
                        port=state.A.gs_port,
                        ready_token=state.A.gs_server_token,
                        log_path=os.path.join(
                            state._DIR, "logs", f"gs_{state.A.gs_port}.log"
                        ),
                        expected_ply_sha256=_expected_ply_sha256(state),
                        argv=state._gs_server_command,
                    )
                )
            except GsServerError as exc:
                raise SystemExit(
                    "[sim] GS server did not publish the expected token/PLY "
                    f"identity @ {state.GS_URL}: {exc}"
                ) from exc
        if not state.A.gs_server_token:
            state.A.gs_server_token = str(
                state.GS_SERVER_IDENTITY.get("ready_token", "")
            )
        print(
            f"[sim] GS server UP with verified PLY identity @ {state.GS_URL}",
            flush=True,
        )


__all__ = ["setup_gs_server"]
