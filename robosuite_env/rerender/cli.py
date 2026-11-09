# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc third-person rendering for no-video (``render.enabled: false`` /
``--no-render``) rollout batches.

Every completed run keeps its full per-frame state in ``logs/sim_log.jsonl``
and its exact worker invocation in ``RUN_COMMAND.md``.  This tool rebuilds
each run's command, swaps the policy-batch source for ``--state-log-replay``,
and renders the third-person MP4 at the canonical
``<scene>/<run>/<scene>_<run>.mp4`` path so the standard template grid can be
built afterwards.  The original policy artifacts (``logs/``,
``sim_summary.json``) are never overwritten: replay logs and the replay
summary go to ``<run>/rerender/``.

Typical single-GPU flow::

    real2sim-run \
        --config config/release/fr3/flat/a_room_cup_cup1trim.yaml
    python -m robosuite_env.tools.rerender_third_person <run_root> --gpu 0
"""


from __future__ import annotations

import argparse
from pathlib import Path

from robosuite_env.rerender.common import RerenderError
from robosuite_env.rerender.gpu import _acquire_posthoc_gpu_lock, resolve_gpu_binding
from robosuite_env.rerender.replay_argv import resolve_view_overrides
from robosuite_env.rerender.run import _run_rerender
from robosuite_env.rerender.sources import prepare_selected_sources


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "zero-based slot in the current CUDA_VISIBLE_DEVICES allocation; "
            "when it is unset, the value is a host physical GPU id"
        ),
    )
    parser.add_argument(
        "--gs-port",
        type=int,
        default=0,
        help=(
            "explicit GS port for deterministic parallel launches; each "
            "additional scene uses the next port (default: allocate one)"
        ),
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="skip the final template grid build",
    )
    parser.add_argument(
        "--runs",
        default="",
        help=(
            "comma-separated run directory names (e.g. run_01,run_02) to "
            "render; default renders every discovered run"
        ),
    )
    parser.add_argument(
        "--wrist-insets",
        choices=("saved", "off"),
        default="saved",
        help=(
            "saved requires and reuses the exact recorded policy wrist sequence "
            "without fallback; off omits wrist insets (default: saved)"
        ),
    )
    parser.add_argument(
        "--saved-wrist-display",
        choices=("aspect-fit", "raw224"),
        default="raw224",
        help=(
            "saved wrist inset presentation: raw224 preserves the full "
            "224x224 policy tensor and its black letterbox; aspect-fit removes "
            "the letterbox (default: raw224)"
        ),
    )
    parser.add_argument(
        "--view-azimuth",
        type=float,
        default=None,
        help=(
            "override the recorded third-person camera azimuth in degrees "
            "(default: keep the recorded viewpoint)"
        ),
    )
    parser.add_argument(
        "--view-elevation",
        type=float,
        default=None,
        help=(
            "override the recorded third-person camera elevation in degrees "
            "(default: keep the recorded viewpoint)"
        ),
    )
    parser.add_argument(
        "--view-distance",
        type=float,
        default=None,
        help=(
            "override the recorded third-person camera distance "
            "(default: keep the recorded viewpoint)"
        ),
    )
    parser.add_argument(
        "--view-lookat",
        default=None,
        help=(
            "override the recorded third-person camera lookat as 'x y z'; "
            "an empty string restores the worker default lookat "
            "(default: keep the recorded viewpoint)"
        ),
    )
    parser.add_argument(
        "--grid-template",
        type=Path,
        default=None,
        help=(
            "explicit grid template JSON used for every scene grid "
            "(default: the checked-in template for the selected run count)"
        ),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    arguments = parser.parse_args()
    if arguments.gpu < 0:
        parser.error("--gpu must be a non-negative integer")
    if arguments.gs_port < 0 or arguments.gs_port > 65535:
        parser.error("--gs-port must be 0 or an integer from 1 through 65535")
    # Fail closed on invalid overrides before touching sources or the GPU.
    resolve_view_overrides(arguments)
    if arguments.grid_template is not None and not Path(
        arguments.grid_template
    ).is_file():
        raise RerenderError(
            f"--grid-template file does not exist: {arguments.grid_template}"
        )
    sources, partial_selection = prepare_selected_sources(arguments)
    arguments.gpu_binding = resolve_gpu_binding(arguments.gpu)
    gpu_lifecycle_lock = _acquire_posthoc_gpu_lock(arguments.gpu_binding)
    try:
        return _run_rerender(
            arguments,
            sources=sources,
            partial_selection=partial_selection,
        )
    finally:
        gpu_lifecycle_lock.close()
