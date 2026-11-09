# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc third-person rendering package.

Behavior-preserving split of ``robosuite_env/tools/rerender_third_person.py``.
Every top-level name of the original module (public and private) is
re-exported here, so ``from robosuite_env.rerender import X`` works for all of
them and the original CLI module keeps working as a thin shim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robosuite_env.rerender.common import (
    GRID_TEMPLATES,
    GRID_TOOL,
    GS_SERVER,
    GpuBinding,
    REPO_ROOT,
    RerenderError,
    SourceRun,
    _BATCH_POLICY_LINEAGE_FILENAMES,
    _COMMIT_RE,
    _CORE_RENDERER_SOURCE_PATHS,
    _GS_OFF_VALUES,
    _SHA256_RE,
    _canonical_video_path,
    _last_option_value,
    _rollout_expected_frames,
    _load_json_object,
    _resolved_recorded_path,
)
from robosuite_env.rerender.provenance import (
    _assert_record_current,
    _assert_renderer_code_current,
    _core_renderer_source_records,
    _file_record,
    _file_state,
    _git_code_identity,
    _git_head_commit,
    _git_tracked_diff_identity,
    _sha256_file,
)
from robosuite_env.rerender.registration import (
    _assert_registration_matches,
    _capture_summary_gs_registration,
    _load_registration_file,
    _registration_matrix,
    _registration_raw_matrix,
    _require_summary_close,
    _required_finite_summary_number,
)
from robosuite_env.rerender.cadence import (
    _fr3_policy_action_playback_contract,
    _validate_policy_action_replay_cadence,
)
from robosuite_env.rerender.replay_argv import (
    _DROP_FLAGS,
    _DROP_WITH_VALUE,
    _REWRITE_WITH_VALUE,
    _VIEW_OVERRIDE_OPTIONS,
    _without_option,
    build_replay_argv,
    parse_run_command,
    resolve_view_overrides,
)
from robosuite_env.rerender.gpu import (
    _acquire_posthoc_gpu_lock,
    _coerce_gpu_binding,
    _nvidia_gpu_uuid_to_index,
    canonical_render_lock,
    resolve_cuda_device,
    resolve_gpu_binding,
)
from robosuite_env.rerender.sources import (
    _assert_batch_policy_lineage_current,
    _capture_batch_policy_lineage,
    _load_generation_summary,
    _load_state_rows,
    _load_state_rows_with_record,
    _option_present,
    _select_run_paths,
    _validate_official_source,
    discover_runs,
    prepare_selected_sources,
    prepare_source_run,
)
from robosuite_env.rerender.gs_server import (
    _capture_scene_inputs,
    _current_gs_registration,
    _free_port,
    _registration_scene_signature,
    _scene_ply,
    _start_gs_server,
    _stop_process,
    _validated_scene_identity,
    gs_server_environment,
    replay_worker_environment,
)
from robosuite_env.rerender.validation import (
    _sanitized_replay_argv,
    validate_rerender_output,
    write_rerender_provenance,
)
from robosuite_env.rerender.batch import (
    _recorded_scene_dir,
    build_scene_grids,
    resolve_wrist_insets,
    update_outputs_csv_after_rerender,
)
from robosuite_env.rerender.run import _run_rerender
from robosuite_env.rerender.cli import build_arg_parser, main

from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    ReplayPolicyInsetSchedule,
    load_replay_policy_insets,
)
from robosuite_env.rollout.state.render_state import (
    validate_render_state_rows,
)
from robosuite_env.contracts.scene import (
    SceneContractError,
    validate_scene_contract,
)
from robosuite_env.contracts.worker import (
    WorkerContractError,
    validate_worker_contract,
)
from robosuite_env.tools.validate_rollout_video import (
    VideoValidationError,
    validate_saved_wrist_insets,
    validate_video,
    write_json_atomic,
)

__all__ = [
    "argparse",
    "csv",
    "datetime",
    "timezone",
    "fcntl",
    "hashlib",
    "http",
    "json",
    "math",
    "os",
    "re",
    "shlex",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "time",
    "urllib",
    "dataclass",
    "Path",
    "Any",
    "np",
    "REPO_ROOT",
    "ReplayPolicyInsetError",
    "ReplayPolicyInsetSchedule",
    "load_replay_policy_insets",
    "validate_render_state_rows",
    "SceneContractError",
    "validate_scene_contract",
    "WorkerContractError",
    "validate_worker_contract",
    "VideoValidationError",
    "validate_saved_wrist_insets",
    "validate_video",
    "write_json_atomic",
    "GRID_TOOL",
    "GS_SERVER",
    "GRID_TEMPLATES",
    "_BATCH_POLICY_LINEAGE_FILENAMES",
    "_CORE_RENDERER_SOURCE_PATHS",
    "_DROP_WITH_VALUE",
    "_DROP_FLAGS",
    "_VIEW_OVERRIDE_OPTIONS",
    "_REWRITE_WITH_VALUE",
    "RerenderError",
    "SourceRun",
    "GpuBinding",
    "_SHA256_RE",
    "_COMMIT_RE",
    "_GS_OFF_VALUES",
    "_file_state",
    "_sha256_file",
    "_file_record",
    "_git_head_commit",
    "_git_tracked_diff_identity",
    "_core_renderer_source_records",
    "_git_code_identity",
    "_assert_renderer_code_current",
    "_resolved_recorded_path",
    "_canonical_video_path",
    "_without_option",
    "resolve_cuda_device",
    "_nvidia_gpu_uuid_to_index",
    "resolve_gpu_binding",
    "parse_run_command",
    "_last_option_value",
    "_rollout_expected_frames",
    "_registration_matrix",
    "_load_registration_file",
    "_capture_summary_gs_registration",
    "_registration_raw_matrix",
    "_assert_registration_matches",
    "_required_finite_summary_number",
    "_require_summary_close",
    "_fr3_policy_action_playback_contract",
    "resolve_view_overrides",
    "build_replay_argv",
    "_load_state_rows_with_record",
    "_load_state_rows",
    "_load_generation_summary",
    "_assert_record_current",
    "_capture_batch_policy_lineage",
    "_assert_batch_policy_lineage_current",
    "_option_present",
    "_load_json_object",
    "_validate_official_source",
    "prepare_source_run",
    "resolve_wrist_insets",
    "_coerce_gpu_binding",
    "canonical_render_lock",
    "_acquire_posthoc_gpu_lock",
    "_free_port",
    "_scene_ply",
    "gs_server_environment",
    "replay_worker_environment",
    "_start_gs_server",
    "_stop_process",
    "_capture_scene_inputs",
    "_current_gs_registration",
    "_registration_scene_signature",
    "_validated_scene_identity",
    "_validate_policy_action_replay_cadence",
    "validate_rerender_output",
    "_sanitized_replay_argv",
    "write_rerender_provenance",
    "discover_runs",
    "update_outputs_csv_after_rerender",
    "_recorded_scene_dir",
    "build_scene_grids",
    "_select_run_paths",
    "prepare_selected_sources",
    "_run_rerender",
    "build_arg_parser",
    "main",
]
