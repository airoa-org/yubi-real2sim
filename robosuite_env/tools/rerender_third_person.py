#!/usr/bin/env python3
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

The implementation lives in the :mod:`robosuite_env.rerender` package; this
module is a thin compatibility shim that re-exports every original top-level
name and keeps the ``python -m robosuite_env.tools.rerender_third_person``
entry point working.
"""

from __future__ import annotations

from robosuite_env.rerender import (  # noqa: F401
    argparse, csv, datetime, timezone, fcntl, hashlib, http, json, math, os,
    re, shlex, socket, subprocess, sys, tempfile, time, urllib, dataclass,
    Path, Any, np, REPO_ROOT, ReplayPolicyInsetError,
    ReplayPolicyInsetSchedule, load_replay_policy_insets,
    validate_render_state_rows, SceneContractError, validate_scene_contract,
    WorkerContractError, validate_worker_contract, VideoValidationError,
    validate_saved_wrist_insets, validate_video, write_json_atomic, GRID_TOOL,
    GS_SERVER, GRID_TEMPLATES, _BATCH_POLICY_LINEAGE_FILENAMES,
    _CORE_RENDERER_SOURCE_PATHS, _DROP_WITH_VALUE, _DROP_FLAGS,
    _VIEW_OVERRIDE_OPTIONS, _REWRITE_WITH_VALUE, RerenderError, SourceRun,
    GpuBinding, _SHA256_RE, _COMMIT_RE, _GS_OFF_VALUES, _file_state,
    _sha256_file, _file_record, _git_head_commit, _git_tracked_diff_identity,
    _core_renderer_source_records, _git_code_identity,
    _assert_renderer_code_current, _resolved_recorded_path,
    _canonical_video_path, _without_option, resolve_cuda_device,
    _nvidia_gpu_uuid_to_index, resolve_gpu_binding, parse_run_command,
    _last_option_value, _registration_matrix, _load_registration_file,
    _capture_summary_gs_registration, _registration_raw_matrix,
    _assert_registration_matches, _required_finite_summary_number,
    _require_summary_close, _fr3_policy_action_playback_contract,
    resolve_view_overrides, build_replay_argv, _load_state_rows_with_record,
    _load_state_rows, _load_generation_summary, _assert_record_current,
    _capture_batch_policy_lineage, _assert_batch_policy_lineage_current,
    _option_present, _load_json_object, _validate_official_source,
    prepare_source_run, resolve_wrist_insets, _coerce_gpu_binding,
    canonical_render_lock, _acquire_posthoc_gpu_lock, _free_port, _scene_ply,
    gs_server_environment, replay_worker_environment, _start_gs_server,
    _stop_process, _capture_scene_inputs, _current_gs_registration,
    _registration_scene_signature, _validated_scene_identity,
    _validate_policy_action_replay_cadence, validate_rerender_output,
    _sanitized_replay_argv, write_rerender_provenance, discover_runs,
    update_outputs_csv_after_rerender, _recorded_scene_dir, build_scene_grids,
    _select_run_paths, prepare_selected_sources, _run_rerender,
    build_arg_parser, main,
)

if __name__ == "__main__":
    raise SystemExit(main())
