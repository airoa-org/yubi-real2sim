# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc rerender orchestration across scenes and selected sources."""


from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from robosuite_env.rerender.batch import (
    _recorded_scene_dir,
    build_scene_grids,
    resolve_wrist_insets,
    update_outputs_csv_after_rerender,
)
from robosuite_env.rerender.common import (
    GpuBinding,
    REPO_ROOT,
    RerenderError,
    SourceRun,
    _canonical_video_path,
)
from robosuite_env.rerender.gpu import canonical_render_lock, resolve_gpu_binding
from robosuite_env.rerender.gs_server import (
    _capture_scene_inputs,
    _free_port,
    _start_gs_server,
    _stop_process,
    _validated_scene_identity,
    replay_worker_environment,
)
from robosuite_env.rerender.replay_argv import (
    build_replay_argv,
    resolve_view_overrides,
)
from robosuite_env.rerender.sources import prepare_selected_sources
from robosuite_env.rerender.validation import (
    validate_rerender_output,
    write_rerender_provenance,
)


def _run_rerender(
    arguments: argparse.Namespace,
    sources: list[SourceRun] | None = None,
    partial_selection: bool | None = None,
) -> int:
    """Run post-hoc rendering while the caller owns the GPU lifecycle lock."""

    run_root = arguments.run_root.resolve()
    if sources is None:
        sources, inferred_partial_selection = prepare_selected_sources(arguments)
        if partial_selection is None:
            partial_selection = inferred_partial_selection
    if partial_selection is None:
        partial_selection = False
    runs = [source.run_dir for source in sources]
    binding = getattr(arguments, "gpu_binding", None)
    if not isinstance(binding, GpuBinding):
        binding = resolve_gpu_binding(arguments.gpu)

    view_overrides = resolve_view_overrides(arguments)
    wrist_selections = {
        source.run_dir: resolve_wrist_insets(
            source.run_dir,
            arguments.wrist_insets,
            generation_summary=source.summary,
            state_rows=source.state_rows,
        )
        for source in sources
    }
    render_lock = canonical_render_lock(binding)

    scene_dirs: dict[str, Path] = {}
    expected_frames_by_scene: dict[str, int] = {}
    # Scenes whose rollouts each ended at the stage they reached, so there is
    # no single length to tile a composite grid on. Their per-rollout videos
    # are rendered and validated exactly like any other run.
    ragged_scenes: set[str] = set()
    for source in sources:
        run_dir = source.run_dir
        scene_tag = run_dir.parent.name
        scene_dir = _recorded_scene_dir(run_dir, source.argv)
        previous = scene_dirs.setdefault(scene_tag, scene_dir)
        if previous != scene_dir:
            raise RerenderError(
                f"{scene_tag}: inconsistent --gs-background paths: "
                f"{previous} != {scene_dir}"
            )
        previous_frames = expected_frames_by_scene.setdefault(
            scene_tag, source.expected_frames
        )
        if previous_frames != source.expected_frames:
            raise RerenderError(
                f"{scene_tag}: inconsistent configured frame counts: "
                f"{previous_frames} != {source.expected_frames}"
            )

    failures: list[str] = []
    for scene_index, (scene_tag, scene_dir) in enumerate(scene_dirs.items()):
        scene_sources = [
            source for source in sources if source.run_dir.parent.name == scene_tag
        ]
        python = scene_sources[0].argv[0]
        configured_port = int(getattr(arguments, "gs_port", 0))
        port = configured_port + scene_index if configured_port else _free_port()
        if not 1 <= port <= 65535:
            raise RerenderError(f"invalid GS port for scene {scene_tag}: {port}")
        token = hashlib.sha256(
            f"rerender:{run_root}:{scene_tag}".encode()
        ).hexdigest()
        selection_digest = hashlib.sha256(
            ",".join(source.run_dir.name for source in scene_sources).encode()
        ).hexdigest()[:10]
        manifest_path, manifest_record, ply_path, ply_state = _capture_scene_inputs(
            scene_dir
        )
        server, health = _start_gs_server(
            python,
            ply_path,
            port,
            token,
            run_root
            / (
                f"rerender_gs_{scene_tag}_gpu{arguments.gpu}_"
                f"{selection_digest}.log"
            ),
            binding,
            render_lock=render_lock,
        )
        try:
            scene_identity = _validated_scene_identity(
                scene_tag=scene_tag,
                scene_dir=scene_dir,
                sources=scene_sources,
                health=health,
                manifest_path=manifest_path,
                manifest_record=manifest_record,
                ply_path=ply_path,
                ply_state=ply_state,
            )
            for source in scene_sources:
                run_dir = source.run_dir
                argv = source.argv
                wrist_mode, policy_obs_dir, unique_pairs, _ = (
                    wrist_selections[run_dir]
                )
                replay = build_replay_argv(
                    argv,
                    run_dir=run_dir,
                    gs_port=port,
                    gs_token=token,
                    render_lock=render_lock,
                    wrist_insets=wrist_mode,
                    saved_wrist_display=getattr(
                        arguments, "saved_wrist_display", "raw224"
                    ),
                    policy_obs_dir=policy_obs_dir,
                    source_summary=source.summary,
                    view_overrides=view_overrides,
                )
                environment = replay_worker_environment(
                    binding, recorded=source.environment
                )
                selection_note = (
                    f"saved policy observations ({unique_pairs} unique pairs)"
                    if wrist_mode == "saved"
                    else wrist_mode
                )
                print(
                    f"[rerender] {run_dir.parent.name}/{run_dir.name} "
                    f"wrist-insets={selection_note}",
                    flush=True,
                )
                log_path = run_dir / "rerender" / "rerender_stdout.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as handle:
                    completed = subprocess.run(
                        replay,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                if completed.returncode != 0:
                    failures.append(
                        f"{run_dir}: replay exited {completed.returncode}; "
                        f"see {log_path}"
                    )
                    continue
                out_path = _canonical_video_path(run_dir)
                if not out_path.is_file() or out_path.stat().st_size <= 0:
                    failures.append(f"{run_dir}: replay wrote no video {out_path}")
                    continue
                try:
                    validation = validate_rerender_output(
                        source,
                        out_path,
                        expected_wrist_insets=wrist_mode,
                    )
                    write_rerender_provenance(
                        source=source,
                        replay=replay,
                        gs_token=token,
                        scene=scene_identity,
                        video_path=out_path,
                        validation=validation,
                        view_overrides=view_overrides,
                    )
                except RerenderError as exc:
                    failures.append(str(exc))
        finally:
            _stop_process(server)

    if failures:
        for failure in failures:
            print(f"[rerender] FAILED {failure}", file=sys.stderr, flush=True)
        return 1

    try:
        outputs_updated = update_outputs_csv_after_rerender(run_root, runs)
    except RerenderError as exc:
        print(f"[rerender] FAILED {exc}", file=sys.stderr, flush=True)
        return 1
    if outputs_updated:
        print(
            f"[rerender] updated validated artifact index: "
            f"{run_root / 'outputs.csv'}",
            flush=True,
        )

    if not arguments.skip_grid:
        grid_failures = build_scene_grids(
            run_root,
            runs,
            expected_frames_by_scene,
            grid_template=getattr(arguments, "grid_template", None),
            ragged_scenes=ragged_scenes,
        )
        if grid_failures:
            for failure in grid_failures:
                print(f"[rerender] FAILED {failure}", file=sys.stderr, flush=True)
            return 1
    print(f"[rerender] complete: {len(runs)} runs", flush=True)
    return 0
