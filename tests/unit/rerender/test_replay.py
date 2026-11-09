# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0


import json

from pathlib import Path


import cv2

import numpy as np

import pytest



import robosuite_env.tools.rerender_third_person as rerender







from robosuite_env.rendering.wrist_render_profiles import FRANKA_PINHOLE_WRIST_RGB_PROFILE
from robosuite_env.rollout.cli import parse
from robosuite_env.rollout.replay_policy_insets import (
    ReplayPolicyInsetError,
    load_replay_policy_insets,
)

from tests._helpers.rollouts import (
    LEGACY_WORKER_HEAD,
    MODULE_WORKER_HEAD,
    _configure_fr3_policy_action_source,
    _fr3_policy_action_summary,
    replayed_worker_arguments,
    run_dir,  # noqa: F401 -- pytest fixture
    use_module_worker_head,
)

@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["sim.py", "--frames", "100", "--frames", "12"], "12"),
        (["sim.py", "--frames", "100", "--frames=13"], "13"),
        (["sim.py", "--frames=100", "--frames", "14"], "14"),
        (["sim.py", "--frames"], None),
    ],
)
def test_last_option_value_uses_final_equals_or_separate_value(
    argv: list[str], expected: str | None
) -> None:
    if expected is None:
        with pytest.raises(rerender.RerenderError, match="missing its value"):
            rerender._last_option_value(argv, "--frames")
    else:
        assert rerender._last_option_value(argv, "--frames") == expected

def _fr3_policy_action_argv(argv: list[str]) -> list[str]:
    return argv + [
        "--physics=dynamic",
        "--policy-action-hz=30",
        "--frame-hz=0",
        "--dynamic-control-repeats=0",
        "--dynamic-control-hz=50",
        "--frames=300",
    ]

def test_build_replay_argv_converts_policy_action_clock_without_time_stretch(
    run_dir: Path,
) -> None:
    _, recorded = rerender.parse_run_command(run_dir)
    replay = rerender.build_replay_argv(
        _fr3_policy_action_argv(recorded),
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=_fr3_policy_action_summary(),
    )

    assert not any(
        token == "--policy-action-hz" or token.startswith("--policy-action-hz=")
        for token in replay
    )
    assert replay.count("--frame-hz") == 1
    assert replay[replay.index("--frame-hz") + 1] == "30"
    assert replay.count("--dynamic-control-repeats") == 1
    assert replay[replay.index("--dynamic-control-repeats") + 1] == "1"

    contract = rerender._fr3_policy_action_playback_contract(
        _fr3_policy_action_argv(recorded), _fr3_policy_action_summary()
    )
    assert contract["replay_expected_frames"] == 300
    assert contract["replay_output_fps"] == 30
    assert contract["replay_expected_duration_s"] == 10.0


def test_build_replay_argv_keeps_selected_box_layout_provenance(run_dir: Path) -> None:
    _, recorded = rerender.parse_run_command(run_dir)
    selected = "/layouts/pen-offset.json"
    replay = rerender.build_replay_argv(
        [
            *recorded,
            "--box-pack-layout-json",
            selected,
            "--box-pack-layout-index",
            "7",
        ],
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
    )
    assert replay[replay.index("--box-pack-layout-json") + 1] == selected
    assert replay[replay.index("--box-pack-layout-index") + 1] == "7"


def test_build_replay_argv_starts_the_worker_module_for_either_recorded_head(
    run_dir: Path,
) -> None:
    """A run recorded with the file-path head replays like a module-head run."""

    def replay_of(recorded: list[str]) -> list[str]:
        return rerender.build_replay_argv(
            recorded,
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
        )

    _, legacy = rerender.parse_run_command(run_dir)
    assert legacy[:2] == LEGACY_WORKER_HEAD.split()
    use_module_worker_head(run_dir)
    _, current = rerender.parse_run_command(run_dir)
    assert current[:3] == MODULE_WORKER_HEAD.split()
    assert legacy[2:] == current[3:]

    legacy_replay = replay_of(legacy)
    assert legacy_replay[:3] == MODULE_WORKER_HEAD.split()
    assert legacy_replay == replay_of(current)
    assert legacy_replay[3] == "--robot"
    assert not any(token.endswith("sim.py") for token in legacy_replay)




def test_build_replay_argv_keeps_legacy_frame_clock_unchanged(run_dir: Path) -> None:
    _, recorded = rerender.parse_run_command(run_dir)
    recorded += ["--frame-hz", "10", "--dynamic-control-repeats", "5"]

    replay = rerender.build_replay_argv(
        recorded,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary={"robot": "fr3"},
    )

    assert replay[replay.index("--frame-hz") + 1] == "10"
    assert replay[replay.index("--dynamic-control-repeats") + 1] == "5"


@pytest.mark.parametrize(
    ("recorded_flag", "summary", "expected"),
    [
        ((), None, "direct_kb4"),
        ((), {"policy_wrist_backend": "direct_kb4"}, "direct_kb4"),
        (("--policy-wrist-backend=direct_kb4",), {"robot": "fr3"}, "direct_kb4"),
    ],
)
def test_build_replay_argv_keeps_the_recorded_wrist_backend(
    run_dir: Path, recorded_flag: tuple[str, ...], summary: dict | None, expected: str | None
) -> None:
    from robosuite_env.rollout.cli import parse

    if summary is None and expected == "direct_kb4":
        row = {"frame": 0, "policy": {"obs_log": {"policy_wrist_backend": "direct_kb4"}}}
        (run_dir / "logs" / "sim_log.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    _, recorded = rerender.parse_run_command(run_dir)
    replay = rerender.build_replay_argv(
        recorded + list(recorded_flag),
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=summary,
    )

    backends = [t for t in replay if t.partition("=")[0] == "--policy-wrist-backend"]
    assert len(backends) == (0 if expected is None else 1)
    worker = parse(replayed_worker_arguments(replay))
    assert worker.policy_wrist_backend == (expected or "direct_kb4")

@pytest.mark.parametrize(
    ("recorded_flag", "summary", "expected"),
    [
        ((), None, "legacy"),  # a run recorded before the option: GaussianWrapping's fixed 0.2 GS-unit cull
        ((), {"policy_wrist_backend": "direct_kb4"}, "legacy"),
        ((), {"gs_kb4_near_cull": "znear"}, "znear"),
        (("--gs-kb4-near-cull=legacy",), {"gs_kb4_near_cull": "znear"}, "legacy"),
        # The launcher always passes the flag: a run that stopped before recording the mode keeps znear.
        (("--gs-kb4-near-cull", "znear"), None, "znear"),
    ],
)
def test_build_replay_argv_keeps_the_recorded_kb4_near_cull(
    run_dir: Path, recorded_flag: tuple[str, ...], summary: dict | None, expected: str
) -> None:
    from robosuite_env.rollout.cli import parse

    _, recorded = rerender.parse_run_command(run_dir)
    replay = rerender.build_replay_argv(
        recorded + list(recorded_flag),
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=summary,
    )
    assert len([t for t in replay if t.partition("=")[0] == "--gs-kb4-near-cull"]) == 1
    assert parse(replayed_worker_arguments(replay)).gs_kb4_near_cull == expected


def test_build_replay_argv_refuses_a_null_wrist_backend(run_dir: Path) -> None:
    (run_dir / "logs" / "sim_log.jsonl").write_text('{"frame": 0}\n', encoding="utf-8")
    _, recorded = rerender.parse_run_command(run_dir)
    with pytest.raises(rerender.RerenderError, match="policy_wrist_backend"):
        rerender.build_replay_argv(
            recorded,
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
            source_summary={"policy_wrist_backend": None},
        )


@pytest.mark.parametrize(
    ("recorded_flag", "summary", "log_row"),
    [
        ((), {"policy_wrist_backend": "retired_backend"}, {"frame": 0}),
        ((), None, {"frame": 0, "policy": {"obs_log": {"policy_wrist_backend": "retired_backend"}}}),
        ((), {"robot": "fr3"}, {"frame": 0}),
        ((), None, {"frame": 0}),
        (("--policy-wrist-backend", "retired_backend"), {"policy_wrist_backend": "retired_backend"}, {"frame": 0}),
        (("--policy-wrist-backend=retired_backend",), None, {"frame": 0}),
    ],
    ids=("summary", "observation", "summary-without-backend", "nothing-recorded", "command", "command-equals"),
)
def test_build_replay_argv_refuses_a_run_without_a_supported_wrist_backend(
    run_dir: Path, recorded_flag: tuple[str, ...], summary: dict | None, log_row: dict
) -> None:
    # A run that records no backend, or one this checkout does not render, is refused instead of
    # being rerendered with direct_kb4.
    from robosuite_env.rollout.replay.recorded_wrist import UNSUPPORTED_RECORDED_WRIST_BACKEND

    (run_dir / "logs" / "sim_log.jsonl").write_text(json.dumps(log_row) + "\n", encoding="utf-8")
    _, argv = rerender.parse_run_command(run_dir)
    with pytest.raises(rerender.RerenderError) as raised:
        rerender.build_replay_argv(
            argv + list(recorded_flag),
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
            source_summary=summary,
        )
    assert str(raised.value).endswith(UNSUPPORTED_RECORDED_WRIST_BACKEND)
    assert "retired_backend" not in str(raised.value)


@pytest.mark.parametrize(
    ("failure", "field", "value", "match"),
    [
        *[
            ("missing", key, None, "source summary")
            for key in (
                "policy_action_hz_effective",
                "frame_hz_effective",
                "frame_dt_s",
                "output_fps",
                "policy_action_dt_s",
                "policy_action_dt_nominal_s",
                "playback_action_dt_s",
                "fr3_dynamic_source_duration_s",
                "fr3_dynamic_runtime_tick_count_matches_schedule",
            )
        ],
        *[
            ("disagreement", field, value, None)
            for field, value in (
                ("policy_action_hz_effective", 10.0),
                ("frame_hz_effective", 29.0),
                ("output_fps", 29),
                ("policy_action_dt_s", 0.1),
                ("playback_action_dt_s", 0.1),
                ("fr3_dynamic_control_tick_count_total", 499),
                ("fr3_dynamic_boundary_phase_error_s_max_abs", 0.02),
            )
        ],
        ("absent", None, None, "requires the source"),
    ],
)
def test_policy_action_replay_rejects_missing_source_cadence_field(
    run_dir: Path,
    failure: str,
    field: str | None,
    value: object,
    match: str | None,
) -> None:
    _, recorded = rerender.parse_run_command(run_dir)
    summary = _fr3_policy_action_summary()
    if failure == "missing":
        assert field is not None
        summary.pop(field)
    elif failure == "disagreement":
        assert field is not None
        summary[field] = value

    source_kwargs = {} if failure == "absent" else {"source_summary": summary}
    with pytest.raises(rerender.RerenderError, match=match):
        rerender.build_replay_argv(
            _fr3_policy_action_argv(recorded),
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
            **source_kwargs,
        )

def test_prepare_source_run_binds_validated_policy_action_playback_contract(
    run_dir: Path,
) -> None:
    _configure_fr3_policy_action_source(run_dir)

    source = rerender.prepare_source_run(
        run_dir,
        renderer_code={"commit": "a" * 40, "dirty": True},
    )

    contract = source.source_records["policy_action_playback_contract"]
    assert contract["source_policy_action_hz"] == 30.0
    assert contract["replay_output_fps"] == 30
    assert contract["replay_expected_frames"] == 300
    assert contract["replay_expected_duration_s"] == 10.0


def test_build_replay_argv_refuses_unknown_task(
    run_dir: Path,
) -> None:
    """Unknown input fails before a replay output directory is created."""

    _, recorded = rerender.parse_run_command(run_dir)

    with pytest.raises(
        rerender.RerenderError,
        match="unknown task.objects 'unknown_task'",
    ):
        rerender.build_replay_argv(
            [*recorded, "--objects=unknown_task"],
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
        )
    assert not (run_dir / "rerender").exists()


@pytest.mark.parametrize("recorded_count", (0, 1, 2))
def test_build_replay_argv_forces_exactly_one_deferred_overview(
    run_dir: Path,
    recorded_count: int,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    argv = [token for token in argv if token != "--defer-overview-render"]
    argv.extend(["--defer-overview-render"] * recorded_count)

    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
    )

    assert replay.count("--state-log-replay") == 1
    assert replay[replay.index("--state-log-replay") + 1] == str(
        run_dir / "logs" / "sim_log.jsonl"
    )
    assert replay.count("--defer-overview-render") == 1
    assert "--policy-batch-server" not in replay
    assert "--policy-batch-client-id" not in replay
    assert "--policy-batch-timeout-sec" not in replay
    assert "--no-render" not in replay
    assert replay[replay.index("--gs-port") + 1] == "9999"
    assert replay[replay.index("--gs-server-token") + 1] == "feedface"
    assert replay.count("--out") == 1
    assert replay[replay.index("--out") + 1] == str(
        (run_dir / "scene_run_01.mp4").resolve()
    )
    assert replay[replay.index("--logdir") + 1] == str(
        run_dir / "rerender" / "logs"
    )
    assert replay[replay.index("--summary-path") + 1] == str(
        run_dir / "rerender" / "sim_summary.json"
    )
    assert "--policy-obs-inset" not in replay
    assert "--wrist-gs" not in replay
    assert replay.count("--no-wrist") == 1
    assert replay[replay.index("--max-wall-sec") + 1] == "0"
    assert replay[replay.index("--dynamic-subframes") + 1] == "1"
    assert replay[replay.index("--init-state-in") + 1] == ""

def test_build_replay_argv_drops_generation_only_options_and_equals_forms(
    run_dir: Path,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    argv.extend(
        [
            "--policy=/checkpoints/policy",
            "--grasp-test",
            "--rf-overlap=4",
            "--rf-overlap-sampler=paired",
            "--rf-overlap-prev-weight=0.5",
            "--rf-overlap-step-shift=1",
            "--policy-image-source=lerobot",
            "--policy-pose-source=lerobot",
            "--policy-image-episode=/data/episode.parquet",
            "--policy-image-loop",
            "--init-state-in=/runs/init.json",
            "--mimicgen-init-episode=/data/init.parquet",
            "--max-wall-sec=30",
            "--dynamic-subframes=4",
            "--official-run-manifest=/runs/run_manifest.json",
            "--official-worker-contract=/runs/worker_contract.json",
            "--official-scene-tag",
            "scene",
            "--strict-assets",
            "--blender-export",
            "/x",
            "--arm-ctrl",
            "pin",
            "--ee-ctrl",
            "weld",
            "--fr3-contact-control",
            "off",
            "--ik-url",
            "http://127.0.0.1:8000",
            "--out=/snapshot/scene/run_01/scene_run_01.mp4",
        ]
    )

    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
    )

    removed_prefixes = (
        "--policy",
        "--grasp-test",
        "--rf-overlap",
        "--rf-overlap-sampler",
        "--rf-overlap-prev-weight",
        "--rf-overlap-step-shift",
        "--policy-image-source",
        "--policy-pose-source",
        "--policy-image-episode",
        "--mimicgen-init-episode",
        "--official-run-manifest",
        "--official-worker-contract",
        "--official-scene-tag",
        "--strict-assets",
        "--blender-export",
        "--arm-ctrl",
        "--ee-ctrl",
        "--fr3-contact-control",
        "--ik-url",
    )
    assert not any(
        token == prefix or token.startswith(f"{prefix}=")
        for token in replay
        for prefix in removed_prefixes
    )
    assert replay.count("--max-wall-sec") == 1
    assert replay[replay.index("--max-wall-sec") + 1] == "0"
    assert replay.count("--dynamic-subframes") == 1
    assert replay[replay.index("--dynamic-subframes") + 1] == "1"
    assert replay.count("--init-state-in") == 1
    assert replay[replay.index("--init-state-in") + 1] == ""
    assert replay.count("--out") == 1
    assert replay[replay.index("--out") + 1] == str(
        (run_dir / "scene_run_01.mp4").resolve()
    )
    assert not any(
        token in replay
        for token in (
            "/x",
            "pin",
            "weld",
            "off",
            "http://127.0.0.1:8000",
            "/runs/run_manifest.json",
            "/runs/worker_contract.json",
            "scene",
        )
    )

@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (None, FRANKA_PINHOLE_WRIST_RGB_PROFILE),
        ({"robot": "fr3"}, FRANKA_PINHOLE_WRIST_RGB_PROFILE),
        # main's summary of a run without a policy
        ({"policy_yubi_render_profile": None}, FRANKA_PINHOLE_WRIST_RGB_PROFILE),
        ({"policy_yubi_render_profile": "elp-mean-wrist-rgb"}, "elp-mean-wrist-rgb"),
    ],
)
def test_build_replay_argv_gives_a_run_without_a_recorded_profile_the_legacy_camera(
    run_dir: Path, summary: dict | None, expected: str
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=summary,
    )
    assert parse(replayed_worker_arguments(replay)).policy_yubi_render_profile == expected


def test_build_replay_argv_refuses_a_null_profile_from_a_summary_with_wrist_intrinsics(run_dir: Path) -> None:
    # A summary with policy_wrist_intrinsics but a null profile names neither camera, so its run may have used either.
    _, argv = rerender.parse_run_command(run_dir)
    with pytest.raises(rerender.RerenderError, match="wrist camera is unknown"):
        rerender.build_replay_argv(
            argv,
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets="off",
            source_summary={"policy_yubi_render_profile": None, "policy_wrist_intrinsics": None},
        )


def test_build_replay_argv_reads_the_profile_from_the_state_log_without_a_summary(
    run_dir: Path,
) -> None:
    (run_dir / "logs" / "sim_log.jsonl").write_text(
        json.dumps({"frame": 0})
        + "\n\n"
        + json.dumps({"frame": 1, "policy": {"obs_log": {
            "policy_yubi_render_profile": FRANKA_PINHOLE_WRIST_RGB_PROFILE, "policy_wrist_backend": "direct_kb4",
        }}})
        + "\n",
        encoding="utf-8",
    )
    _, argv = rerender.parse_run_command(run_dir)
    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        source_summary=None,
    )
    assert parse(replayed_worker_arguments(replay)).policy_yubi_render_profile == FRANKA_PINHOLE_WRIST_RGB_PROFILE


@pytest.mark.parametrize("saved_wrist_display", (None, "raw224"))
def test_build_replay_argv_uses_saved_insets_without_live_wrist_renderers(
    run_dir: Path,
    saved_wrist_display: str | None,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    policy_obs_dir = run_dir / "logs" / "policy_obs"
    policy_obs_dir.mkdir()

    display_kwargs = (
        {}
        if saved_wrist_display is None
        else {"saved_wrist_display": saved_wrist_display}
    )
    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="saved",
        policy_obs_dir=policy_obs_dir,
        **display_kwargs,
    )

    assert "--policy-obs-inset" not in replay
    assert "--wrist-gs" not in replay
    assert "--no-wrist" not in replay
    assert "--defer-overview-render" in replay
    assert replay[replay.index("--state-replay-policy-obs-dir") + 1] == str(
        policy_obs_dir
    )
    assert replay[replay.index("--state-replay-policy-obs-display") + 1] == (
        "raw224"
    )

@pytest.mark.parametrize("case", ("all", "partial", "recorded", "unknown"))
def test_build_replay_argv_replaces_recorded_view_flags_with_overrides(
    run_dir: Path,
    case: str,
) -> None:
    _, argv = rerender.parse_run_command(run_dir)
    if case == "all":
        argv.extend(
            [
                "--view-azimuth",
                "130",
                "--view-elevation=-22",
                "--view-distance",
                "2.0",
                "--view-lookat=0.1 0.2 0.3",
            ]
        )
        overrides = {
            "--view-azimuth": "45",
            "--view-elevation": "-10",
            "--view-distance": "1.5",
            "--view-lookat": "0.4 0.5 0.6",
        }
    elif case == "partial":
        argv.extend(["--view-azimuth=130", "--view-elevation", "-22"])
        overrides = {"--view-azimuth": "45"}
    elif case == "recorded":
        argv.extend(["--view-azimuth=130", "--view-lookat", "0.1 0.2 0.3"])
        overrides = None
    else:
        overrides = {"--view-fov": "60"}

    kwargs = {} if overrides is None else {"view_overrides": overrides}
    if case == "unknown":
        with pytest.raises(rerender.RerenderError, match="unknown view override"):
            rerender.build_replay_argv(
                argv,
                run_dir=run_dir,
                gs_port=9999,
                gs_token="feedface",
                render_lock="/tmp/rerender.lock",
                wrist_insets="off",
                **kwargs,
            )
        return

    replay = rerender.build_replay_argv(
        argv,
        run_dir=run_dir,
        gs_port=9999,
        gs_token="feedface",
        render_lock="/tmp/rerender.lock",
        wrist_insets="off",
        **kwargs,
    )
    if case == "all":
        for option, value in (
            ("--view-azimuth", "45"),
            ("--view-elevation", "-10"),
            ("--view-distance", "1.5"),
            ("--view-lookat", "0.4 0.5 0.6"),
        ):
            assert replay.count(option) == 1
            assert replay[replay.index(option) + 1] == value
        assert not any(
            token.startswith("--view-") and "=" in token for token in replay
        )
        assert "130" not in replay
        assert "0.1 0.2 0.3" not in replay
    elif case == "partial":
        assert replay.count("--view-azimuth") == 1
        assert replay[replay.index("--view-azimuth") + 1] == "45"
        assert "--view-azimuth=130" not in replay
        assert replay[replay.index("--view-elevation") + 1] == "-22"
    else:
        assert "--view-azimuth=130" in replay
        assert replay[replay.index("--view-lookat") + 1] == "0.1 0.2 0.3"

@pytest.mark.parametrize("lookat", ("0.1 0.2", "a b c", "1 2 3 4", "nan nan nan"))
def test_resolve_view_overrides_rejects_malformed_lookat(lookat: str) -> None:
    arguments = rerender.argparse.Namespace(
        view_azimuth=45.0,
        view_elevation=None,
        view_distance=1.5,
        view_lookat="0.1, 0.2, 0.3",
    )
    assert rerender.resolve_view_overrides(arguments) == {
        "--view-azimuth": "45",
        "--view-distance": "1.5",
        "--view-lookat": "0.1 0.2 0.3",
    }
    assert rerender.resolve_view_overrides(rerender.argparse.Namespace()) == {}
    assert rerender.resolve_view_overrides(
        rerender.argparse.Namespace(view_lookat="")
    ) == {"--view-lookat": ""}

    with pytest.raises(rerender.RerenderError, match="three numbers"):
        rerender.resolve_view_overrides(
            rerender.argparse.Namespace(view_lookat=lookat)
        )

@pytest.mark.parametrize("legacy_mode", ("auto", "render"))
def test_wrist_cli_defaults_saved_and_rejects_removed_fallback_modes(
    run_dir: Path,
    legacy_mode: str,
) -> None:
    parser = rerender.build_arg_parser()
    defaults = parser.parse_args([str(run_dir.parents[1])])
    assert defaults.wrist_insets == "saved"
    assert defaults.saved_wrist_display == "raw224"
    assert defaults.gs_port == 0
    assert parser.parse_args(
        [str(run_dir.parents[1]), "--gs-port", "19002"]
    ).gs_port == 19002
    assert parser.parse_args(
        [str(run_dir.parents[1]), "--wrist-insets", "off"]
    ).wrist_insets == "off"
    assert parser.parse_args(
        [str(run_dir.parents[1]), "--saved-wrist-display", "raw224"]
    ).saved_wrist_display == "raw224"

    # All three public layers reject the removed fallback modes.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [str(run_dir.parents[1]), "--wrist-insets", legacy_mode]
        )
    with pytest.raises(rerender.RerenderError, match="unknown wrist inset mode"):
        rerender.resolve_wrist_insets(run_dir, legacy_mode)

    _, argv = rerender.parse_run_command(run_dir)
    argv.append("--no-wrist")
    with pytest.raises(rerender.RerenderError, match="unknown resolved wrist"):
        rerender.build_replay_argv(
            argv,
            run_dir=run_dir,
            gs_port=9999,
            gs_token="feedface",
            render_lock="/tmp/rerender.lock",
            wrist_insets=legacy_mode,
        )

@pytest.mark.parametrize(
    ("runs", "match"),
    [
        ("run_01,run_01", "duplicate"),
        ("run_01,", "non-empty"),
        (",run_01", "non-empty"),
        ("one", "run_<integer>"),
    ],
)
def test_select_run_paths_rejects_ambiguous_run_filters(
    run_dir: Path, runs: str, match: str
) -> None:
    arguments = rerender.argparse.Namespace(
        run_root=run_dir.parents[1], runs=runs
    )
    with pytest.raises(rerender.RerenderError, match=match):
        rerender._select_run_paths(arguments)

def _write_policy_image(path: Path, value: int) -> None:
    image = np.full((224, 224, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

def test_resolve_wrist_insets_saved_selects_complete_sequence(
    run_dir: Path,
) -> None:
    policy_obs_dir = run_dir / "logs" / "policy_obs"
    policy_obs_dir.mkdir()
    left = policy_obs_dir / "obs_L_f0000.png"
    right = policy_obs_dir / "obs_R_f0000.png"
    _write_policy_image(left, 10)
    _write_policy_image(right, 20)
    row = {
        "policy": {
            "observation_frame": 0,
            "obs_log": {
                "frame": 0,
                "observation_frame": 0,
                "paths": {"primary_L": str(left), "primary_R": str(right)},
                "policy_image_resize": "pad",
                "image_source_shapes": {
                    "L": [480, 640, 3],
                    "R": [480, 640, 3],
                },
            },
        }
    }
    (run_dir / "logs" / "sim_log.jsonl").write_text(
        f"{json.dumps(row)}\n{json.dumps(row)}\n",
        encoding="utf-8",
    )

    mode, directory, unique_pairs, fallback = rerender.resolve_wrist_insets(
        run_dir, "saved"
    )

    assert mode == "saved"
    assert directory == policy_obs_dir.resolve()
    assert unique_pairs == 1
    assert fallback is None

def test_resolve_wrist_insets_saved_fails_closed_and_off_is_explicit(
    run_dir: Path,
) -> None:
    with pytest.raises(rerender.RerenderError, match="validation failed"):
        rerender.resolve_wrist_insets(run_dir, "saved")
    assert rerender.resolve_wrist_insets(run_dir, "off") == (
        "off",
        None,
        0,
        None,
    )

@pytest.mark.parametrize("record_source", ("summary", "command"))
def test_resolve_wrist_insets_rejects_stale_rf_overlap_mapping(
    run_dir: Path,
    record_source: str,
) -> None:
    if record_source == "summary":
        (run_dir / "sim_summary.json").write_text(
            json.dumps({"rf_overlap": 8}), encoding="utf-8"
        )
        match = "differs from recorded command"
    else:
        (run_dir / "sim_summary.json").unlink()
        command = (run_dir / "RUN_COMMAND.md").read_text(encoding="utf-8")
        command = command.replace("--no-render", "--no-render --rf-overlap 8")
        (run_dir / "RUN_COMMAND.md").write_text(command, encoding="utf-8")
        match = "rf_overlap is 8"

    with pytest.raises(rerender.RerenderError, match=match):
        rerender.resolve_wrist_insets(run_dir, "saved")

def test_build_replay_argv_requires_state_log(run_dir: Path) -> None:
    (run_dir / "logs" / "sim_log.jsonl").unlink()
    _, argv = rerender.parse_run_command(run_dir)
    with pytest.raises(rerender.RerenderError, match="missing"):
        rerender.build_replay_argv(
            argv,
            run_dir=run_dir,
            gs_port=1,
            gs_token="x",
            render_lock="/tmp/l",
        )

def _write_image(path: Path, bgr: tuple[int, int, int], size: int = 224) -> None:
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = bgr
    assert cv2.imwrite(str(path), image)

def _row(
    left: str,
    right: str,
    *,
    resize: str = "pad",
    source_shapes: dict | None = None,
    observation_frame: int = 0,
) -> dict:
    obs_log = {
        "paths": {"primary_L": left, "primary_R": right},
        "image_source": "sim",
        "policy_image_resize": resize,
        "policy_yubi_render_profile": "franka-pinhole-wrist-rgb",
        "frame": observation_frame,
        "observation_frame": observation_frame,
    }
    if source_shapes is not None:
        obs_log["image_source_shapes"] = source_shapes
    return {"policy": {"observation_frame": observation_frame, "obs_log": obs_log}}

def test_load_replay_policy_insets_holds_and_caches_active_pair(
    tmp_path: Path,
) -> None:
    obs_dir = tmp_path / "policy_obs"
    obs_dir.mkdir()
    left0 = obs_dir / "obs_L_f0000.png"
    right0 = obs_dir / "obs_R_f0000.png"
    left1 = obs_dir / "obs_L_f0002.png"
    right1 = obs_dir / "obs_R_f0002.png"
    _write_image(left0, (1, 2, 3))
    _write_image(right0, (4, 5, 6))
    _write_image(left1, (7, 8, 9))
    _write_image(right1, (10, 11, 12))

    schedule = load_replay_policy_insets(
        [
            _row(str(left0), str(right0)),
            _row(str(left0), str(right0)),
            _row(str(left1), str(right1), observation_frame=2),
        ],
        obs_dir,
    )

    assert len(schedule.frames) == 3
    assert schedule.unique_pairs == 2
    assert schedule.frames[0] is schedule.frames[1]
    assert schedule.frames[0][0][0, 0].tolist() == [3, 2, 1]
    assert schedule.frames[2][1][0, 0].tolist() == [12, 11, 10]
    assert schedule.raw_display_frames[0] is schedule.raw_display_frames[1]
    assert schedule.raw_display_frames[0][0].shape == (135, 180, 3)
    assert schedule.display_frames[0] is schedule.display_frames[1]
    assert schedule.observation_frames == (0, 0, 2)
    assert schedule.display_frames[0][0].shape == (135, 180, 3)
    transform = schedule.display_transforms[0][0]
    assert transform.source_hw == (539, 719)
    assert transform.policy_content_xywh == (0, 28, 224, 167)

def test_load_replay_policy_insets_raw_display_preserves_letterbox_bands(
    tmp_path: Path,
) -> None:
    left = tmp_path / "obs_L_f0000.png"
    right = tmp_path / "obs_R_f0000.png"
    letterboxed = np.zeros((224, 224, 3), dtype=np.uint8)
    letterboxed[28:195] = (20, 120, 240)
    assert cv2.imwrite(str(left), letterboxed)
    assert cv2.imwrite(str(right), letterboxed)

    schedule = load_replay_policy_insets(
        [_row(str(left), str(right))],
        tmp_path,
    )

    raw_left, raw_right = schedule.raw_display_frames[0]
    assert raw_left.shape == (135, 180, 3)
    assert raw_right.shape == (135, 180, 3)
    assert not raw_left.flags.writeable
    assert not raw_right.flags.writeable
    assert np.count_nonzero(raw_left[:16]) == 0
    assert np.count_nonzero(raw_left[-16:]) == 0
    assert np.count_nonzero(raw_left[18:-18]) > 0
    assert np.array_equal(raw_left, raw_right)

def test_load_replay_policy_insets_relocates_stale_absolute_paths(
    tmp_path: Path,
) -> None:
    obs_dir = tmp_path / "copied" / "policy_obs"
    obs_dir.mkdir(parents=True)
    left = obs_dir / "obs_L_f0000.png"
    right = obs_dir / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))

    schedule = load_replay_policy_insets(
        [
            _row(
                "/old/run/logs/policy_obs/obs_L_f0000.png",
                "/old/run/logs/policy_obs/obs_R_f0000.png",
            )
        ],
        obs_dir,
    )

    assert schedule.unique_pairs == 1

def test_load_replay_policy_insets_never_falls_back_to_stale_tree(
    tmp_path: Path,
) -> None:
    authoritative = tmp_path / "current" / "policy_obs"
    authoritative.mkdir(parents=True)
    stale = tmp_path / "stale" / "policy_obs"
    stale.mkdir(parents=True)
    left = stale / "obs_L_f0000.png"
    right = stale / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))

    with pytest.raises(ReplayPolicyInsetError, match="authoritative directory"):
        load_replay_policy_insets(
            [_row(str(left), str(right))], authoritative
        )

@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{}], "policy.obs_log"),
        ([_row("obs_L_f0000.png", "obs_R_f0000.png")], "missing"),
    ],
)
def test_load_replay_policy_insets_rejects_incomplete_schedule(
    tmp_path: Path, rows: list[dict], match: str
) -> None:
    with pytest.raises(ReplayPolicyInsetError, match=match):
        load_replay_policy_insets(rows, tmp_path)

@pytest.mark.parametrize("kind", ["wrong_shape", "corrupt"])
def test_load_replay_policy_insets_rejects_invalid_image(
    tmp_path: Path, kind: str
) -> None:
    left = tmp_path / "obs_L_f0000.png"
    right = tmp_path / "obs_R_f0000.png"
    if kind == "wrong_shape":
        _write_image(left, (1, 2, 3), size=32)
    else:
        left.write_bytes(b"not a png")
    _write_image(right, (4, 5, 6))

    with pytest.raises(ReplayPolicyInsetError, match="decode|224x224"):
        load_replay_policy_insets([_row(str(left), str(right))], tmp_path)

def test_load_replay_policy_insets_uses_recorded_source_shape(tmp_path: Path) -> None:
    left = tmp_path / "obs_L_f0000.png"
    right = tmp_path / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))
    schedule = load_replay_policy_insets(
        [
            _row(
                str(left),
                str(right),
                source_shapes={"L": [300, 400, 3], "R": [600, 400, 3]},
            )
        ],
        tmp_path,
    )
    left_transform, right_transform = schedule.display_transforms[0]
    assert left_transform.policy_content_xywh == (0, 28, 224, 168)
    assert left_transform.output_content_xywh == (0, 0, 180, 135)
    assert right_transform.policy_content_xywh == (37, 0, 149, 224)
    assert right_transform.output_content_xywh == (45, 0, 90, 135)

def test_load_replay_policy_insets_fails_without_declared_geometry(
    tmp_path: Path,
) -> None:
    left = tmp_path / "obs_L_f0000.png"
    right = tmp_path / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))
    row = _row(str(left), str(right))
    row["policy"]["obs_log"]["image_source"] = "lerobot"
    row["policy"]["obs_log"].pop("policy_yubi_render_profile")
    with pytest.raises(ReplayPolicyInsetError, match="no image_source_shapes"):
        load_replay_policy_insets([row], tmp_path)

def test_load_replay_policy_insets_rejects_metadata_change_for_same_pair(
    tmp_path: Path,
) -> None:
    left = tmp_path / "obs_L_f0000.png"
    right = tmp_path / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))
    rows = [
        _row(str(left), str(right)),
        _row(
            str(left),
            str(right),
            source_shapes={"L": [300, 400, 3], "R": [300, 400, 3]},
        ),
    ]
    with pytest.raises(ReplayPolicyInsetError, match="changes display metadata"):
        load_replay_policy_insets(rows, tmp_path)

@pytest.mark.parametrize(
    ("left_name", "right_name", "observation_frame", "match"),
    [
        ("obs_R_f0000.png", "obs_R_f0000.png", 0, "primary_L"),
        ("obs_L_f0000.png", "obs_L_f0001.png", 0, "primary_R"),
        ("obs_L_f0016.png", "obs_R_f0016.png", 0, "filename frame 16"),
    ],
)
def test_load_replay_policy_insets_rejects_side_or_frame_filename_mismatch(
    tmp_path: Path,
    left_name: str,
    right_name: str,
    observation_frame: int,
    match: str,
) -> None:
    left = tmp_path / left_name
    right = tmp_path / right_name
    _write_image(left, (1, 2, 3))
    if right != left:
        _write_image(right, (4, 5, 6))
    row = _row(
        str(left), str(right), observation_frame=observation_frame
    )
    with pytest.raises(ReplayPolicyInsetError, match=match):
        load_replay_policy_insets([row], tmp_path)


# --------------------------------------------------------------------------
# Saved-policy-OBS schedule: early fail-closed branches and the stretch
# display contract.
# --------------------------------------------------------------------------


def test_load_replay_policy_insets_requires_an_existing_observation_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ReplayPolicyInsetError, match="saved policy observation directory does not exist"
    ):
        load_replay_policy_insets([], tmp_path / "missing")


def test_load_replay_policy_insets_rejects_an_empty_state_log(tmp_path: Path) -> None:
    obs_dir = tmp_path / "policy_obs"
    obs_dir.mkdir()

    with pytest.raises(ReplayPolicyInsetError, match="state log is empty"):
        load_replay_policy_insets([], obs_dir)


def test_load_replay_policy_insets_rejects_a_non_object_state_row(
    tmp_path: Path,
) -> None:
    obs_dir = tmp_path / "policy_obs"
    obs_dir.mkdir()

    with pytest.raises(
        ReplayPolicyInsetError, match="state-log frame 0 is not a JSON object"
    ):
        load_replay_policy_insets(["not-an-object"], obs_dir)


def test_load_replay_policy_insets_supports_the_stretch_display_contract(
    tmp_path: Path,
) -> None:
    obs_dir = tmp_path / "policy_obs"
    obs_dir.mkdir()
    left = obs_dir / "obs_L_f0000.png"
    right = obs_dir / "obs_R_f0000.png"
    _write_image(left, (1, 2, 3))
    _write_image(right, (4, 5, 6))
    row = _row(
        str(left),
        str(right),
        resize="stretch",
        source_shapes={"L": [300, 400, 3], "R": [300, 400, 3]},
    )

    schedule = load_replay_policy_insets([row], obs_dir)

    transform = schedule.display_transforms[0][0]
    assert transform.resize_mode == "stretch"
    assert transform.policy_content_xywh == (0, 0, 224, 224)
    assert schedule.display_frames[0][0].shape == (135, 180, 3)
    assert schedule.raw_display_frames[0][0].shape == (135, 180, 3)


def test_load_replay_policy_insets_rejects_a_backwards_observation_frame(
    tmp_path: Path,
) -> None:
    obs_dir = tmp_path / "policy_obs"
    obs_dir.mkdir()
    for frame in (0, 2):
        _write_image(obs_dir / f"obs_L_f{frame:04d}.png", (1, 2, 3))
        _write_image(obs_dir / f"obs_R_f{frame:04d}.png", (4, 5, 6))
    rows = [
        _row(
            str(obs_dir / "obs_L_f0002.png"),
            str(obs_dir / "obs_R_f0002.png"),
            observation_frame=2,
        ),
        _row(
            str(obs_dir / "obs_L_f0000.png"),
            str(obs_dir / "obs_R_f0000.png"),
            observation_frame=0,
        ),
    ]
    rows.insert(
        0,
        _row(
            str(obs_dir / "obs_L_f0000.png"),
            str(obs_dir / "obs_R_f0000.png"),
            observation_frame=0,
        ),
    )
    rows.insert(
        1,
        _row(
            str(obs_dir / "obs_L_f0000.png"),
            str(obs_dir / "obs_R_f0000.png"),
            observation_frame=0,
        ),
    )

    with pytest.raises(
        ReplayPolicyInsetError, match="moves policy observation backwards"
    ):
        load_replay_policy_insets(rows, obs_dir)
