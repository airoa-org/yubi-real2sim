# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from robosuite_env.policy.policy_io import prepare_policy_wrist_display_image
from robosuite_env.rollout.state.render_state import (
    build_render_state_contract,
    build_render_state_summary_contract,
    encode_render_state_row,
)


SAVED_WRIST_LEFT_XYWH = (4, 4, 180, 135)
SAVED_WRIST_RIGHT_XYWH = (188, 4, 180, 135)
RAW224_DISPLAY_CONTRACT = "raw_policy_tensor_full_canvas_resize_v1"
CROPPED_DISPLAY_CONTRACT = "metadata_unpad_aspect_fit_v1"


def saved_policy_obs_tensor(seed: int, side_index: int) -> np.ndarray:
    """A deterministic, non-symmetric 224x224 RGB policy observation.

    The gradients stay low-frequency so a lossy MP4 round trip keeps them
    within the exact saved-wrist thresholds, while the solid quadrant makes a
    side swap or a stale observation an unmistakable error.
    """

    step = seed % 8
    yy, xx = np.indices((224, 224))
    image = np.stack(
        (
            20 + xx // 4 + step * 3 + side_index * 7,
            30 + yy // 4 + step * 2,
            40 + (xx + yy) // 8 + side_index * 11,
        ),
        axis=-1,
    ).astype(np.uint8)
    image[:112, :112] = (30 + step * 25, 200 - step * 20, 90 + side_index * 90)
    return image


def _write_saved_policy_obs(
    run_dir: Path, frames: int, index: int, exec_steps: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    """Save the 224px policy OBS a rollout would have written per inference."""

    logs = run_dir / "logs"
    obs_dir = logs / "policy_obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    observation_frames = list(range(0, frames, exec_steps))
    saved: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order, observation_frame in enumerate(observation_frames):
        pair = []
        for side_index, side in enumerate(("L", "R")):
            tensor = saved_policy_obs_tensor(index * 7 + order, side_index)
            path = obs_dir / f"obs_{side}_f{observation_frame:04d}.png"
            assert cv2.imwrite(
                str(path), cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
            )
            pair.append(tensor)
        saved[observation_frame] = (pair[0], pair[1])

    rows: list[dict] = []
    schedule: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_index in range(frames):
        observation_frame = observation_frames[frame_index // exec_steps]
        rows.append(
            {
                "frame": frame_index,
                "policy": {
                    "observation_frame": observation_frame,
                    "obs_log": {
                        "frame": observation_frame,
                        "observation_frame": observation_frame,
                        "policy_image_resize": "pad",
                        "image_source_shapes": {
                            "L": [539, 719, 3],
                            "R": [539, 719, 3],
                        },
                        "paths": {
                            "primary_L": str(
                                obs_dir / f"obs_L_f{observation_frame:04d}.png"
                            ),
                            "primary_R": str(
                                obs_dir / f"obs_R_f{observation_frame:04d}.png"
                            ),
                        },
                    },
                },
            }
        )
        schedule.append(saved[observation_frame])
    (logs / "sim_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    chunk_rows = [
        {
            "chunk_index": order,
            "observation_frame": observation_frame,
            "execution_start": 0,
            "execution_stop": exec_steps,
            "actions": [[0.0] * 32 for _ in range(2 * exec_steps)],
        }
        for order, observation_frame in enumerate(observation_frames)
    ]
    (logs / "policy_chunks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in chunk_rows), encoding="utf-8"
    )
    return schedule, chunk_rows


def _wrist_inset_bgr(tensor: np.ndarray, wrist_display: str) -> np.ndarray:
    """The 180x135 BGR inset a rollout video burns for one saved OBS tensor."""

    if wrist_display == "raw224":
        return cv2.cvtColor(
            cv2.resize(tensor, (180, 135), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR,
        )
    if wrist_display == "presentation_crop":
        display, _transform = prepare_policy_wrist_display_image(
            tensor, resize_mode="pad", source_hw=(539, 719), output_hw=(135, 180)
        )
        return cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unknown wrist_display mode: {wrist_display!r}")


def write_rollout(
    run_root: Path,
    index: int,
    *,
    frames: int = 2,
    color: tuple[int, int, int] | None = None,
    saved_policy_obs: bool = False,
    exec_steps: int = 1,
    wrist_display: str = "raw224",
    textured: bool = False,
) -> None:
    """Write the small synthetic rollout used by grid-rendering tests.

    ``saved_policy_obs`` burns a wrist inset derived from the saved 224px
    policy observation into the two canonical ROIs and writes the state log,
    policy OBS files and chunk log the grid renderer reads back. With
    ``wrist_display="raw224"`` the inset is the exact saved tensor scaled as a
    complete canvas, which the grid contract requires; with
    ``"presentation_crop"`` it is the forbidden letterbox-cropped variant an
    inline-rendered run produces.
    """
    run_dir = run_root / "test_scene" / f"run_{index:02d}"
    run_dir.mkdir(parents=True)
    video = run_dir / f"test_scene_run_{index:02d}.mp4"
    schedule: list[tuple[np.ndarray, np.ndarray]] = []
    chunk_rows: list[dict] = []
    if saved_policy_obs:
        schedule, chunk_rows = _write_saved_policy_obs(
            run_dir, frames, index, exec_steps
        )
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(frames):
            frame = np.full(
                (480, 640, 3),
                color or (70 + index, 100 + frame_index, 140),
                dtype=np.uint8,
            )
            if textured:
                yy, xx = np.indices((480, 640))
                frame = np.stack(
                    (
                        20 + xx // 8 + frame_index * 3,
                        30 + yy // 8 + index * 2,
                        40 + (xx + yy) // 16,
                    ),
                    axis=-1,
                ).astype(np.uint8)
            if saved_policy_obs:
                for xywh, tensor in zip(
                    (SAVED_WRIST_LEFT_XYWH, SAVED_WRIST_RIGHT_XYWH),
                    schedule[frame_index],
                ):
                    x, y, width, height = xywh
                    frame[y : y + height, x : x + width] = _wrist_inset_bgr(
                        tensor, wrist_display
                    )
            writer.write(frame)
    finally:
        writer.release()
    summary = {
        "new_final_success": index % 2 == 0,
        "score_best": index - 1,
        "video_wrist_insets_enabled": False,
        "video_wrist_inset_source": "off",
    }
    if saved_policy_obs:
        raw = wrist_display == "raw224"
        summary.update(
            {
                "frames": frames,
                "rendered_frames": frames,
                "output_fps": 10.0,
                "playback_action_dt_s": 0.1,
                "video_wrist_insets_enabled": True,
                "video_wrist_inset_source": "saved" if raw else "policy_observation",
                "video_sim_panel_x_offset": 0,
                "video_wrist_inset_layout": {
                    "version": 3 if raw else 2,
                    "left_xywh": list(SAVED_WRIST_LEFT_XYWH),
                    "right_xywh": list(SAVED_WRIST_RIGHT_XYWH),
                },
                "video_wrist_display_contract": (
                    RAW224_DISPLAY_CONTRACT if raw else CROPPED_DISPLAY_CONTRACT
                ),
                "policy_chunks_jsonl": str(
                    run_dir / "logs" / "policy_chunks.jsonl"
                ),
                "policy_chunks": len(chunk_rows),
                "action_horizon": 32,
                "adopted_action_chunks": exec_steps,
            }
        )
    (run_dir / "sim_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
RUN_COMMAND = """# Run Command

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 /venv/bin/python robosuite_env/sim.py \
--robot FR3 --policy-batch-server 127.0.0.1:8765 \
--policy-batch-client-id scene_run_01 --policy-batch-timeout-sec 7200 \
--no-render --frames 1 --gs-background /scenes/panorama \
--policy-obs-inset --wrist-gs --defer-overview-render \
--gs-port 8321 --gs-server-token deadbeef \
--render-lock /tmp/lock --out /runs/scene/run_01/scene_run_01.mp4 \
--logdir /runs/scene/run_01/logs \
--summary-path /runs/scene/run_01/sim_summary.json
```
"""

# RUN_COMMAND above has the file-path head that runs recorded before the batch
# runner switched to ``-m robosuite_env.sim``; both heads must parse and replay.
LEGACY_WORKER_HEAD = "/venv/bin/python robosuite_env/sim.py"
MODULE_WORKER_HEAD = "/venv/bin/python -m robosuite_env.sim"


def use_module_worker_head(run: Path) -> None:
    """Rewrite a fixture run's RUN_COMMAND.md to the current module head."""

    command_path = run / "RUN_COMMAND.md"
    command = command_path.read_text(encoding="utf-8")
    assert command.count(LEGACY_WORKER_HEAD) == 1
    command_path.write_text(
        command.replace(LEGACY_WORKER_HEAD, MODULE_WORKER_HEAD), encoding="utf-8"
    )


def replayed_worker_arguments(replay: list[str]) -> list[str]:
    """The worker arguments of a replay command, which always has the module head."""

    assert replay[1:3] == ["-m", "robosuite_env.sim"], replay[:3]
    return replay[3:]


# The run's wrist backend, as a policy observation records it: a run that records none
# names no supported backend, which a replay refuses.
_RECORDED_FIRST_FRAME = {"frame": 0, "policy": {"obs_log": {"policy_wrist_backend": "direct_kb4"}}}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "scene" / "run_01"
    (run / "logs").mkdir(parents=True)
    w2gs = tmp_path / "source_w2gs.npy"
    np.save(w2gs, np.eye(4, dtype=np.float64))
    (run / "logs" / "sim_log.jsonl").write_text(
        json.dumps(_RECORDED_FIRST_FRAME) + "\n", encoding="utf-8"
    )
    (run / "RUN_COMMAND.md").write_text(RUN_COMMAND, encoding="utf-8")
    (run / "sim_summary.json").write_text(
        json.dumps(
            {
                "execution_class": "diagnostic",
                "frames": 1,
                "wall_stopped_early": False,
                "rf_overlap": 0,
                "gs_w2gs": str(w2gs.resolve()),
                "gs_w2gs_raw": np.eye(4).tolist(),
                "gs_w2gs_effective": np.eye(4).tolist(),
                "gs_calibration_world_offset_m": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    return run

def _configure_complete_source(run: Path, frames: int) -> None:
    command_path = run / "RUN_COMMAND.md"
    command = command_path.read_text(encoding="utf-8")
    command = command.replace("--frames 1", f"--frames {frames}")
    command_path.write_text(command, encoding="utf-8")
    (run / "logs" / "sim_log.jsonl").write_text(
        "".join(
            json.dumps(_RECORDED_FIRST_FRAME if index == 0 else {"frame": index}) + "\n"
            for index in range(frames)
        ),
        encoding="utf-8",
    )
    summary_path = run / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "execution_class": "diagnostic",
            "frames": frames,
            "wall_stopped_early": False,
            "rf_overlap": 0,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

def _source_registration_summary(run: Path) -> dict:
    source = json.loads((run / "sim_summary.json").read_text(encoding="utf-8"))
    return {
        key: source[key]
        for key in (
            "gs_w2gs",
            "gs_w2gs_raw",
            "gs_w2gs_effective",
            "gs_calibration_world_offset_m",
        )
    }

def _configure_render_state_source(run: Path, frames: int = 1) -> dict:
    _configure_complete_source(run, frames)
    names = ["cup", "plate"]
    inventory = [
        {
            "name": name,
            "role": "target" if name == "cup" else "distractor",
            "joint_name": f"{name}_joint0",
            "qpos_address": 18 + 7 * index,
            "dof_address": 16 + 6 * index,
        }
        for index, name in enumerate(names)
    ]
    contract = build_render_state_contract(
        model_signature="render-state-test-model-signature",
        nq=123,
        target="cup",
        inventory=inventory,
    )
    rows = []
    for frame in range(frames):
        qpos = [0.0] * 123
        for index, entry in enumerate(inventory):
            address = entry["qpos_address"]
            qpos[address : address + 7] = [
                0.01 * index,
                -0.01 * index,
                0.2 + 0.001 * frame,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        rows.append({"frame": frame, **encode_render_state_row(qpos, contract)})
    (run / "logs" / "sim_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = run / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "objects": "cup_plate",
            "render_state_log": build_render_state_summary_contract(
                contract,
                rows_written=frames,
                complete=True,
            ),
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return contract

def _fr3_policy_action_summary(frames: int = 300) -> dict:
    action_hz = 30.0
    control_hz = 50.0
    ticks = round(frames * control_hz / action_hz)
    source_duration = frames / action_hz
    simulated_duration = ticks / control_hz
    return {
        "robot": "fr3",
        "physics": "dynamic",
        "frames": frames,
        "policy_action_hz_requested": action_hz,
        "policy_action_hz_effective": action_hz,
        "policy_action_dt_s": round(1.0 / action_hz, 9),
        "policy_action_dt_nominal_s": round(1.0 / action_hz, 9),
        "frame_hz_effective": action_hz,
        "frame_dt_s": round(1.0 / action_hz, 6),
        "output_fps": 30,
        "playback_action_dt_s": round(1.0 / action_hz, 6),
        "dynamic_subframes": 1,
        "dynamic_control_repeats_requested": 0,
        "dynamic_control_hz": 50,
        "fr3_dynamic_tick_schedule": (
            "source_to_controller_cumulative_nearest_half_up"
        ),
        "fr3_dynamic_control_tick_count_total": ticks,
        "fr3_dynamic_runtime_control_tick_count_total": ticks,
        "fr3_dynamic_runtime_tick_count_matches_schedule": True,
        "fr3_dynamic_source_duration_s": round(source_duration, 9),
        "fr3_dynamic_simulated_duration_s": round(simulated_duration, 9),
        "fr3_dynamic_timing_error_s": round(
            simulated_duration - source_duration, 9
        ),
        "dynamic_action_sim_dt_s": round(simulated_duration / frames, 9),
        "fr3_dynamic_boundary_phase_error_bound_s": 0.01,
        "fr3_dynamic_boundary_phase_error_s_max_abs": 0.006666667,
    }

def _configure_fr3_policy_action_source(run_dir: Path, frames: int = 300) -> None:
    _configure_complete_source(run_dir, frames)
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--robot FR3",
            "--robot FR3 --physics dynamic --policy-action-hz 30 "
            "--frame-hz 0 --dynamic-control-repeats 0 "
            "--dynamic-control-hz 50",
        ),
        encoding="utf-8",
    )
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(_fr3_policy_action_summary(frames))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _configure_frame_hz_source(
    run_dir: Path, frames: int = 300, frame_hz: int = 10
) -> None:
    """A source rollout recorded with ``--frame-hz``, as every retained profile.

    The summary values are what the worker writes for that clock with one
    dynamic subframe: ``output_fps`` is the rounded rate and
    ``playback_action_dt_s`` the period of one saved row, rounded to 1e-6 s.
    """

    _configure_complete_source(run_dir, frames)
    command_path = run_dir / "RUN_COMMAND.md"
    command_path.write_text(
        command_path.read_text(encoding="utf-8").replace(
            "--robot FR3",
            f"--robot FR3 --physics dynamic --frame-hz {float(frame_hz)} "
            "--dynamic-subframes 1",
        ),
        encoding="utf-8",
    )
    summary_path = run_dir / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "robot": "fr3",
            "physics": "dynamic",
            "frames": frames,
            "rendered_frames": 0,
            "output_fps": frame_hz,
            "playback_action_dt_s": round(1.0 / frame_hz, 6),
            "frame_hz_effective": float(frame_hz),
            "frame_dt_s": round(1.0 / frame_hz, 6),
            "dynamic_subframes": 1,
            "policy_action_hz_requested": 0.0,
            "policy_action_hz_effective": None,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
