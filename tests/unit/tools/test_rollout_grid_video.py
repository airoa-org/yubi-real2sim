# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import cv2
import pytest

from robosuite_env.rollout import outputs as outputs_module
from robosuite_env.tools import make_rollout_grid_video as grid
from tests._helpers.rollouts import write_rollout as _write_rollout


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_grid_is_rendered_only_from_clear_header_template(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index)
    output = run_root / "rollout_grid_10.mp4"

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=2,
        scene="test_scene",
    )

    assert manifest["scene"] == "test_scene"
    assert manifest["title"] == "test_scene"
    assert manifest["template"] == "clear-headers-4x3"
    assert manifest["template_path"] == (
        "config/render_templates/rollout_grid_clear_headers_4x3.json"
    )
    assert len(manifest["template_sha256"]) == 64
    assert manifest["render_contract"] == {
        "layout_source": "required_json_template",
        "template_only": True,
        "labels_obscure_rollout_pixels": False,
        "labels_obscure_rollout_pixels_check": (
            "per_frame_exact_cell_pixel_equality"
        ),
        "legacy_layout_cli_supported": False,
        "wrist_obs_source": "exact_saved_224px_policy_obs",
        "wrist_obs_rerendered": False,
    }
    # These rollouts declare no burned-in wrist OBS at all, so there is
    # nothing that could misrepresent the policy input.
    saved_obs = manifest["saved_policy_obs_validation"]
    assert saved_obs["contract"] == grid.GRID_SAVED_POLICY_OBS_CONTRACT
    assert saved_obs["rerendered_wrist_obs_allowed"] is False
    assert saved_obs["checked_rollouts"] == 10
    assert saved_obs["exact_saved_obs_rollouts"] == 0
    assert all(
        item["wrist_inset_source"] == "off" for item in saved_obs["inputs"]
    )
    assert manifest["geometry"]["canvas_wh"] == [1920, 1080]
    assert manifest["geometry"]["columns"] == 4
    assert manifest["geometry"]["rows"] == 3
    assert manifest["geometry"]["header_height"] == 36
    assert manifest["geometry"]["source_xywh_within_cell"] == [24, 36, 432, 324]
    assert manifest["geometry"]["summary_xywh"] == [960, 720, 960, 360]
    assert manifest["metrics"]["new_final_success_count"] == 5
    assert manifest["metrics"]["new_final_success_rate"] == 0.5
    assert manifest["metrics"]["score_best_values"] == list(range(10))
    assert manifest["content_validation"]["checked_frames"] == 2
    assert manifest["content_validation"]["checked_rollouts"] == 10
    assert manifest["content_validation"]["metrics"]["mean_abs_error"] < 8.0
    provenance = manifest["provenance"]
    assert provenance["schema_version"] == 1
    assert len(provenance["inputs"]) == 10
    assert set(provenance["inputs"][0]) == {
        "scene",
        "run",
        "video",
        "sim_summary",
    }
    assert provenance["renderer"]["path"] == str(Path(grid.__file__).resolve())
    assert provenance["template"]["sha256"] == manifest["template_sha256"]
    assert provenance["outputs"]["video"]["sha256"] == grid._sha256_file(output)

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) == 1920
        assert int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == 1080
        assert int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) == 2
        ok, frame = capture.read()
        assert ok
    finally:
        capture.release()
    assert float(frame[0:36, 0:480].mean()) > 180.0
    # The source begins below the header. Its pixels remain visible instead of
    # being covered by the legacy in-frame black label rectangle.
    assert float(frame[45:80, 30:180].mean()) > 70.0
    assert output.with_name("rollout_grid_10_poster.jpg").is_file()
    saved_manifest = json.loads(output.with_suffix(".json").read_text())
    assert saved_manifest == manifest


def test_provenance_binds_optional_rerender_artifacts_and_poster(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index, frames=1)
    rerender_dir = run_root / "test_scene" / "run_01" / "rerender"
    rerender_dir.mkdir()
    validation = rerender_dir / "video_validation.json"
    provenance = rerender_dir / "provenance.json"
    validation.write_text('{"ok": true}\n', encoding="utf-8")
    provenance.write_text('{"schema_version": 2}\n', encoding="utf-8")
    output = run_root / "rollout_grid_10.mp4"

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=1,
        scene="test_scene",
    )

    first = manifest["provenance"]["inputs"][0]
    assert first["rerender_validation"]["sha256"] == grid._sha256_file(validation)
    assert first["rerender_provenance"]["sha256"] == grid._sha256_file(provenance)
    poster = Path(manifest["poster"])
    assert manifest["provenance"]["outputs"]["poster"]["sha256"] == (
        grid._sha256_file(poster)
    )
    for record in (
        first["video"],
        first["sim_summary"],
        first["rerender_validation"],
        first["rerender_provenance"],
        manifest["provenance"]["renderer"],
        manifest["provenance"]["template"],
        manifest["provenance"]["outputs"]["video"],
        manifest["provenance"]["outputs"]["poster"],
    ):
        assert len(record["sha256"]) == 64
        assert set(record["state"]) == {
            "device",
            "inode",
            "mode",
            "size",
            "mtime_ns",
            "ctime_ns",
        }


def test_content_validation_rejects_run_swap(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        value = index * 22
        _write_rollout(
            run_root,
            index,
            frames=1,
            color=(value, 255 - value, (value * 3) % 256),
        )
    output = run_root / "rollout_grid_10.mp4"
    grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=1,
        scene="test_scene",
    )
    template = grid.load_template(grid.DEFAULT_TEMPLATE_PATH)
    videos = grid._discover_videos(run_root, template["video_glob"], scene="test_scene")

    with pytest.raises(RuntimeError, match="does not match its bound inputs"):
        grid._validate_grid_content(
            output,
            list(reversed(videos)),
            expected_frames=1,
            expected_size=tuple(template["geometry"]["canvas_wh"]),
            columns=template["geometry"]["columns"],
            cell_wh=tuple(template["geometry"]["cell_wh"]),
            source_xywh=tuple(template["geometry"]["source_xywh_within_cell"]),
        )


def test_changed_input_fails_before_replacing_known_good_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index, frames=1)
    output = run_root / "rollout_grid_10.mp4"
    known_good = b"pre-existing known-good grid"
    output.write_bytes(known_good)
    original_validate = grid._validate_grid_content

    def mutate_after_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        summary = run_root / "test_scene" / "run_01" / "sim_summary.json"
        summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(grid, "_validate_grid_content", mutate_after_validation)
    with pytest.raises(RuntimeError, match="changed during grid rendering"):
        grid.render_from_template(
            run_root=run_root,
            output=output,
            template_path=grid.DEFAULT_TEMPLATE_PATH,
            expected_frames=1,
            scene="test_scene",
        )
    assert output.read_bytes() == known_good
    assert not list(run_root.glob(".rollout_grid_10.tmp-*.mp4"))


def test_legacy_freeform_layout_flags_are_rejected(tmp_path: Path):
    base = [str(tmp_path), "--output", str(tmp_path / "grid.mp4")]
    legacy_flags = {
        "--columns": "5",
        "--cell-width": "280",
        "--source-crop": "0",
        "--fps": "10",
        "--expected-videos": "10",
        "--poster-frame": "149",
        "--video-glob": "*/run_*/*.mp4",
    }
    for flag, value in legacy_flags.items():
        with pytest.raises(SystemExit):
            grid.parse_args([*base, flag, value])

    parameters = set(inspect.signature(grid.render_from_template).parameters)
    assert parameters == {
        "run_root",
        "output",
        "template_path",
        "title",
        "expected_frames",
        "scene",
        "runs",
    }
    source = Path(grid.__file__).read_text(encoding="utf-8")
    assert "def make_grid(" not in source
    assert "_draw_label" not in source


def test_batch_runner_requires_template_grid_for_supported_rollouts():
    source = (REPO_ROOT / "scripts" / "run_policy_batch.sh").read_text(
        encoding="utf-8"
    )
    assert "render_rollout_grid_template" in source
    assert '--template "${GRID_TEMPLATE}"' in source
    assert '--output "${grid_root}/rollout_grid_${REPEATS}.mp4"' in source
    assert '--scene "${tag}"' in source
    assert '--title "${tag}"' in source
    assert "required template grid render failed" in source


def test_run_filter_selects_exact_videos_in_numeric_order(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 101):
        video = run_root / "test_scene" / f"run_{index:02d}" / "rollout.mp4"
        video.parent.mkdir(parents=True)
        video.touch()

    selected = tuple(f"run_{index:02d}" for index in range(20, 10, -1))
    videos = grid._discover_videos(
        run_root,
        "*/run_*/*.mp4",
        scene="test_scene",
        runs=selected,
    )

    assert [path.parent.name for path in videos] == [
        f"run_{index:02d}" for index in range(11, 21)
    ]
    all_videos = grid._discover_videos(run_root, "*/run_*/*.mp4")
    assert [path.parent.name for path in all_videos] == [
        f"run_{index:02d}" for index in range(1, 101)
    ]


def test_run_filter_is_fail_closed_for_missing_duplicate_and_ambiguous(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    video = run_root / "test_scene" / "run_01" / "rollout.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    with pytest.raises(grid.TemplateError, match="missing=.*run_02"):
        grid._discover_videos(
            run_root,
            "*/run_*/*.mp4",
            scene="test_scene",
            runs=("run_01", "run_02"),
        )
    with pytest.raises(grid.TemplateError, match="duplicate entries"):
        grid._discover_videos(
            run_root,
            "*/run_*/*.mp4",
            scene="test_scene",
            runs=("run_01", "run_01"),
        )

    second = video.parent / "second.mp4"
    second.touch()
    with pytest.raises(grid.TemplateError, match="ambiguous=.*run_01"):
        grid._discover_videos(
            run_root,
            "*/run_*/*.mp4",
            scene="test_scene",
            runs=("run_01",),
        )


def test_partial_exact_ten_filter_renders_only_selected_runs(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 13):
        _write_rollout(run_root, index, frames=1)
    selected = tuple(f"run_{index:02d}" for index in range(3, 13))

    manifest = grid.render_from_template(
        run_root=run_root,
        output=run_root / "rollout_grid_selected_10.mp4",
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=1,
        scene="test_scene",
        runs=selected,
    )

    assert manifest["runs"] == list(selected)
    assert [Path(path).parent.name for path in manifest["inputs"]] == list(selected)


def test_runs_cli_normalizes_exact_names_and_rejects_invalid_filters(
    tmp_path: Path,
):
    base = [str(tmp_path), "--output", str(tmp_path / "grid.mp4")]
    parsed = grid.parse_args([*base, "--runs", "run_03,run_01"])
    assert parsed.runs == ("run_03", "run_01")

    for value in ("run_01,run_01", "run_01,", "one"):
        with pytest.raises(SystemExit):
            grid.parse_args([*base, "--runs", value])


def test_hundred_rollout_template_reserves_two_summary_rows():
    template = grid.load_template(
        REPO_ROOT
        / "config"
        / "render_templates"
        / "rollout_grid_clear_headers_10x12.json"
    )

    assert template["expected_videos"] == 100
    assert template["geometry"]["rollout_cells"] == 100
    assert template["geometry"]["columns"] == 10
    assert template["geometry"]["rows"] == 12
    assert template["geometry"]["summary_xywh"] == [0, 1080, 1920, 216]


def test_thirty_rollout_grid_renders_and_validates_all_cells(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 31):
        _write_rollout(run_root, index, frames=1)
    output = run_root / "rollout_grid_30.mp4"
    runs = tuple(f"run_{index:02d}" for index in range(1, 31))

    template_path = (
        REPO_ROOT
        / "config"
        / "render_templates"
        / "rollout_grid_clear_headers_6x6.json"
    )
    template = grid.load_template(template_path)
    assert template["expected_videos"] == 30
    assert template["geometry"]["rollout_cells"] == 30
    assert template["geometry"]["columns"] == 6
    assert template["geometry"]["rows"] == 6
    assert template["geometry"]["source_xywh_within_cell"] == [56, 24, 208, 156]
    assert template["geometry"]["summary_xywh"] == [0, 900, 1920, 180]

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=template_path,
        title="Thirty rollout grid",
        expected_frames=1,
        scene="test_scene",
        runs=runs,
    )

    assert manifest["template"] == "clear-headers-6x6"
    assert manifest["runs"] == list(runs)
    assert manifest["content_validation"]["checked_rollouts"] == 30
    assert manifest["content_validation"]["checked_frames"] == 1
    assert output.is_file()


@pytest.mark.parametrize("fail_during_render", [False, True])
def test_grid_releases_render_decoders_before_postprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_during_render: bool
):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index)
    output = run_root / "rollout_grid_10.mp4"
    real_capture = cv2.VideoCapture
    readers = []
    active = set()
    postprocessing = []

    class TrackedCapture:
        def __init__(self, *args, **kwargs):
            # One source set plus the encoded grid, independent of how many
            # decoder threads OpenCV chooses on the host running this test.
            assert len(active) < 11, "render and validation decoders overlap"
            self.capture = real_capture(*args, **kwargs)
            assert self.capture.isOpened()
            self.decoded_frames = 0
            readers.append(self)
            active.add(self)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def read(self):
            ok, frame = self.capture.read()
            self.decoded_frames += int(ok)
            return ok, frame

        def release(self):
            self.capture.release()
            active.discard(self)

    def after_render_readers_close(function):
        def checked(*args, **kwargs):
            assert not active, "render decoders must close before postprocessing"
            postprocessing.append(function.__name__)
            return function(*args, **kwargs)

        return checked

    monkeypatch.setattr(cv2, "VideoCapture", TrackedCapture)
    for name in ("_transcode_to_h264", "_validate_grid_content"):
        monkeypatch.setattr(grid, name, after_render_readers_close(getattr(grid, name)))
    if fail_during_render:
        real_put_text = grid._put_text_fit

        def overdraw(image, text, **kwargs):
            real_put_text(image, text, **kwargs)
            image[40:120, 30:200] = 0

        monkeypatch.setattr(grid, "_put_text_fit", overdraw)

    arguments = dict(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=2,
        scene="test_scene",
    )
    if fail_during_render:
        with pytest.raises(RuntimeError, match="covered rollout pixels"):
            grid.render_from_template(**arguments)
        assert not postprocessing
        assert not output.exists()
        assert len(readers) == 10
    else:
        manifest = grid.render_from_template(**arguments)
        assert "_validate_grid_content" in postprocessing
        assert manifest["content_validation"]["checked_rollouts"] == 10
        assert manifest["content_validation"]["checked_frames"] == 2
        assert len(readers) == 21
        assert all(reader.decoded_frames == 2 for reader in readers)
    assert not active
    assert all(not reader.isOpened() for reader in readers)


def test_grid_readers_start_only_the_helper_decoder_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Render and validation readers must start only the helper's threads.

    A default OpenCV reader starts cv2.getNumberOfCPUs() decoder threads when it
    opens an mp4v or H.264 video. Validation holds every source reader and the
    encoded grid at once, so a 30-rollout grid then needs about 31 x CPUs tasks:
    496 at 16 CPUs, which a 512-task container limit cannot hold. Each reader
    opened on the production path must decode with one thread and add no thread
    at open.

    On a 1-CPU host a default reader starts no thread either. So the grid is
    first rendered with the helper's setting raised to 2, a positive control
    that does not depend on the CPU count: every reader must then report 2 and
    start threads, which shows that each one opens through the helper and that
    the thread count is measured live.
    """

    real_capture = cv2.VideoCapture

    def render(name: str, threads: int) -> list[tuple[float, int]]:
        run_root = tmp_path / name
        for index in range(1, 11):
            _write_rollout(run_root, index, textured=True)
        opened = []

        def counted_capture(*args, **kwargs):
            before = len(os.listdir("/proc/self/task"))
            capture = real_capture(*args, **kwargs)
            added = len(os.listdir("/proc/self/task")) - before
            opened.append((capture.get(cv2.CAP_PROP_N_THREADS), max(added, 0)))
            return capture

        with monkeypatch.context() as patch:
            patch.setattr(cv2, "VideoCapture", counted_capture)
            patch.setattr(outputs_module, "VIDEO_DECODER_THREADS", threads)
            manifest = grid.render_from_template(
                run_root=run_root,
                output=run_root / "rollout_grid_10.mp4",
                template_path=grid.DEFAULT_TEMPLATE_PATH,
                expected_frames=2,
                scene="test_scene",
            )
        assert manifest["content_validation"]["checked_rollouts"] == 10
        assert manifest["content_validation"]["checked_frames"] == 2
        return opened

    # Ten render readers, then the encoded grid and ten validation readers.
    control = render("control", 2)
    assert len(control) == 21
    assert all(threads == 2.0 and added > 0 for threads, added in control), control
    assert render("run", 1) == [(1.0, 0)] * 21


"""Contract tests for the policy-eval 10-rollout grid template."""

from pathlib import Path


from robosuite_env.rollout.outputs import _resolve_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_EVAL_TEMPLATE_PATH = (
    REPO_ROOT / "config" / "render_templates" / "rollout_grid_policy_eval_4x3.json"
)


def _add_policy_identity(
    run_root: Path,
    *,
    checkpoint: str = "/ckpt/pi05_cup",
    instruction: str = "Pick up the cup with your right hand and set it on the plate",
    task: str = "cup_on_plate",
) -> None:
    for summary_path in sorted(run_root.glob("*/run_*/sim_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["policy"] = checkpoint
        summary["instruction"] = instruction
        summary["success_criterion"] = {"name": task}
        summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_policy_eval_template_reserves_bottom_right_panel():
    template = grid.load_template(POLICY_EVAL_TEMPLATE_PATH)
    assert template["id"] == "policy-eval-4x3"
    assert template["summary_content"] == "policy_eval"
    assert template["expected_videos"] == 10
    geometry = template["geometry"]
    # The info panel occupies exactly the two reserved bottom-right cells.
    assert geometry["summary_xywh"] == [960, 720, 960, 360]
    assert geometry["columns"] == 4 and geometry["rows"] == 3
    legacy_template = grid.load_template(grid.DEFAULT_TEMPLATE_PATH)
    assert legacy_template["summary_content"] == "legacy"


@pytest.mark.skipif(
    _resolve_ffmpeg() is None,
    reason="ffmpeg is not installed (scripts/bootstrap.sh provides it)",
)
def test_policy_eval_grid_publishes_h264_with_shared_identity(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index)
    _add_policy_identity(run_root)
    output = run_root / "rollout_grid_policy_eval.mp4"

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=POLICY_EVAL_TEMPLATE_PATH,
        title="policy eval grid",
        expected_frames=2,
    )

    assert manifest["template"] == "policy-eval-4x3"
    assert manifest["codec"] == "h264"
    assert manifest["intermediate_codec"] in {"mp4v", "avc1"}
    # The panel title is always the experiment directory, not the CLI title.
    assert manifest["title"] == run_root.name
    assert output.is_file()
    capture = cv2.VideoCapture(str(output))
    try:
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        published = "".join(
            chr((fourcc >> (8 * shift)) & 0xFF) for shift in range(4)
        )
    finally:
        capture.release()
    assert published in {"avc1", "h264", "H264"}


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("checkpoint", "one shared checkpoint"),
        ("instruction", "non-empty instruction"),
    ),
)
def test_policy_eval_grid_requires_shared_policy_identity(
    tmp_path: Path,
    failure: str,
    message: str,
):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index)
    _add_policy_identity(run_root)
    run_index = 5 if failure == "checkpoint" else 3
    broken = run_root / "test_scene" / f"run_{run_index:02d}" / "sim_summary.json"
    summary = json.loads(broken.read_text(encoding="utf-8"))
    if failure == "checkpoint":
        summary["policy"] = "/ckpt/other/1"
    else:
        del summary["instruction"]
    broken.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(grid.TemplateError, match=message):
        grid.render_from_template(
            run_root=run_root,
            output=run_root / "grid.mp4",
            template_path=POLICY_EVAL_TEMPLATE_PATH,
            expected_frames=2,
        )


"""Contract tests for the exact-saved-policy-OBS grid input rule."""

import numpy as np

from tests._helpers.rollouts import saved_policy_obs_tensor


def _saved_obs_run_root(
    tmp_path: Path,
    *,
    frames: int = 4,
    exec_steps: int = 2,
    wrist_display: str = "raw224",
    count: int = 10,
) -> Path:
    run_root = tmp_path / "run"
    for index in range(1, count + 1):
        _write_rollout(
            run_root,
            index,
            frames=frames,
            saved_policy_obs=True,
            exec_steps=exec_steps,
            wrist_display=wrist_display,
        )
    return run_root


def test_grid_binds_every_input_to_its_exact_saved_policy_obs(tmp_path: Path):
    run_root = _saved_obs_run_root(tmp_path)
    output = run_root / "rollout_grid_10.mp4"

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=4,
        scene="test_scene",
    )

    report = manifest["saved_policy_obs_validation"]
    assert report["saved_wrist_exact_match"] is True
    assert report["checked_rollouts"] == 10
    assert report["exact_saved_obs_rollouts"] == 10
    for item in report["inputs"]:
        exact = item["saved_wrist_exact_match"]
        assert exact["ok"] is True
        assert exact["layout_version"] == 3
        assert exact["transition_frames"] == [2]
        assert item["display_contract"] == grid.SAVED_WRIST_DISPLAY_CONTRACT
        assert item["unique_saved_obs_pairs"] == 2
    assert manifest["render_contract"]["wrist_obs_rerendered"] is False


def test_grid_rejects_presentation_cropped_wrist_obs(tmp_path: Path):
    """A burned-in inset must equal the saved policy observation, not a crop."""

    run_root = _saved_obs_run_root(tmp_path, wrist_display="presentation_crop")
    output = run_root / "rollout_grid_10.mp4"

    with pytest.raises(grid.TemplateError, match="exact saved 224px"):
        grid.render_from_template(
            run_root=run_root,
            output=output,
            template_path=grid.DEFAULT_TEMPLATE_PATH,
            expected_frames=4,
            scene="test_scene",
        )
    assert not output.exists()


def test_grid_rejects_wrist_pixels_that_are_not_the_saved_observation(
    tmp_path: Path,
):
    run_root = _saved_obs_run_root(tmp_path)
    # Keep the declared contract but replace one saved observation on disk, so
    # the burned-in inset is no longer the tensor the policy consumed.
    replaced = (
        run_root / "test_scene" / "run_04" / "logs" / "policy_obs" / "obs_L_f0000.png"
    )
    assert cv2.imwrite(
        str(replaced),
        cv2.cvtColor(saved_policy_obs_tensor(5, 1), cv2.COLOR_RGB2BGR),
    )
    output = run_root / "rollout_grid_10.mp4"

    with pytest.raises(grid.TemplateError, match="differs from the saved 224px"):
        grid.render_from_template(
            run_root=run_root,
            output=output,
            template_path=grid.DEFAULT_TEMPLATE_PATH,
            expected_frames=4,
            scene="test_scene",
        )
    assert not output.exists()


def test_grid_requires_declared_wrist_inset_provenance(tmp_path: Path):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index, frames=1)
    summary_path = run_root / "test_scene" / "run_02" / "sim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["video_wrist_insets_enabled"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = run_root / "rollout_grid_10.mp4"

    with pytest.raises(
        grid.TemplateError, match="does not declare video_wrist_insets_enabled"
    ):
        grid.render_from_template(
            run_root=run_root,
            output=output,
            template_path=grid.DEFAULT_TEMPLATE_PATH,
            expected_frames=1,
            scene="test_scene",
        )
    assert not output.exists()


def test_published_video_summary_prefers_the_republished_descriptor(
    tmp_path: Path,
):
    run_dir = tmp_path / "test_scene" / "run_01"
    (run_dir).mkdir(parents=True)
    (run_dir / "sim_summary.json").write_text("{}", encoding="utf-8")
    assert grid.published_video_summary_path(run_dir) == (
        run_dir / "sim_summary.json"
    )

    rerender = run_dir / "rerender"
    rerender.mkdir()
    (rerender / "sim_summary.json").write_text("{}", encoding="utf-8")
    assert grid.published_video_summary_path(run_dir) == (
        rerender / "sim_summary.json"
    )

    overlay = run_dir / grid.SAVED_POLICY_OBS_DIRNAME
    overlay.mkdir()
    (overlay / "sim_summary.json").write_text("{}", encoding="utf-8")
    assert grid.published_video_summary_path(run_dir) == (
        overlay / "sim_summary.json"
    )


def test_adopted_chunk_timeline_tracks_the_executing_chunk():
    rows = [
        {
            "chunk_index": 0,
            "observation_frame": 0,
            "execution_start": 0,
            "execution_stop": 16,
        },
        {
            "chunk_index": 1,
            "observation_frame": 16,
            "execution_start": 0,
            "execution_stop": 16,
        },
    ]

    timeline = grid._adopted_chunk_timeline(rows, 30, "test_scene/run_01")

    assert timeline[0] == {
        "chunk_index": 0,
        "adopted": 1,
        "step": 1,
        "adopted_steps": 16,
        "start": 0,
        "stop": 16,
    }
    assert timeline[15]["step"] == 16
    assert timeline[16]["chunk_index"] == 1
    assert timeline[16]["step"] == 1
    assert timeline[29]["step"] == 14
    line = grid._adopted_chunk_panel_line(
        [timeline], [2], [32], [16], 17
    )
    assert line == (
        "adopted action chunks 2/2 | chunk 1 step 2/16 "
        "| frames 16-31 | ah=32 exec=16"
    )


def test_adopted_chunk_panel_line_reports_disagreeing_rollouts():
    fast = grid._adopted_chunk_timeline(
        [{"chunk_index": 0, "observation_frame": 0,
          "execution_start": 0, "execution_stop": 2},
         {"chunk_index": 1, "observation_frame": 2,
          "execution_start": 0, "execution_stop": 2}],
        4,
        "test_scene/run_01",
    )
    slow = grid._adopted_chunk_timeline(
        [{"chunk_index": 0, "observation_frame": 0,
          "execution_start": 0, "execution_stop": 4}],
        4,
        "test_scene/run_02",
    )

    line = grid._adopted_chunk_panel_line([fast, slow], [2, 1], [32, 32], [2, 4], 2)

    assert "adopted action chunks 1~2/1~2" in line
    assert "chunk 0~1" in line
    assert grid._adopted_chunk_panel_line([None], [None], [None], [None], 0) == (
        "adopted action chunks: not recorded"
    )


def test_summary_panel_prints_adopted_action_chunks_per_frame(tmp_path: Path):
    run_root = _saved_obs_run_root(tmp_path, frames=4, exec_steps=2)
    output = run_root / "rollout_grid_10.mp4"

    manifest = grid.render_from_template(
        run_root=run_root,
        output=output,
        template_path=grid.DEFAULT_TEMPLATE_PATH,
        expected_frames=4,
        scene="test_scene",
    )

    qx, qy, qw, qh = manifest["geometry"]["summary_xywh"]
    capture = cv2.VideoCapture(str(output))
    try:
        panels = []
        for _ in range(4):
            ok, frame = capture.read()
            assert ok
            panels.append(frame[qy : qy + qh, qx : qx + qw].copy())
    finally:
        capture.release()
    # The adopted-chunk line advances with the chunk, so the frames that cross
    # a replan boundary must not paint an identical panel.
    assert not np.array_equal(panels[1], panels[2])
    # The bottom band of the panel holds the adopted-chunk line; it must
    # carry ink rather than being an untouched background rectangle.
    band = panels[0][int(qh * 5 / 7) : int(qh * 6.6 / 7)]
    assert float(band.std()) > 5.0


def test_a_label_that_covers_rollout_pixels_fails_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_root = tmp_path / "run"
    for index in range(1, 11):
        _write_rollout(run_root, index, frames=1)
    output = run_root / "rollout_grid_10.mp4"
    original = grid._put_text_fit

    def overdraw(image, text, **kwargs):
        original(image, text, **kwargs)
        # A legacy in-frame label: paint over the first cell's rollout pixels.
        image[40:120, 30:200] = 0

    monkeypatch.setattr(grid, "_put_text_fit", overdraw)
    with pytest.raises(RuntimeError, match="covered rollout pixels"):
        grid.render_from_template(
            run_root=run_root,
            output=output,
            template_path=grid.DEFAULT_TEMPLATE_PATH,
            expected_frames=1,
            scene="test_scene",
        )
    assert not output.exists()


def test_batch_runner_republishes_saved_policy_obs_before_the_grid():
    source = (REPO_ROOT / "scripts" / "run_policy_batch.sh").read_text(
        encoding="utf-8"
    )
    assert "publish_saved_policy_obs_videos()" in source
    assert "-m robosuite_env.tools.rewrite_saved_policy_obs_video" in source
    assert "required saved policy OBS republish failed" in source
    assert source.index("! publish_saved_policy_obs_videos") < source.index(
        "! render_rollout_grid_template"
    )
