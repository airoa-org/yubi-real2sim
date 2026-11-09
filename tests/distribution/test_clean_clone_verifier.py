# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "verify_clean_clone.sh"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_clean_clone_acceptance.py"
VALID_COMMIT = "a" * 40
# The artifacts config/release/fr3/flat/a_room_cup_cup1trim.yaml binds: the
# production environment-check profile that full mode installs and runs.
RELEASE_ARTIFACT_IDS = {
    "pi05-cup",
    "public-scene-flat-a-room-v2",
    "public-cup-plate-visual-v1",
    "fr3-collision-franka-description-2fd6aaa8",
}


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "clean_clone_acceptance_validator", VALIDATOR_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _run_wrapper(*args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _base_args(work_root: Path, *, mode: str = "fixture") -> list[str]:
    return [
        "--source",
        str(REPO_ROOT),
        "--commit",
        VALID_COMMIT,
        "--work-root",
        str(work_root),
        "--mode",
        mode,
        "--timestamp",
        "20260824_120000",
    ]


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_wrapper_has_no_scheduler_or_gpu_override_interface():
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'unset CUDA_VISIBLE_DEVICES' not in source
    assert 'export CUDA_VISIBLE_DEVICES' not in source
    assert "--gpu-ids" not in source

    retired_gpu_variable = "".join(("REAL2SIM_", "GPU_IDS"))
    scheduler_tokens = (
        "".join(("s", "batch")),
        "".join(("s", "run")),
        "".join(("s", "queue")),
        "".join(("SLURM", "_")),
        "".join(("PBS", "_")),
    )
    assert retired_gpu_variable not in source
    assert not any(token in source for token in scheduler_tokens)
    assert "".join(("rm ", "-rf")) not in source
    assert RELEASE_ARTIFACT_IDS <= set(source.split())


def test_full_mode_runs_the_production_environment_check_profile():
    import yaml

    source = WRAPPER.read_text(encoding="utf-8")
    profile_path = "config/release/fr3/flat/a_room_cup_cup1trim.yaml"
    profile = yaml.safe_load((REPO_ROOT / profile_path).read_text(encoding="utf-8"))

    assert f'RELEASE_CONFIG="{profile_path}"' in source
    assert 'DOWNLOAD="${REPO}/.venv/bin/real2sim-download-release"' in source
    assert source.count('"${DOWNLOAD}" --profile "${RELEASE_CONFIG}"') == 2
    assert profile["reproducibility"]["mode"] == "production"
    assert set(profile["reproducibility"]["artifacts"].values()) == RELEASE_ARTIFACT_IDS
    assert "--mode production --expected-rollouts 10" in source
    assert profile["runner"]["expected_rollouts"] == 10


def test_wrapper_help_is_side_effect_free(tmp_path: Path):
    result = _run_wrapper("--help")

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--min-successes N" in result.stdout
    assert "GPU-free fixture and common CPU checks" in result.stdout
    assert not list(tmp_path.iterdir())


def test_invalid_commit_fails_before_creating_a_run_root(tmp_path: Path):
    args = _base_args(tmp_path)
    args[args.index(VALID_COMMIT)] = "ABC123"

    result = _run_wrapper(*args)

    assert result.returncode == 2
    assert "40 lowercase hexadecimal" in result.stderr
    assert not list(tmp_path.glob("real2sim-acceptance.*"))


def test_full_mode_requires_explicit_confirmation_before_clone(tmp_path: Path):
    result = _run_wrapper(*_base_args(tmp_path, mode="full"))

    assert result.returncode == 2
    assert "requires --confirm-full" in result.stderr
    assert not list(tmp_path.glob("real2sim-acceptance.*"))


def test_full_mode_requires_a_success_floor_before_clone(tmp_path: Path):
    result = _run_wrapper(
        *_base_args(tmp_path, mode="full"),
        "--confirm-full",
    )

    assert result.returncode == 2
    assert "requires --min-successes N" in result.stderr
    assert not list(tmp_path.glob("real2sim-acceptance.*"))


@pytest.mark.parametrize("value", ["0", "01", "11", "1.5", "many"])
def test_full_mode_rejects_an_invalid_success_floor_before_clone(
    tmp_path: Path, value: str
):
    result = _run_wrapper(
        *_base_args(tmp_path, mode="full"),
        "--confirm-full",
        "--min-successes",
        value,
    )

    assert result.returncode == 2
    assert "integer from 1 through 10" in result.stderr
    assert not list(tmp_path.glob("real2sim-acceptance.*"))


def test_fixture_mode_rejects_a_success_floor(tmp_path: Path):
    result = _run_wrapper(
        *_base_args(tmp_path),
        "--min-successes",
        "1",
    )

    assert result.returncode == 2
    assert "valid only in full mode" in result.stderr
    assert not list(tmp_path.glob("real2sim-acceptance.*"))


def test_valid_full_success_floor_reaches_clone(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    _write_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\nexit 39\n",
    )
    work_root = tmp_path / "runs"
    work_root.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = _run_wrapper(
        *_base_args(work_root, mode="full"),
        "--confirm-full",
        "--min-successes",
        "4",
        env=env,
    )

    assert result.returncode == 39
    assert "clone source repository" in result.stderr
    assert len(list(work_root.glob("real2sim-acceptance.*"))) == 1


@pytest.mark.parametrize("cuda_value", [None, "2,5"])
def test_clone_failure_preserves_rc_and_accelerator_visibility(
    tmp_path: Path, cuda_value: str | None
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    observation = tmp_path / "git-environment.txt"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/usr/bin/env bash
set -eu
{
  if [[ -v PYTHONPATH ]]; then echo PYTHONPATH=set; else echo PYTHONPATH=unset; fi
  if [[ -v PYTHONHOME ]]; then echo PYTHONHOME=set; else echo PYTHONHOME=unset; fi
  if [[ -v VIRTUAL_ENV ]]; then echo VIRTUAL_ENV=set; else echo VIRTUAL_ENV=unset; fi
  if [[ -v UV_PROJECT_ENVIRONMENT ]]; then echo UV_PROJECT_ENVIRONMENT=set; else echo UV_PROJECT_ENVIRONMENT=unset; fi
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
  printf 'argv=%s\n' "$*"
} > "${FAKE_GIT_OBSERVATION}"
exit 37
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    work_root = tmp_path / "runs"
    work_root.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_GIT_OBSERVATION": str(observation),
            "PYTHONPATH": "/contaminated/source",
            "PYTHONHOME": "/contaminated/python",
            "VIRTUAL_ENV": "/contaminated/venv",
            "UV_PROJECT_ENVIRONMENT": "/contaminated/uv-project",
        }
    )
    if cuda_value is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = cuda_value

    result = _run_wrapper(*_base_args(work_root), env=env)

    assert result.returncode == 37
    assert "PASS" not in result.stdout
    run_roots = list(work_root.glob("real2sim-acceptance.*"))
    assert len(run_roots) == 1
    evidence = run_roots[0] / "evidence"
    assert (evidence / "failure.txt").read_text(encoding="utf-8").splitlines() == [
        "status=error",
        "step=clone source repository",
        "exit_code=37",
        f"log={evidence / 'clone.log'}",
    ]
    expected_cuda = cuda_value if cuda_value is not None else "<unset>"
    assert (evidence / "accelerator-environment.txt").read_text(
        encoding="utf-8"
    ) == f"CUDA_VISIBLE_DEVICES={expected_cuda}\n"
    observed = observation.read_text(encoding="utf-8")
    assert "PYTHONPATH=unset\n" in observed
    assert "PYTHONHOME=unset\n" in observed
    assert "VIRTUAL_ENV=unset\n" in observed
    assert "UV_PROJECT_ENVIRONMENT=unset\n" in observed
    assert f"CUDA_VISIBLE_DEVICES={expected_cuda}\n" in observed
    assert "argv=clone --no-local --no-checkout --" in observed


def test_fixture_major_flow_unsets_child_cuda_but_preserves_result_evidence(
    tmp_path: Path,
):
    template = tmp_path / "clone-template"
    fake_bin = tmp_path / "fake-bin"
    work_root = tmp_path / "runs"
    flow_log = tmp_path / "flow.log"
    work_root.mkdir()
    (template / "scripts").mkdir(parents=True)
    (template / "config" / "release").mkdir(parents=True)
    (template / "artifacts").mkdir()
    (template / ".venv" / "bin").mkdir(parents=True)
    (template / "scripts" / "validate_clean_clone_acceptance.py").write_text(
        "# Dispatched by the fake clone-local Python executable.\n",
        encoding="utf-8",
    )
    for path in (
        template / "config" / "release" / "fixture_v2.yaml",
        template / "config" / "support-matrix.yaml",
        template / "artifacts" / "manifest.yaml",
        template / "requirements-lock.ci.txt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    _write_executable(
        template / "scripts" / "bootstrap.sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'bootstrap cvd=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
[[ ! -v CUDA_VISIBLE_DEVICES ]]
""",
    )
    _write_executable(
        template / ".venv" / "bin" / "python",
        """#!/usr/bin/env bash
set -euo pipefail
cvd="${CUDA_VISIBLE_DEVICES-<unset>}"
if [[ "$1" == */scripts/validate_clean_clone_acceptance.py ]]; then
  command="$2"
  printf 'python-validator:%s cvd=%s\n' "$command" "$cvd" >> "${FAKE_FLOW_LOG}"
  shift 2
  if [ "$command" = result ]; then
    [ "$cvd" = 0 ]
    output=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --output) output="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '{"status":"ok","cuda_visible_devices":"0"}\n' > "$output"
    exit 0
  fi
  [ "$cvd" = '<unset>' ]
  if [ "$command" = dry-run ]; then
    run_root_output=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --run-root-output) run_root_output="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '%s/outputs/fake-fixture\n' "$PWD" > "$run_root_output"
  fi
  exit 0
fi
if [ "$1" = -m ]; then
  printf 'python-module:%s cvd=%s\n' "$2" "$cvd" >> "${FAKE_FLOW_LOG}"
  [ "$cvd" = '<unset>' ]
  exit 0
fi
exit 70
""",
    )
    _write_executable(
        template / ".venv" / "bin" / "real2sim-artifacts",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'artifacts cvd=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
[[ ! -v CUDA_VISIBLE_DEVICES ]]
printf '{"status":"ok","artifacts":[]}\n'
""",
    )
    _write_executable(
        template / ".venv" / "bin" / "real2sim-doctor",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'doctor cvd=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
[[ ! -v CUDA_VISIBLE_DEVICES ]]
printf '{"status":"ok"}\n'
""",
    )
    _write_executable(
        template / ".venv" / "bin" / "real2sim-run",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'run:fixture-dry-run cvd=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
[[ ! -v CUDA_VISIBLE_DEVICES ]]
printf '{"schema_version":2,"reproducibility_mode":"fixture","expected_rollouts":1,"config":"%s/config/release/fixture_v2.yaml","runner":"%s/scripts/run_policy_batch.sh","run_root":"%s/outputs/fake-fixture"}\n' "$PWD" "$PWD" "$PWD"
""",
    )
    _write_executable(
        template / ".venv" / "bin" / "real2sim-download-release",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'download-release cvd=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
exit 72
""",
    )
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'git:%s cvd=%s\n' "$*" "${CUDA_VISIBLE_DEVICES-<unset>}" >> "${FAKE_FLOW_LOG}"
if [ "$1" = clone ]; then
  destination="${!#}"
  cp -a "${FAKE_CLONE_TEMPLATE}/." "$destination"
  mkdir -p "$destination/.git"
  exit 0
fi
if [ "$1" = -C ]; then
  shift 2
fi
case "$1" in
  checkout|status) exit 0 ;;
  rev-parse) printf '%s\n' "${FAKE_COMMIT}" ;;
  diff-tree)
    [ "$*" = "diff-tree --check --root -m --no-commit-id -r ${FAKE_COMMIT}" ]
    ;;
  *) exit 71 ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_CLONE_TEMPLATE": str(template),
            "FAKE_COMMIT": VALID_COMMIT,
            "FAKE_FLOW_LOG": str(flow_log),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    result = _run_wrapper(
        "--source",
        "fake://clean-repository",
        "--commit",
        VALID_COMMIT,
        "--work-root",
        str(work_root),
        "--mode",
        "fixture",
        "--timestamp",
        "20260824_120000",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS mode=fixture" in result.stdout
    run_roots = list(work_root.glob("real2sim-acceptance.*"))
    assert len(run_roots) == 1
    evidence = run_roots[0] / "evidence"
    assert (evidence / "accelerator-environment.txt").read_text(
        encoding="utf-8"
    ) == "CUDA_VISIBLE_DEVICES=0\n"
    assert json.loads(
        (evidence / "acceptance-result.json").read_text(encoding="utf-8")
    )["cuda_visible_devices"] == "0"
    observed = flow_log.read_text(encoding="utf-8").splitlines()
    required_in_order = [
        "bootstrap cvd=<unset>",
        "python-validator:identity cvd=<unset>",
        "python-module:compileall cvd=<unset>",
        "python-module:build cvd=<unset>",
        "python-module:pytest cvd=<unset>",
        "artifacts cvd=<unset>",
        "doctor cvd=<unset>",
        "run:fixture-dry-run cvd=<unset>",
        "python-validator:dry-run cvd=<unset>",
        "python-validator:result cvd=0",
    ]
    positions = [observed.index(item) for item in required_in_order]
    assert positions == sorted(positions)
    assert any(
        " diff-tree --check --root -m --no-commit-id -r " in line
        for line in observed
    )


def test_identity_accepts_only_clone_cwd_and_clone_python(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "prefix", str(REPO_ROOT / ".venv"))

    VALIDATOR.validate_identity(REPO_ROOT)

    monkeypatch.setattr(sys, "prefix", str(REPO_ROOT.parent / "unrelated-venv"))
    with pytest.raises(VALIDATOR.AcceptanceValidationError, match="not clone-local"):
        VALIDATOR.validate_identity(REPO_ROOT)


def test_artifact_validation_requires_exact_unique_ids(tmp_path: Path):
    expected = ["artifact-a", "artifact-b"]

    def write(ids: list[str]) -> Path:
        path = tmp_path / "artifacts.json"
        path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "artifacts": [
                        {
                            "id": artifact_id,
                            "sha256": "0" * 64,
                            "size": 0,
                            "files": 1,
                        }
                        for artifact_id in ids
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    VALIDATOR.validate_artifacts(write(expected), expected)

    with pytest.raises(VALIDATOR.AcceptanceValidationError, match="differ"):
        VALIDATOR.validate_artifacts(write(["artifact-a"]), expected)
    with pytest.raises(VALIDATOR.AcceptanceValidationError, match="duplicate"):
        VALIDATOR.validate_artifacts(
            write(["artifact-a", "artifact-a"]), expected
        )


def test_dry_run_rejects_a_run_root_outside_the_clone(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "config" / "release").mkdir(parents=True)
    (repo / "outputs").mkdir()
    payload = {
        "schema_version": 2,
        "reproducibility_mode": "fixture",
        "expected_rollouts": 1,
        "config": str(repo / "config" / "release" / "fixture.yaml"),
        "runner": str(repo / "scripts" / "run_policy_batch.sh"),
        "run_root": str(tmp_path / "escaped-output"),
    }
    evidence = tmp_path / "dry-run.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VALIDATOR.AcceptanceValidationError, match="run root"):
        VALIDATOR.validate_dry_run(
            evidence,
            repo_root=repo,
            mode="fixture",
            expected_rollouts=1,
            run_root_output=tmp_path / "run-root.txt",
        )


def test_summary_enforces_configured_success_count_and_matching_rate(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    for name in VALIDATOR.RUN_EVIDENCE_FILES:
        (run_root / name).write_bytes(b"evidence\n")

    def write_summary(count: int, rate: float) -> None:
        (run_root / "rollout_success_summary.json").write_text(
            json.dumps(
                {
                    "official_success_field": "new_final_success",
                    "total_runs": 10,
                    "new_final_success_count": count,
                    "new_final_success_rate": rate,
                }
            ),
            encoding="utf-8",
        )

    write_summary(5, 0.5)
    with pytest.raises(
        VALIDATOR.AcceptanceValidationError, match="threshold was not met"
    ):
        VALIDATOR.validate_summary(
            run_root,
            expected_rollouts=10,
            min_successes=6,
            evidence_output=tmp_path / "failed-threshold",
        )

    write_summary(5, 0.6)
    with pytest.raises(
        VALIDATOR.AcceptanceValidationError, match="count and rate are inconsistent"
    ):
        VALIDATOR.validate_summary(
            run_root,
            expected_rollouts=10,
            min_successes=5,
            evidence_output=tmp_path / "failed-rate",
        )

    write_summary(5, 0.5)
    accepted = tmp_path / "accepted"
    VALIDATOR.validate_summary(
        run_root,
        expected_rollouts=10,
        min_successes=5,
        evidence_output=accepted,
    )
    manifest = json.loads(
        (accepted / "run-evidence.json").read_text(encoding="utf-8")
    )
    assert manifest["min_successes"] == 5
    assert manifest["new_final_success_count"] == 5
    assert manifest["new_final_success_rate"] == 0.5


def test_full_result_records_success_floor_and_caller_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "acceptance-result.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    VALIDATOR.write_result(
        output=output,
        mode="full",
        commit=VALID_COMMIT,
        repo_root=REPO_ROOT,
        run_root=tmp_path / "run",
        min_successes=4,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["min_successes"] == 4
    assert payload["cuda_visible_devices"] == "0"
    with pytest.raises(
        VALIDATOR.AcceptanceValidationError,
        match="requires minimum successes",
    ):
        VALIDATOR.write_result(
            output=output,
            mode="full",
            commit=VALID_COMMIT,
            repo_root=REPO_ROOT,
            run_root=tmp_path / "run",
            min_successes=None,
        )
