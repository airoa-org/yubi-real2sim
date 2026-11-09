# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_PROFILES = tuple(sorted((REPO_ROOT / "config" / "release").glob("*/*/*.yaml")))


def _joined(*parts: str) -> str:
    return "".join(parts)


def test_public_profiles_use_explicit_visible_gpu_slots():
    assert len(RELEASE_PROFILES) == 2
    retired_gpu_variable = _joined("REAL2SIM_", "GPU_IDS")

    for profile in RELEASE_PROFILES:
        source = profile.read_text(encoding="utf-8")
        loaded = yaml.safe_load(source)
        runtime = loaded["runtime"]
        environment = loaded["reproducibility"]["environment"]

        # One 16 GB GPU runs every role; a two-GPU host overrides these locally.
        assert runtime["gs_cuda_devices"] == "0", profile
        assert runtime["policy_cuda_device"] == "0", profile
        assert runtime["sim_cuda_devices"] == "0", profile
        assert "CUDA_VISIBLE_DEVICES" in environment["allowlist"], profile
        assert "CUDA_VISIBLE_DEVICES" not in environment["required"], profile
        assert retired_gpu_variable not in source, profile


def test_environment_check_runs_two_workers_on_one_gpu(monkeypatch, tmp_path):
    from real2sim import launcher

    monkeypatch.setenv("REAL2SIM_INPUTS_ROOT", str(tmp_path))
    plan = launcher.load_rollout_config(
        REPO_ROOT / "config" / "release" / "fr3" / "flat" / "a_room_cup_cup1trim.yaml",
        timestamp="20261101_000000",
    )
    resolved = plan["resolved"]

    assert plan["expected_rollouts"] == 10
    assert resolved["runner"]["policy_batch_size"] == 2
    assert resolved["runner"]["parallel_repeats"] == 2
    # Below batch 10 the third-person video renders live, not post-hoc.
    assert resolved["render"]["enabled"] is True
    assert resolved["render"]["forced_posthoc"] is False
    environment = launcher.build_runner_environment(plan)
    assert environment["POLICY_BATCH_SIZE"] == "2"
    assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.45"
    assert environment["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"


def test_example_env_has_no_gpu_selector():
    active_lines = [
        line
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert active_lines == ["REAL2SIM_INPUTS_ROOT=/ABSOLUTE/PATH/TO/local-inputs"]


def test_public_instructions_use_the_config_driven_gpu_entrypoint():
    public_documents = (
        REPO_ROOT / "README.md",
        *(sorted((REPO_ROOT / "docs").glob("*.md"))),
    )
    for path in public_documents:
        source = path.read_text(encoding="utf-8")
        bash_blocks = re.findall(r"```bash\n(.*?)```", source, flags=re.DOTALL)
        assert all("scripts/run_policy_batch.sh" not in block for block in bash_blocks), path

    completed = subprocess.run(
        [str(REPO_ROOT / "scripts" / "run_policy_batch.sh"), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Supported public entry point:" in completed.stdout
    assert "real2sim-run" in completed.stdout
    assert "internal current-host process orchestration" in completed.stdout
    for role in ("GS_CUDA_DEVICES", "POLICY_CUDA_DEVICE", "SIM_CUDA_DEVICES"):
        assert role in completed.stdout
    assert "Example:\n  scripts/run_policy_batch.sh" not in completed.stdout


# SHA-256 digests of the two full 40-character revisions of non-public sources
# that the FR3 impedance port follows (#85 R2). A revision has 160 bits, so no
# one can recover it from its digest by trying values, and this file does not
# publish it. Only full-length revisions are matched. A shortened revision is
# not caught: Git abbreviations (7 to 12 characters) have few enough values to
# hash every one, so storing a digest of a prefix would publish the prefix.
_NON_PUBLIC_REVISION_DIGESTS = frozenset({
    "fdae9bb754a68fbdff6061f367b1cefa82ea613e547f8a885744344058fe32b2",
    "5196d2a11a4a2430a293dc68cd073e0f3bbd6fe1b9de86194ee93c77bff27d4c",
})
_IMPEDANCE_PORT = "robosuite_env/robots/fr3/impedance.py"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_tracked_files_do_not_carry_non_public_impedance_revisions():
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout.decode().split("\0")
    revision = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
    scanned, hits = set(), []
    for name in listed:
        path = REPO_ROOT / name
        if not name or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        scanned.add(name)
        text = data.decode("utf-8", errors="replace").lower()
        if any(_sha256(token) in _NON_PUBLIC_REVISION_DIGESTS for token in revision.findall(text)):
            hits.append(f"{name}: non-public source revision")
    # Whatever the size of the tree, the scan must read the file that held them.
    assert _IMPEDANCE_PORT in scanned
    assert hits == []
