# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Keep the resolver, bootstrap, and runtime toolchain in agreement."""

import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import yaml

from real2sim import support_matrix as support


ROOT = Path(__file__).resolve().parents[2]


def test_matrix_is_readable_before_python_dependencies_are_installed():
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "src/real2sim/support_matrix.py"), "shell"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "R2S_CUDA_VERSION=12.8" in result.stdout
    assert support.load_bootstrap_profile() == yaml.safe_load(
        support.DEFAULT_MATRIX.read_text()
    )["profiles"][support.DEFAULT_PROFILE]


def test_resolver_and_locked_wheels_match_the_support_matrix():
    profile = support.load_bootstrap_profile()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    uv = project["tool"]["uv"]
    indexes = {entry["name"]: entry for entry in uv["index"]}
    packages = {item["name"]: item for item in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    for name, version in (("torch", "2.7.1"), ("torchvision", "0.22.1")):
        assert f"{name}=={version}" in project["project"]["dependencies"]
        index = indexes[uv["sources"][name]["index"]]
        assert index["explicit"] is True
        assert index["url"] == profile["cuda"]["pytorch_index"]
        package = packages[name]
        assert package["source"]["registry"] == index["url"]
        assert package["version"] == f"{version}+cu128"
        assert any("cp311" in wheel["url"] and "x86_64" in wheel["url"] for wheel in package["wheels"])


def test_toolchain_documentation_matches_the_profile():
    profile = support.load_bootstrap_profile()
    cuda = profile["cuda"]
    version = f"{cuda['major']}.{cuda['minor']}"
    for name in ("README.md", "docs/configuration.md", ".env.example"):
        text = (ROOT / name).read_text()
        assert f"CUDA {version}" in text
        assert "CUDA 12.6" not in text
    assert profile["driver"]["minimum"] in (ROOT / "README.md").read_text()
    assert cuda["toolkit_path"] in (ROOT / ".env.example").read_text()


def test_container_image_matches_the_profile_and_the_uv_pin():
    profile = support.load_bootstrap_profile()
    cuda = profile["cuda"]
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    images = re.findall(r"^FROM (\S+)", dockerfile, flags=re.MULTILINE)
    assert len(images) == 2
    assert all(re.fullmatch(r"[^@\s]+:[^@\s]+@sha256:[0-9a-f]{64}", image) for image in images)
    uv_version = re.search(r'^UV_VERSION="([^"]+)"$', (ROOT / "scripts/bootstrap.sh").read_text(), re.MULTILINE)[1]
    assert f'EXPECTED_UV="{uv_version}"' in (ROOT / "scripts/check_host.sh").read_text()
    assert {image.split("@")[0] for image in images} == {
        f"ghcr.io/astral-sh/uv:{uv_version}",
        f"nvidia/cuda:{cuda['major']}.{cuda['minor']}.1-devel-ubuntu24.04",
    }
    assert f"gcc-{profile['compiler']['major']} g++-{profile['compiler']['major']}" in dockerfile
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in dockerfile
    # No repository content in the image: the context is docker/ and nothing is added from it.
    assert not re.search(r"^(ADD|COPY)(?! --from=)", dockerfile, flags=re.MULTILINE)
    assert sorted(path.name for path in (ROOT / "docker").iterdir()) == ["Dockerfile"]


def _nvcc(root):
    path = root / "bin/nvcc"
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.61'\n")
    path.chmod(0o755)
    return path


def test_nvcc_selection_prefers_explicit_home_then_path_then_profile(tmp_path):
    profile = support.load_bootstrap_profile()
    explicit = _nvcc(tmp_path / "explicit")
    on_path = _nvcc(tmp_path / "on-path")
    fallback = _nvcc(tmp_path / "fallback")
    profile["cuda"]["toolkit_path"] = str(fallback.parent.parent)
    env = {"PATH": str(on_path.parent), "CUDA_HOME": str(explicit.parent.parent)}
    assert support.find_nvcc(profile, env) == str(explicit)
    del env["CUDA_HOME"]
    assert support.find_nvcc(profile, env) == str(on_path)
    env["PATH"] = ""
    assert support.find_nvcc(profile, env) == str(fallback)
    env["CUDA_HOME"] = str(tmp_path / "missing")
    assert support.find_nvcc(profile, env) is None


@pytest.mark.parametrize("section,key,value", [
    ("cuda", "minor", 9),
    ("cuda", "pytorch_index", "https://example.invalid/changed"),
    ("cuda", "toolkit_path", "/opt/cuda"),
    ("driver", "minimum", "999.0"),
    ("compiler", "major", 14),
])
def test_profile_changes_invalidate_the_build_fingerprint(section, key, value):
    original = support.load_bootstrap_profile()
    changed = copy.deepcopy(original)
    changed[section][key] = value
    assert support.profile_fingerprint(original) != support.profile_fingerprint(changed)
    assert support.profile_fingerprint(original) == support.profile_fingerprint({"name": "display", **original})


@pytest.mark.parametrize("version,cuda", [("2.7.1+cu126", "12.6"), ("2.7.1+cpu", None)])
def test_torch_probe_rejects_the_wrong_wheel(monkeypatch, version, cuda):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda)))
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(__version__="0.22.1+cu128"))
    with pytest.raises(ValueError, match="expected torch/torchvision"):
        support.check_torch(support.load_bootstrap_profile())


def test_cuda_arch_flags_follow_the_list_or_else_the_visible_gpus(monkeypatch):
    torch = pytest.importorskip("torch")
    # The 8.9 GPU is first on the PCI bus, but CUDA lists the faster 12.0 GPU first by default.
    enumeration = {"PCI_BUS_ID": [(8, 9), (12, 0)], "FASTEST_FIRST": [(12, 0), (8, 9)]}

    def visible():
        ordered = enumeration[os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST")]
        selected = os.environ.get("CUDA_VISIBLE_DEVICES")
        return ordered if selected is None else [ordered[int(i)] for i in selected.split(",") if i]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(visible()))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: visible()[index])
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90", "sm_120"])

    def flags(**environment):
        for name in ("TORCH_CUDA_ARCH_LIST", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            monkeypatch.delenv(name, raising=False)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        return support.cuda_arch_flags()

    fastest = flags(CUDA_VISIBLE_DEVICES="0")
    assert "sm_120" in fastest and "sm_89" not in fastest
    on_first_bus = flags(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    assert "sm_89" in on_first_bus and "sm_120" not in on_first_bus
    assert flags(CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID") == fastest
    listed = flags(TORCH_CUDA_ARCH_LIST="8.9;12.0")
    for gpu, order in (("0", "PCI_BUS_ID"), ("1", "FASTEST_FIRST"), ("", "FASTEST_FIRST")):
        assert flags(
            TORCH_CUDA_ARCH_LIST="12.0 8.9", CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER=order
        ) == listed
    with pytest.raises(ValueError, match="set TORCH_CUDA_ARCH_LIST"):
        flags(CUDA_VISIBLE_DEVICES="")
