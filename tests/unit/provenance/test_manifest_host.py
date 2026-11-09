# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The manifest says which machine ran it, and under which scheduler job."""

from __future__ import annotations

import os
from pathlib import Path
import platform

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

from real2sim.provenance import build_run_manifest  # noqa: E402


#: Scheduler-shaped names. The code matches on the shape of the name rather
#: than on a particular scheduler, so any batch system is recorded and the
#: tests can use a made-up one.
JOB = "EXAMPLESCHED_JOBID"
QUEUE = "EXAMPLESCHED_QUEUE"
NODEFILE = "EXAMPLESCHED_NODEFILE"


def _manifest(**overrides):
    arguments = dict(
        repo_root=REPO_ROOT,
        config={},
        environment={},
        artifact_manifest_path="artifacts/manifest.yaml",
        artifact_manifest_sha256="0" * 64,
        artifacts=[],
        lockfiles=[],
        source_dependencies=[],
        accelerator_profile={"kind": "cpu"},
        production=False,
    )
    arguments.update(overrides)
    return build_run_manifest(**arguments)


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch):
    for name in list(os.environ):
        if name.endswith(("_JOBID", "_JOB_ID", "_QUEUE", "_NODEFILE")):
            monkeypatch.delenv(name, raising=False)


def test_the_manifest_names_the_machine_that_ran_it():
    # The GPUs are already recorded by UUID; the host they sit in belongs at the
    # same grain. Without it a finished run cannot be matched against what else
    # was running on its node.
    assert _manifest()["host"]["hostname"] == platform.node()


def test_no_scheduler_no_key():
    # A workstation run has no scheduler. An empty dict in every manifest would
    # read as "the scheduler told us nothing" rather than "there was none".
    assert "scheduler" not in _manifest()["host"]


def test_the_job_is_recorded_when_a_scheduler_names_one(monkeypatch):
    monkeypatch.setenv(JOB, "1234567.head")
    monkeypatch.setenv(QUEUE, "example_queue")

    scheduler = _manifest()["host"]["scheduler"]

    assert scheduler == {JOB: "1234567.head", QUEUE: "example_queue"}
    # Only what was set: an unset node file is absent, not empty.
    assert NODEFILE not in scheduler


def test_the_scheduler_is_recognised_by_the_shape_of_the_name(monkeypatch):
    # Named by shape rather than by scheduler, so a site running a different
    # batch system is recorded too. Unrelated variables must not be swept in.
    monkeypatch.setenv("SOMETHING_JOB_ID", "42")
    monkeypatch.setenv("MESSAGE_QUEUE_HOST", "broker.example")
    monkeypatch.setenv("lowercase_jobid", "no")

    scheduler = _manifest()["host"]["scheduler"]

    assert scheduler == {"SOMETHING_JOB_ID": "42"}


def test_the_scheduler_is_read_from_the_process_not_the_run_environment(monkeypatch):
    # The run's own environment is an allowlist that carries no scheduler
    # variables, and the launch scripts strip behaviour-changing ones before
    # calling the CLI. These belong to the process that submitted the work, so
    # they are read from os.environ and stay out of what the workers are given.
    monkeypatch.setenv(JOB, "7654321.head")

    manifest = _manifest(environment={"PATH": os.environ.get("PATH", "")})

    assert manifest["host"]["scheduler"][JOB] == "7654321.head"
    assert JOB not in manifest["resolved_environment"]
