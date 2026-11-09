# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Retired source dependencies stay replayable but are no longer hashed."""

from __future__ import annotations


def test_retired_openpi_source_dependency_is_recorded_not_hashed(tmp_path):
    """Profiles recorded with an OpenPI source checkout keep listing third_party/openpi.

    Provenance must neither require nor hash that checkout (nothing runs from
    it), even when a leftover clone exists.
    """

    import subprocess

    from real2sim.provenance import RETIRED_SOURCE_DEPENDENCIES, collect_source_dependencies

    assert RETIRED_SOURCE_DEPENDENCIES == ("third_party/openpi", "third_party/robosuite")
    leftover = tmp_path / "third_party" / "openpi"
    leftover.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(leftover)], check=True)
    [record] = collect_source_dependencies(
        tmp_path, ["{}/third_party/openpi".format(tmp_path)], production=False
    )
    assert record == {
        "path": str(leftover),
        "retired": True,
        "commit": None,
        "dirty": None,
        "remote": None,
    }
