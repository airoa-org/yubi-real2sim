# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Each release condition MANIFEST.json pins the bytes of the files it names.

A file entry names its tracked copy with ``path``, or, without ``path``, the
copy stored next to the manifest under the entry's name. ``sha256`` is the
tracked bytes. ``recorded_sha256`` is present only where the tracked bytes
were edited after the published runs read the file; it keeps the digest those
runs recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = sorted((REPO_ROOT / "assets" / "release").glob("*/*/*/MANIFEST.json"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned_files():
    for manifest in MANIFESTS:
        conditions = json.loads(manifest.read_text(encoding="utf-8"))
        for condition, entry in conditions.items():
            for name, record in entry["files"].items():
                tracked = REPO_ROOT / record["path"] if "path" in record else manifest.parent / name
                label = f"{manifest.relative_to(REPO_ROOT)}: {condition}/{name}"
                yield label, record, tracked


def test_every_pinned_path_is_the_tracked_file():
    failures = []
    checked = 0
    for label, record, tracked in _pinned_files():
        if not tracked.is_file():
            failures.append(f"{label}: {tracked.relative_to(REPO_ROOT)} is missing")
            continue
        checked += 1
        if _sha256(tracked) != record["sha256"]:
            failures.append(f"{label}: sha256 does not match {tracked.relative_to(REPO_ROOT)}")
    assert MANIFESTS and checked
    assert not failures, "\n".join(failures)


def test_a_recorded_digest_is_kept_only_for_an_edited_copy():
    recorded = 0
    for label, record, _ in _pinned_files():
        if "recorded_sha256" not in record:
            continue
        recorded += 1
        digest = record["recorded_sha256"]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), label
        assert digest != record["sha256"], label
    assert recorded
