# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The vulture baseline check: new findings fail, listed ones pass, stale entries are reported."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_vulture_baseline.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_vulture_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load_check()

LISTED = "robosuite_env/example.py: unused variable 'listed'"
NEW = "robosuite_env/example.py: unused function 'added'"


def _run(tmp_path: Path, found: Counter, baseline_text: str) -> int:
    baseline = tmp_path / "vulture_baseline.txt"
    baseline.write_text(baseline_text, encoding="utf-8")
    return check.main(findings=lambda: found, baseline_path=baseline)


def test_a_new_unmarked_finding_fails(tmp_path, capsys):
    assert _run(tmp_path, Counter([LISTED, NEW]), f"{LISTED}\n") == 1
    assert f"new finding: {NEW}" in capsys.readouterr().err


def test_a_listed_finding_passes(tmp_path, capsys):
    assert _run(tmp_path, Counter([LISTED]), f"# comment\n\n{LISTED}\n") == 0
    assert "1 listed, 0 stale entries" in capsys.readouterr().out


def test_a_stale_baseline_entry_is_reported_and_passes(tmp_path, capsys):
    assert _run(tmp_path, Counter(), f"{LISTED}\n") == 0
    assert f"stale entry: {LISTED}" in capsys.readouterr().out


def test_a_stale_entry_does_not_excuse_a_new_finding(tmp_path, capsys):
    assert _run(tmp_path, Counter([NEW]), f"{LISTED}\n") == 1
    captured = capsys.readouterr()
    assert f"new finding: {NEW}" in captured.err
    assert f"stale entry: {LISTED}" in captured.out


def test_entries_count_repeated_findings():
    assert check.compare(Counter({LISTED: 2}), Counter({LISTED: 1})) == ([LISTED], [])
    assert check.compare(Counter({LISTED: 1}), Counter({LISTED: 2})) == ([], [LISTED])
    assert check.compare(Counter({LISTED: 2}), Counter({LISTED: 2})) == ([], [])


def test_a_scan_error_fails(tmp_path, capsys):
    def broken():
        raise RuntimeError("vulture could not scan every file")

    baseline = tmp_path / "vulture_baseline.txt"
    baseline.write_text("", encoding="utf-8")
    assert check.main(findings=broken, baseline_path=baseline) == 1
    assert "could not scan" in capsys.readouterr().err


def test_the_checked_in_baseline_names_files_not_lines():
    entry = re.compile(r"^[\w./-]+\.py: unused (attribute|class|function|import|method|property|variable) '\w+'$")
    text = (REPO_ROOT / "scripts" / "vulture_baseline.txt").read_text(encoding="utf-8")
    entries = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert entries == sorted(entries)
    for line in entries:
        assert entry.match(line), line
