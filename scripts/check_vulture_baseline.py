# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Fail when vulture reports an unmarked finding that the baseline does not list.

scripts/vulture_baseline.txt lists the unmarked findings that main already had,
one per line as ``<path>: <message>``: vulture's report without the line number
and confidence, so an edit elsewhere in the file does not move an entry. A
finding that occurs twice in one file is listed twice.

The check fails on a finding the baseline does not list. A baseline entry that
no longer occurs is printed but does not fail, so a stacked change that removes
a finding passes whether or not it also removes the entry; remove the entry
when you can. Never add an entry for a new finding: delete the code or mark it.
Run from the repository root with vulture 2.16.
"""

from collections import Counter
from pathlib import Path
import sys

BASELINE = Path("scripts/vulture_baseline.txt")


def read_baseline(text: str) -> Counter:
    """Count the entries in baseline text; blank lines and # comments are skipped."""

    lines = (line.strip() for line in text.splitlines())
    return Counter(line for line in lines if line and not line.startswith("#"))


def compare(found: Counter, baseline: Counter) -> tuple[list[str], list[str]]:
    """Return the findings the baseline lacks and the entries that no longer occur."""

    new = [key for key in sorted(found) for _ in range(found[key] - baseline[key])]
    stale = [key for key in sorted(baseline) for _ in range(baseline[key] - found[key])]
    return new, stale


def vulture_findings() -> Counter:
    """Unmarked findings with the settings in pyproject.toml, keyed without line numbers."""

    from vulture import core, utils
    from vulture.config import make_config

    config = make_config([])
    scan = core.Vulture(ignore_names=config["ignore_names"], ignore_decorators=config["ignore_decorators"])
    scan.scavenge(config["paths"], exclude=config["exclude"])
    if scan.exit_code != utils.ExitCode.NoDeadCode:
        raise RuntimeError("vulture could not scan every file")
    items = scan.get_unused_code(min_confidence=config["min_confidence"], sort_by_size=config["sort_by_size"])
    return Counter(f"{utils.format_path(item.filename)}: {item.message}" for item in items)


def main(findings=vulture_findings, baseline_path: Path = BASELINE) -> int:
    try:
        found = findings()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    baseline = read_baseline(baseline_path.read_text(encoding="utf-8"))
    new, stale = compare(found, baseline)
    if stale:
        print(
            f"Baseline entries that no longer occur; remove them from {baseline_path}:",
            *(f"  stale entry: {key}" for key in stale),
            sep="\n",
        )
    if new:
        print(
            "Delete each new finding, or mark it and name its reader (see README.md, Lint checks).",
            *(f"new finding: {key}" for key in new),
            sep="\n",
            file=sys.stderr,
        )
        return 1
    print(f"No new vulture findings ({sum(found.values())} listed, {len(stale)} stale entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
