# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Fail when a vulture noqa mark (V101-V107, V201 or a bare noqa) hides no finding.

A mark goes stale when its definition moves or scanned code starts to read it.
A reader outside the scanned code stays unchecked. Run from the repository root with vulture 2.16.
"""

import io
import sys
import tokenize

from vulture import core, lines, noqa, utils
from vulture.config import make_config


class MarkCheck(core.Vulture):
    """Vulture run that records the marks and each finding a mark hides."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.marks = []
        self.hidden = []

    def scan(self, code, filename=""):
        super().scan(code, filename)
        if str(filename).startswith("whitelists"):  # vulture's bundled whitelists
            return
        try:
            comments = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT]
        except (tokenize.TokenError, SyntaxError):
            return  # vulture already reports the file
        for token in comments:
            match = noqa.NOQA_REGEXP.search(token.string)
            if match:  # ruff's F401/F841 stay unmapped: ruff owns those marks.
                self.marks += [(utils.format_path(filename), token.start[0], c) for c in noqa._parse_error_codes(match)]

    def _define(self, collection, name, first_node, last_node=None, message="",
                confidence=core.DEFAULT_CONFIDENCE, ignore=None):
        lineno = lines.get_first_line_number(first_node)
        code = core.ERROR_CODES[collection.typ]
        if not ((ignore and ignore(self.filename, name)) or core._match(name, self.ignore_names)) \
                and noqa.ignore_line(self.noqa_lines, lineno, code):
            self.hidden.append((utils.format_path(self.filename), lineno, code, name, confidence))
        super()._define(collection, name, first_node, last_node, message, confidence, ignore)


def main() -> int:
    config = make_config([])
    check = MarkCheck(ignore_names=config["ignore_names"], ignore_decorators=config["ignore_decorators"])
    check.scavenge(config["paths"], exclude=config["exclude"])
    if check.exit_code != utils.ExitCode.NoDeadCode:
        print("vulture could not scan every file", file=sys.stderr)
        return 1
    live = {
        (path, lineno, code)
        for path, lineno, code, name, confidence in check.hidden
        if confidence >= config["min_confidence"] and (code == "V201" or name not in check.used_names)
    }
    live_lines = {(path, lineno) for path, lineno, _ in live}
    codes = set(core.ERROR_CODES.values())
    checked, problems = 0, []
    for path, lineno, code in sorted(check.marks):
        if code == "all":
            checked += 1
            if (path, lineno) not in live_lines:
                problems.append(f"{path}:{lineno}: bare noqa hides no vulture finding")
        elif code in codes:
            checked += 1
            if (path, lineno, code) not in live:
                problems.append(f"{path}:{lineno}: noqa {code} hides no vulture finding")
        elif any(part.upper().startswith("V") for part in code.split()):
            problems.append(f"{path}:{lineno}: vulture does not apply noqa {code}")
    if problems:
        print("Delete each mark below, or the definition if nothing reads it:", *problems, sep="\n", file=sys.stderr)
        return 1
    print(f"All {checked} vulture noqa marks hide a finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
