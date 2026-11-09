#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Stable command-line entry point for the structured rollout runtime.

Implementation lives under :mod:`robosuite_env.rollout`; launchers start it as
``python -m robosuite_env.sim`` from the checkout root, so ``robosuite_env``
resolves from that checkout (the target of its editable install) without a
``sys.path`` insert.
"""

from __future__ import annotations

import runpy


def main() -> int:
    """Execute the rollout worker with the original command-line arguments."""

    # alter_sys=True makes sys.modules["__main__"] the worker module while it
    # runs; the rollout stages receive that module object and read its globals.
    runpy.run_module("robosuite_env.rollout.worker", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
