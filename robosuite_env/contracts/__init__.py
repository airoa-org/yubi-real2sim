# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts for simulation assets, scenes, and workers.

Each contract has a narrow owner:

* :mod:`assets` validates repository-owned runtime asset bundles.
* :mod:`provenance` holds the canonical renderer code-identity primitives
  shared by the rerender package and the launcher.
* :mod:`scene` validates one 3DGS and table-collision scene bundle.
* :mod:`worker` binds an official worker to launcher-authorized arguments.

Import concrete APIs from their owning module instead of this package root.
"""
