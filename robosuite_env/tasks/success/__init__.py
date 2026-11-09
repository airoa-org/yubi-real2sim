# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Task success criteria used by rollout workers."""

from .box_pack import BoxPackCriterion
from .cup_on_plate import CupOnPlateCriterion
from .registry import available_criteria, make_success_criterion, summarize_success_records

__all__ = [
    "BoxPackCriterion",
    "CupOnPlateCriterion",
    "available_criteria",
    "make_success_criterion",
    "summarize_success_records",
]
