#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Reconstructed task objects as native robosuite XML objects."""
import os

from robosuite.models.objects import MujocoXMLObject

from ..contracts.assets import validate_repository_assets
from ..paths import ASSETS_ROOT


_VALIDATED_OBJECT_BUNDLES = {
    "cup.xml": "cup",
    "plate.xml": "plate",
    "box.xml": "box",
    "tape.xml": "tape",
    "pen.xml": "pen",
    "lan.xml": "lan",
}


def _validate_object_bundle(bundle):
    validate_repository_assets(ASSETS_ROOT, bundle_ids=(bundle,))


class _MeshObj(MujocoXMLObject):
    def __init__(self, name, xml, *, duplicate_collision_geoms=True):
        asset_xml = os.fspath(ASSETS_ROOT / xml)
        bundle = _VALIDATED_OBJECT_BUNDLES.get(xml)
        if bundle is not None:
            _validate_object_bundle(bundle)
        super().__init__(
            asset_xml, name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all", duplicate_collision_geoms=duplicate_collision_geoms,
        )


class CupObject(_MeshObj):
    def __init__(self, name="cup"):
        super().__init__(name, "cup.xml")


class PlateObject(_MeshObj):
    def __init__(self, name="plate"):
        super().__init__(name, "plate.xml")








class BoxObject(_MeshObj):
    """The open cardboard box the packing task fills.

    The reconstruction keeps the lid standing, so the body is taller than the
    walls; ``robosuite_env/assets/manifest.json`` records both extents.
    """

    def __init__(self, name="box"):
        super().__init__(name, "box.xml")


class TapeObject(_MeshObj):
    """Packing tape roll, 8 cm outer diameter."""

    def __init__(self, name="tape"):
        super().__init__(name, "tape.xml")


class PenObject(_MeshObj):
    """Ball point pen, 14.5 cm long."""

    def __init__(self, name="pen"):
        super().__init__(name, "pen.xml")


class LanCableObject(_MeshObj):
    """LAN cable, coiled as photographed; 30 cm straightened.

    The cable is rigid here. A convex decomposition cannot bend, so the sim
    models the coil's shape but not its compliance.
    """

    def __init__(self, name="lan"):
        super().__init__(name, "lan.xml")


OBJECT_ASSET_REGISTRY = {
    "cup": CupObject,
    "plate": PlateObject,
    "box": BoxObject,
    "tape": TapeObject,
    "pen": PenObject,
    "lan": LanCableObject,
}
