#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Convert the TRELLIS.2 office-object reconstructions into MuJoCo assets.

The input reconstructions are not part of this repository; point
``REAL2SIM_OFFICE_PREDICTIONS_ROOT`` at a local copy (default
``local-inputs/office-object-predictions``). The release installer downloads
the converted meshes, so running a profile does not need this script.

The reconstructions ship as a textured visual GLB plus a CoACD convex
decomposition (``part_NNN.obj``) that shares its coordinate frame. Neither is
usable as-is: the runtime reads OBJ (never GLB), MuJoCo is Z-up while glTF is
Y-up, and TRELLIS.2 coordinates sit in the canonical cube with no physical
scale at all -- "a 5 cm tape roll and a 30 cm container come out the same size".

This script therefore

* rotates both meshes from Y-up into MuJoCo's Z-up,
* moves the origin to the bottom-centre of the *collision* geometry, which is
  the convention ``robosuite_env/assets/cup.xml`` follows (``bottom_site`` at
  the body origin), and
* writes the visual mesh as OBJ + MTL + PNG and every convex hull as its own
  OBJ, the two roles the asset contract requires.

Scale stays out of the meshes: it is a per-``<mesh>`` attribute in the MJCF,
so the checked-in geometry remains canonical and the scale a profile actually
ran with is readable in one place. The factors this script prints are uniform
and chosen so no axis lands *under* its measured dimension (an object may come
out larger than measured, never smaller). The one exception is the cardboard box, which is measured on every axis
and scaled per axis to those measurements; see ``OBJECTS`` for why.

The checked-in ``robosuite_env/assets/box.xml`` is not this script's output.
It scales the box by 0.315980 0.315980 0.248269, puts the top site at
z = 0.240357, gives each of the 23 hulls its own density, gives the base slab
and the visual mesh ``mass="0"``, and declares no ``<inertial>``; the script
writes the per-axis scale, the collision top, density 120 kg/m^3 on every hull
and the thin-shell ``<inertial>``. ``tests/unit/tasks/test_office_box_mass.py``
pins the hull densities, the massless slab and visual mesh and the absence of
an ``<inertial>``; the asset contract (``robosuite_env/contracts/assets.py``
and ``robosuite_env/assets/manifest.json``) pins the top site. Running the
script for the box (it is in the default selection, and the default
``--asset-root`` is ``robosuite_env/assets``) overwrites box.xml and the box
meshes under ``trellis_office/`` with an MJCF that fails that test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh


PREDICTIONS = Path(
    os.environ.get(
        "REAL2SIM_OFFICE_PREDICTIONS_ROOT",
        str(Path(__file__).resolve().parents[1] / "local-inputs" / "office-object-predictions"),
    )
).expanduser()

#: Measured real-world dimensions, in metres, keyed by the mesh axis they
#: constrain *after* the Y-up -> Z-up rotation (x, y, z). ``None`` means the
#: dimension was not measured and places no lower bound on the scale.
#:
#: The mesh reference extents are measured on the canonical reconstruction.
#:
#: The cardboard box is the one object measured on all three axes, and it is
#: scaled per axis (``scale_mode``) rather than uniformly. Its reconstruction
#: comes from a single photo and gets the depth-to-width ratio wrong: its walls
#: are 0.8233 x 0.5723 x 0.4534 (ratio 1 : 0.70 : 0.55) against the measured
#: 23 x 21 x 11.5 cm (1 : 0.91 : 0.50). No uniform factor fits both, and the
#: "no axis under" rule picks the depth factor, which would make the box
#: 28.6 cm wide and 19.4 cm tall at the walls -- wider than the gap between the
#: Duo's parked hands, which the real box fits into.
#:
#: The box references are its *walls*: side wall to side wall (x) and front
#: wall to back wall (y) on the visual mesh, and base plane to wall top edge on
#: the collision hulls (z). Each is the smaller of the visual and collision
#: measurement, so neither lands under the measured size. The dust flaps and
#: the lid stand above the wall top edge (z = 0.487 above the lowest vertex)
#: and are not part of any measured dimension.
#:
#: ``base_plane`` fixes a rocking base. The reconstructed base is not flat: one
#: CoACD hull dips 0.0337 below the other eight floor hulls, which lie within
#: 0.0036 of one another, so the box rested on that one patch and settled
#: 5.8-7.8 deg off vertical. Every hull is clipped to the plane that far above
#: the lowest vertex, which gives one flat support across the whole footprint,
#: the way a real cardboard box stands.
OBJECTS = {
    # The MJCF this entry produces is not the checked-in box.xml; see the
    # module docstring.
    "box": {
        "source": "01_cardboard_box",
        "label": "cardboard box",
        "mesh_reference": (0.8233, 0.5723, 0.4534),
        "real_m": (0.23, 0.21, 0.115),
        "reference_note": (
            "outer wall extents: side wall to side wall and front wall to back "
            "wall on the visual mesh, base plane to wall top edge on the "
            "collision hulls; dust flaps and lid excluded"
        ),
        "scale_mode": "per_axis",
        "base_plane": 0.0337,
        "density": 120.0,
        # The box is a few millimetres of cardboard, but the CoACD hulls fill
        # its walls and floor solid, so density on the hulls put the mass low
        # and toward the front wall: centre of mass (-1, -43, 78) mm. Treating
        # the reconstructed surface as a uniform thin shell puts it at about
        # (-1, -31, 87) mm -- higher and nearer the raised lid, which is a
        # quarter of the surface. The mass stays what density 120 kg/m^3 gave
        # the hulls, so only its distribution changes.
        "inertial": {"model": "thin_shell_visual", "mass_kg": 0.2794},
        # The clipped floor hulls only reach the base plane over y -156..+9 mm
        # of a box whose walls stand on -184..+29 mm: the reconstruction's
        # floor curls up at the front and back edges, so the box pivoted on a
        # base 51 mm short of its walls and tipped back onto the lid at ~25
        # deg. A real box stands on its whole bottom panel, so one thin slab
        # spanning the wall footprint (every hull vertex at or below the wall
        # top) is added under the floor hulls.
        "base_slab": {"wall_top_m": 0.1155, "thickness_m": 0.003, "inset_m": 0.001},
    },
    "tape": {
        "source": "02_green_tape",
        "label": "packing tape roll",
        "mesh_reference": (1.0004, 1.0004, 0.6369),
        "real_m": (0.08, 0.08, None),
        "reference_note": "outer diameter on the two circular axes, roll width on the third",
        "density": 300.0,
    },
    "pen": {
        # This reconstruction is the ball point pen named in the box prompts.
        "source": "12_ballpoint_pen",
        "label": "ball point pen",
        "mesh_reference": (0.1165, 0.1130, 1.0018),
        "real_m": (None, None, 0.145),
        "reference_note": "barrel length along the long axis",
        "density": 300.0,
    },
    "lan": {
        "source": "04_lan_cable",
        "label": "LAN cable",
        "mesh_reference": (2.830,),
        "real_m": (0.30,),
        "reference_note": (
            "centreline length of the coiled cable (projected footprint area "
            "divided by the median tube diameter), matched to the cable's "
            "straightened length"
        ),
        "density": 400.0,
    },
}

#: glTF is Y-up, MuJoCo is Z-up: (x, y, z) -> (x, -z, y).
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0))


def uniform_scale(spec: dict) -> float:
    """Smallest uniform scale that leaves no measured axis under its dimension."""

    factors = [
        real / mesh
        for mesh, real in zip(spec["mesh_reference"], spec["real_m"])
        if real is not None
    ]
    if not factors:
        raise ValueError("an object needs at least one measured dimension")
    return max(factors)


def mesh_scale(spec: dict) -> np.ndarray:
    """Per-axis (x, y, z) mesh scale.

    Uniform unless the object opts into ``scale_mode: per_axis``, which needs a
    measured dimension on every axis and matches each one exactly.
    """

    mode = spec.get("scale_mode", "uniform")
    if mode == "uniform":
        return np.full(3, uniform_scale(spec))
    if mode != "per_axis":
        raise ValueError(f"unknown scale_mode {mode!r}")
    if len(spec["real_m"]) != 3 or any(real is None for real in spec["real_m"]):
        raise ValueError("per_axis scale needs a measured dimension on all three axes")
    return np.array(
        [real / mesh for mesh, real in zip(spec["mesh_reference"], spec["real_m"])]
    )


def thin_shell_mass_properties(
    vertices: np.ndarray, faces: np.ndarray, mass: float
) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and inertia of a uniform thin shell over a triangle mesh.

    Every triangle carries mass in proportion to its area. Returns the centre
    of mass and the inertia about it as MuJoCo's ``fullinertia`` order
    ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``, in the mesh's own units.
    """

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(area.sum())
    if not total > 0.0 or not mass > 0.0:
        raise ValueError("a thin shell needs a positive area and mass")
    density = float(mass) / total
    centre = (area[:, None] * (a + b + c) / 3.0).sum(axis=0) / total
    s = a + b + c
    # Integral of x x^T over a triangle: A / 12 (a a^T + b b^T + c c^T + s s^T).
    second = (
        np.einsum("n,ni,nj->ij", area, a, a)
        + np.einsum("n,ni,nj->ij", area, b, b)
        + np.einsum("n,ni,nj->ij", area, c, c)
        + np.einsum("n,ni,nj->ij", area, s, s)
    ) / 12.0
    moment = density * second - float(mass) * np.outer(centre, centre)
    inertia = np.trace(moment) * np.eye(3) - moment
    full = np.array(
        [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    )
    return centre, full


def clip_to_base_plane(parts: list[trimesh.Trimesh], height: float) -> list[trimesh.Trimesh]:
    """Clip every hull to ``z >= lowest vertex + height``.

    CoACD hulls are convex, so the clipped hull is the convex hull of the
    vertices on or above the plane plus the points where edges cross it: closed,
    convex, and ending exactly on the plane wherever the hull reached below it.
    """

    plane = min(float(part.bounds[0][2]) for part in parts) + height
    clipped = []
    for part in parts:
        if float(part.bounds[0][2]) >= plane:
            clipped.append(part)
            continue
        if float(part.bounds[1][2]) <= plane:
            continue
        vertices = np.asarray(part.vertices, dtype=float)
        edges = vertices[np.asarray(part.edges_unique)]
        z0, z1 = edges[:, 0, 2], edges[:, 1, 2]
        crossing = (z0 - plane) * (z1 - plane) < 0.0
        t = (plane - z0[crossing]) / (z1[crossing] - z0[crossing])
        cut = edges[crossing, 0] + t[:, None] * (edges[crossing, 1] - edges[crossing, 0])
        cut[:, 2] = plane
        kept = vertices[vertices[:, 2] >= plane]
        clipped.append(trimesh.convex.convex_hull(np.vstack([kept, cut])))
    return clipped


def load_visual(source: str) -> trimesh.Trimesh:
    mesh = trimesh.load(PREDICTIONS / source / "trellis2_only.glb", force="mesh")
    mesh.apply_transform(Y_UP_TO_Z_UP)
    return mesh


def load_parts(source: str) -> list[trimesh.Trimesh]:
    paths = sorted((PREDICTIONS / source / "trellis2_only_coacd_parts").glob("part_*.obj"))
    if not paths:
        raise FileNotFoundError(f"no CoACD parts under {source}")
    parts = []
    for path in paths:
        part = trimesh.load(path, force="mesh")
        part.apply_transform(Y_UP_TO_Z_UP)
        parts.append(part)
    return parts


def base_slab(parts: list[trimesh.Trimesh], spec: dict, scale: np.ndarray) -> trimesh.Trimesh:
    """One thin convex hull under the walls, so the object stands on its whole base.

    ``parts`` are already at the MJCF origin (base plane at z = 0) in canonical
    units; ``spec`` gives the wall top, thickness and inset in metres, which
    ``scale`` converts. The slab spans the xy extent of every hull vertex at or
    below the wall top, shrunk by the inset so it stays inside the walls.
    """

    slab = spec["base_slab"]
    scale = np.asarray(scale, dtype=float)
    wall_top = float(slab["wall_top_m"]) / float(scale[2])
    thickness = float(slab["thickness_m"]) / float(scale[2])
    inset = np.array([float(slab["inset_m"]) / float(scale[0]), float(slab["inset_m"]) / float(scale[1])])
    vertices = np.vstack([np.asarray(part.vertices, dtype=float) for part in parts])
    walls = vertices[vertices[:, 2] <= wall_top]
    lower = walls[:, :2].min(axis=0) + inset
    upper = walls[:, :2].max(axis=0) - inset
    base = float(vertices[:, 2].min())
    box = trimesh.creation.box(extents=[upper[0] - lower[0], upper[1] - lower[1], thickness])
    box.apply_translation([(lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, base + thickness / 2.0])
    return box


def build(name: str, spec: dict, out_root: Path) -> dict:
    visual = load_visual(spec["source"])
    parts = load_parts(spec["source"])
    if "base_plane" in spec:
        parts = clip_to_base_plane(parts, float(spec["base_plane"]))

    # The collision hulls are what rests on the table, so they define the
    # origin the MJCF's bottom_site sits at.
    lower = np.min([part.bounds[0] for part in parts], axis=0)
    upper = np.max([part.bounds[1] for part in parts], axis=0)
    offset = np.array([
        -(lower[0] + upper[0]) / 2.0,
        -(lower[1] + upper[1]) / 2.0,
        -lower[2],
    ])
    visual.apply_translation(offset)
    for part in parts:
        part.apply_translation(offset)

    scale = mesh_scale(spec)
    if "base_slab" in spec:
        # Appended last, so every existing hull keeps its index and bytes.
        parts = [*parts, base_slab(parts, spec, scale)]
    mesh_dir = out_root / "trellis_office"
    coll_dir = mesh_dir / "coll"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    coll_dir.mkdir(parents=True, exist_ok=True)

    # trimesh names the exported texture after the material, so every object
    # has to carry its own material name or they all write "material_0.png"
    # and the last export silently overwrites the rest.
    material = getattr(visual.visual, "material", None)
    if material is None:
        raise RuntimeError(f"{name}: the reconstruction carries no texture")
    material.name = name
    visual.export(mesh_dir / f"{name}_vis.obj", include_texture=True, mtl_name=f"{name}_vis.mtl")
    # MuJoCo binds the texture through the MJCF's own <texture>/<material>, so
    # the exported MTL is dead weight; trellis_cup_plate ships OBJ + PNG only.
    (mesh_dir / f"{name}_vis.mtl").unlink(missing_ok=True)
    for index, part in enumerate(parts):
        part.export(coll_dir / f"{name}_{index:03d}.obj")

    scaled_lower = (lower + offset) * scale
    scaled_upper = (upper + offset) * scale
    radius = float(np.max(np.abs(np.stack([scaled_lower[:2], scaled_upper[:2]]))))
    inertial = None
    if "inertial" in spec:
        model = spec["inertial"]["model"]
        if model != "thin_shell_visual":
            raise ValueError(f"{name}: unknown inertial model {model!r}")
        mass = float(spec["inertial"]["mass_kg"])
        # The visual vertices already carry the origin offset; the MJCF scales
        # the mesh, so the shell is measured at that scale.
        centre, full = thin_shell_mass_properties(
            np.asarray(visual.vertices, dtype=float) * scale, visual.faces, mass
        )
        inertial = {
            "model": model,
            "mass_kg": mass,
            "pos_m": [round(float(v), 6) for v in centre],
            "fullinertia_kg_m2": [float(f"{v:.6e}") for v in full],
        }
    report = {
        "name": name,
        "label": spec["label"],
        "source": spec["source"],
        "scale": (
            round(float(scale[0]), 6)
            if np.all(scale == scale[0])
            else [round(float(v), 6) for v in scale]
        ),
        "hulls": len(parts),
        "visual_faces": int(len(visual.faces)),
        "bottom_site_m": [0.0, 0.0, 0.0],
        "top_site_m": [0.0, 0.0, round(float(scaled_upper[2]), 6)],
        "horizontal_radius_site_m": [round(radius, 6), 0.0, 0.0],
        "scaled_extent_m": [round(float(v), 6) for v in (scaled_upper - scaled_lower)],
        "reference_note": spec["reference_note"],
        "density": spec["density"],
    }
    if inertial is not None:
        report["inertial"] = inertial
    return report


MJCF_TEMPLATE = """<mujoco model="{name}">
  <asset>
    <texture name="{name}_tex" type="2d" file="trellis_office/{name}.png"/>
    <material name="{name}_mat" texture="{name}_tex"/>
    <mesh name="p_vis" file="trellis_office/{name}_vis.obj" scale="{scale}"/>
{collision_meshes}  </asset>
  <worldbody>
    <body>
      <body name="object">
{inertial}{collision_geoms}        <geom name="p_vis_g" type="mesh" mesh="p_vis" material="{name}_mat" group="1" contype="0" conaffinity="0"/>
      </body>
      <site name="bottom_site" pos="0 0 0" size="0.003" rgba="1 0 0 0"/>
      <site name="top_site" pos="0 0 {top:.4f}" size="0.003" rgba="0 1 0 0"/>
      <site name="horizontal_radius_site" pos="{radius:.4f} 0 0" size="0.003" rgba="0 0 1 0"/>
    </body>
  </worldbody>
</mujoco>
"""

#: Contact parameters copied from robosuite_env/assets/cup.xml so a new object
#: enters the same contact regime the tuned cup/plate rollouts run in.
GEOM_TEMPLATE = (
    '        <geom name="p_c{index}" type="mesh" mesh="p_coll_{index}" group="0" '
    'condim="4" friction="1 0.05 0.0001" solref="0.005 1" solimp="0.95 0.99 0.001" '
    'density="{density:g}" rgba="0.5 0.5 0.5 0"/>\n'
)


def write_mjcf(report: dict, asset_root: Path) -> Path:
    name = report["name"]
    factors = report["scale"] if isinstance(report["scale"], list) else [report["scale"]] * 3
    scale = " ".join(f"{float(v):.6f}" for v in factors)
    meshes = "".join(
        f'    <mesh name="p_coll_{index}" file="trellis_office/coll/{name}_{index:03d}.obj" '
        f'scale="{scale}"/>\n'
        for index in range(report["hulls"])
    )
    geoms = "".join(
        GEOM_TEMPLATE.format(index=index, density=report["density"])
        for index in range(report["hulls"])
    )
    # An object that declares its inertial gets it from that element; MuJoCo
    # then ignores the hull densities for its mass. Every other object keeps
    # deriving mass from the hulls, and its MJCF text is unchanged.
    inertial = ""
    if "inertial" in report:
        spec = report["inertial"]
        inertial = (
            '        <inertial pos="{pos}" mass="{mass:g}" fullinertia="{full}"/>\n'.format(
                pos=" ".join(f"{float(v):.6f}" for v in spec["pos_m"]),
                mass=float(spec["mass_kg"]),
                full=" ".join(f"{float(v):.6e}" for v in spec["fullinertia_kg_m2"]),
            )
        )
    path = asset_root / f"{name}.xml"
    path.write_text(
        MJCF_TEMPLATE.format(
            name=name,
            scale=scale,
            inertial=inertial,
            collision_meshes=meshes,
            collision_geoms=geoms,
            top=report["top_site_m"][2],
            radius=report["horizontal_radius_site_m"][0],
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "robosuite_env" / "assets",
    )
    parser.add_argument("--object", action="append", choices=sorted(OBJECTS))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    selected = args.object or sorted(OBJECTS)
    report = [build(name, OBJECTS[name], args.asset_root) for name in selected]
    for entry in report:
        entry["mjcf"] = str(write_mjcf(entry, args.asset_root).name)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
