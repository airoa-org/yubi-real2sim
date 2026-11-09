#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""CupPlateLift variant matching the REAL rig geometry:
  - FR3 bases placed at the real scene_yubitex_menagerie poses (robot0=R, robot1=L), ON the desk.
  - registered desk is collision-only (CoACD parts); its visual appearance is supplied by 3DGS.
    TableArena's fallback floor/table/room geometry is removed.
The rollout worker sets the class attributes before reset."""
import os, numpy as np
import xml.etree.ElementTree as ET
import trimesh

from .cup_plate import CupPlateLift
from ..equalities import add_rule_grasp_weld_equalities
from ..layouts import LayoutError, convex_hull_halfspaces, seat_lift
from ..scene import place_robots_at_calibrated_bases


def remove_table_arena_fallback_geometries(root):
    """Remove geometry whose appearance or physics belongs to the 3DGS scene.

    Alpha-zero is insufficient here: MuJoCo's stock ``floor`` geom still has
    collision enabled and can silently become an infinite support plane under
    the registered table.  Removing the stock floor / table collision and all
    arena-only visuals leaves the validated CoACD geoms as the sole table
    physics, while preserving task geometry such as ``container_wall_*``.
    """

    removed = set()
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            name = child.get("name", "")
            is_arena_fallback = name in {
                "floor",
                "table_collision",
                "table_visual",
            }
            is_arena_visual = (
                (name.startswith("table_leg") or name.startswith("wall_"))
                and name.endswith("_visual")
            )
            if is_arena_fallback or is_arena_visual:
                parent.remove(child)
                removed.add(name)
    if "floor" not in removed or "table_collision" not in removed:
        raise ValueError(
            "TableArena fallback floor/table collision was not found; refusing "
            "to run with ambiguous table physics"
        )
    return removed


#: Name prefix of the compiled table collision parts. The settle stage's
#: support registry keys on the same prefix.
TABLE_COLLISION_GEOM_PREFIX = "gw_desk_coll_"


def _body_in_subtree(model, body_id, root_id):
    body_id = int(body_id)
    while body_id > 0:
        if body_id == int(root_id):
            return True
        body_id = int(model.body_parentid[body_id])
    return body_id == int(root_id)


def _geom_world_vertices(mujoco, model, data, geom_id):
    """World-frame vertices of one collision geom's convex support set."""

    geom_type = int(model.geom_type[geom_id])
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        local = np.asarray(
            model.mesh_vert[start : start + int(model.mesh_vertnum[mesh_id])],
            dtype=float,
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        sx, sy, sz = (float(value) for value in model.geom_size[geom_id])
        local = np.array(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
        )
    else:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        raise LayoutError(
            f"cannot seat against collision geom {name or geom_id!r}: only mesh "
            "and box geoms have an exact hull here"
        )
    return local @ rotation.T + position


def _collision_hulls(mujoco, model, data, geom_ids):
    return [
        convex_hull_halfspaces(_geom_world_vertices(mujoco, model, data, geom_id))
        for geom_id in geom_ids
    ]


def seat_objects_on_table(mujoco, model, data, objects, *, table_geom_ids):
    """Seat free-joint objects on table collision, preserving XY and orientation.

    ``objects`` maps each name to ``(qpos_address, collision_geom_ids)``.
    Only free-joint z changes; the report follows the input object order.
    """
    if not table_geom_ids:
        raise LayoutError(
            "no table collision geoms to seat layout objects on; expected "
            f"geoms named {TABLE_COLLISION_GEOM_PREFIX}*"
        )
    for name, (_, geom_ids) in objects.items():
        if not geom_ids:
            raise LayoutError(f"layout object {name!r} has no collision geoms to seat")
    mujoco.mj_kinematics(model, data)
    table_hulls = _collision_hulls(mujoco, model, data, table_geom_ids)
    report = []
    for name, (qpos_address, geom_ids) in objects.items():
        object_hulls = _collision_hulls(mujoco, model, data, geom_ids)
        lift = seat_lift(object_hulls, table_hulls)
        if lift is None:
            raise LayoutError(
                f"layout object {name!r} is not over the table collision; "
                "its authored XY places it off the table"
            )
        before = float(data.qpos[qpos_address + 2])
        data.qpos[qpos_address + 2] = before + lift
        mujoco.mj_kinematics(model, data)
        report.append({
            "object": name,
            "support": "table",
            "z_before_m": before,
            "z_seated_m": before + lift,
            "lift_m": lift,
        })
    return report


class CupPlateLiftGW(CupPlateLift):
    gw_desk_obj = None
    gw_desk_tex = None
    # Official runs populate this from the validated 3DGS scene contract.
    # Keeping the exact file list here avoids a second table-path option and
    # prevents a missing / empty CoACD directory from silently falling back to
    # the flat TableArena collision plane.
    gw_table_collision_files = ()
    # Tag of that validated scene contract (scene_tag of its name), set with
    # the collision files. A hand-authored container / box layout records the
    # scene it was placed for and is refused on any other, or when this is
    # unset (see tasks/layout_selection.py require_layout_scene).
    gw_scene_tag = None
    gw_desk_width_m = 1.50        # scale desk so its X-extent == this (meters)
    gw_desk_top_pct = 88.0
    gw_desk_world_aligned = False  # --desk vertices already live in MuJoCo world meters
    # Validated RobotBasePlacement for whichever rig this env builds. There
    # is no default; a calibration is required, so a registered scene never
    # falls back to robosuite's nominal base offsets.
    base_placement = None
    table_z = 0.8                 # world z of the table TOP surface (set 0.0 to put the surface at z=0)
    _desk_scale = 1.0

    def _configure_task_objects(self):
        """Build task objects and their placement sampler.

        Kept as a hook so tasks that share the calibrated FR3 / reconstructed
        desk geometry can replace cup+plate without duplicating that geometry.
        """
        from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
        from ..objects import CupObject, PlateObject
        import os as _os

        self.cube = CupObject("cube")
        self.plate = PlateObject("plate")
        self.extra_task_objects = []
        _cx, _cy = [float(v) for v in _os.environ.get("RS_CUP_XY", "0.0,-0.10").split(",")]
        _px, _py = [float(v) for v in _os.environ.get("RS_PLATE_XY", "0.0,0.10").split(",")]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="PlateSampler", mujoco_objects=self.plate, x_range=[_px, _px], y_range=[_py, _py],
            rotation=0.0, ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.001, rng=self.rng))
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="CupSampler", mujoco_objects=self.cube, x_range=[_cx, _cx], y_range=[_cy, _cy],
            rotation=0.0, ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.02, rng=self.rng))
        # The samplers' z offsets assume a table top at table_z, but the
        # registered collision top is wherever the scene put it (+34 mm on
        # Duo), so both objects are seated on it after sampling. XY still
        # comes from the samplers, which keeps their RNG draws unchanged.
        self.table_rest_objects = (self.plate, self.cube)

    def _weld_placeholder_bodies(self):
        """Name the two bodies the inactive rule-grasp welds are declared on.

        MuJoCo fixes the equality count at compile time, so the pair only has
        to exist and be distinct -- it is never welded as written. A task
        without a plate overrides this with two bodies it does have, which also
        keeps its compiled model fingerprinting the same way every run.
        """

        return (self.cube.root_body, self.plate.root_body)

    #: Objects placed by a sampler rather than a layout that must still start
    #: resting on the table. Cup/plate sets its pair; the other tasks leave
    #: this empty.
    table_rest_objects = ()

    def _reset_internal(self):
        super()._reset_internal()
        # Lift samples placements only when the model was not restored from
        # an XML state; a deterministic reset carries exact poses already.
        if not self.deterministic_reset:
            self._seat_layout_placements()

    def _seat_layout_placements(self):
        """Rest every exactly-placed layout object on what supports it.

        An :class:`ExactObjectPoseSampler` pose is an authored rest pose whose
        z was written against a flat plane at table z = 0, and the cup/plate
        sampler's z offsets assume the same plane. Here that z is replaced by
        contact with the compiled table collision, so objects start on
        the table whatever height it has. Other sampled placements are left
        where their sampler put them.
        """

        import mujoco
        from robosuite.utils.placement_samplers import SequentialCompositeSampler

        from .placement import ExactObjectPoseSampler

        pending = [self.placement_initializer]
        exact = []
        while pending:
            sampler = pending.pop(0)
            if isinstance(sampler, ExactObjectPoseSampler):
                exact.append(sampler)
            elif isinstance(sampler, SequentialCompositeSampler):
                pending.extend(sampler.samplers.values())
        groups = [sampler.mujoco_objects for sampler in exact]
        groups.append([obj for obj in self.table_rest_objects if obj is not None])
        self.layout_seating_report = []
        if not any(groups):
            return
        model = self.sim.model._model
        data = self.sim.data._data
        objects = {}
        for group in groups:
            for obj in group:
                if obj.name in objects:
                    continue
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj.joints[0])
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
                if joint_id < 0 or body_id < 0:
                    raise LayoutError(
                        f"layout object {obj.name!r} is not compiled: joint "
                        f"{obj.joints[0]!r} / body {obj.root_body!r}"
                    )
                if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    raise LayoutError(f"layout object {obj.name!r} has no free joint to seat")
                geom_ids = tuple(
                    geom_id
                    for geom_id in range(model.ngeom)
                    if (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
                    and _body_in_subtree(model, model.geom_bodyid[geom_id], body_id)
                )
                objects[obj.name] = (int(model.jnt_qposadr[joint_id]), geom_ids)
        table_geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            ).startswith(TABLE_COLLISION_GEOM_PREFIX)
        )
        self.layout_seating_report = seat_objects_on_table(
            mujoco,
            model,
            data,
            objects,
            table_geom_ids=table_geom_ids,
        )
        self.sim.forward()
        for row in self.layout_seating_report:
            print(
                f"[layout] seated {row['object']} on {row['support']}: z "
                f"{row['z_before_m']:+.5f} -> {row['z_seated_m']:+.5f} m "
                f"({row['lift_m'] * 1000.0:+.2f} mm)",
                flush=True,
            )

    def _load_model(self):
        # replicate CupPlateLift._load_model but set REAL FR3 bases BEFORE building the task
        if not self.gw_desk_obj and not self.gw_table_collision_files:
            raise ValueError(
                "CupPlateLiftGW requires a registered desk OBJ (gw_desk_obj) "
                "or scene-contract collision files; the native TableArena "
                "environment was removed"
            )
        from robosuite.environments.manipulation.lift import Lift
        from robosuite.models.arenas import TableArena
        from robosuite.models.tasks import ManipulationTask
        super(Lift, self)._load_model()
        self.table_offset = np.array([0.0, 0.0, float(self.table_z)])   # put the table TOP surface at z=table_z
        ttop = float(self.table_offset[2]); tx, ty = float(self.table_offset[0]), float(self.table_offset[1])
        if self.base_placement is None:
            raise ValueError(
                f"{type(self).__name__} requires a base_placement: a registered "
                "scene places its rig from a calibration, and falling back to "
                "the nominal offsets would mis-place it silently"
            )
        place_robots_at_calibrated_bases(
            self.robots,
            table_top=np.array([tx, ty, ttop]),
            placement=self.base_placement,
        )
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self._configure_task_objects()
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=[obj for obj in (self.cube, self.plate, *self.extra_task_objects) if obj is not None])
        root = self.model.root
        add_rule_grasp_weld_equalities(
            root,
            placeholder_bodies=self._weld_placeholder_bodies(),
        )
        # ---- desk mesh: keep only the collision representation.  The desk
        # visual is intentionally supplied by 3DGS, so it must not be added
        # to MuJoCo even when a registered desk OBJ is provided.
        if self.gw_desk_obj or self.gw_table_collision_files:
            contract_collision_files = tuple(
                os.path.abspath(os.fspath(path))
                for path in self.gw_table_collision_files
            )
            if contract_collision_files:
                missing = [path for path in contract_collision_files if not os.path.isfile(path)]
                if missing:
                    raise FileNotFoundError(
                        "scene-contract table collision mesh is missing: "
                        + ", ".join(missing)
                    )
                if len(set(contract_collision_files)) != len(contract_collision_files):
                    raise ValueError("scene-contract table collision list contains duplicates")
                self._desk_scale = 1.0
                tx = ty = zoff = 0.0
                xext = None
            elif self.gw_desk_world_aligned:
                self._desk_scale = 1.0
                tx = ty = zoff = 0.0
                xext = None
            else:
                v = np.asarray(trimesh.load(self.gw_desk_obj, process=False, force="mesh").vertices)
                xext = float(v[:, 0].max() - v[:, 0].min())
                self._desk_scale = self.gw_desk_width_m / xext if xext > 1e-6 else 1.0
                topz = float(np.percentile(v[:, 2], self.gw_desk_top_pct)) * self._desk_scale
                zoff = ttop - topz
            asset = root.find("asset")
            # The registered desk visual is intentionally not compiled into
            # MuJoCo. The 3DGS background supplies the appearance; only the
            # low-poly CoACD parts below provide physical contact.
            # desk COLLISION (CoACD) so franka/desk penetration can be MEASURED (contype=1)
            if contract_collision_files:
                collision_files = contract_collision_files
            else:
                cdir = os.path.join(os.path.dirname(self.gw_desk_obj), "coacd")
                if not os.path.isdir(cdir):
                    raise FileNotFoundError(
                        f"registered table collision directory is missing: {cdir}"
                    )
                collision_files = tuple(
                    os.path.abspath(os.path.join(cdir, filename))
                    for filename in sorted(os.listdir(cdir))
                    if filename.endswith(".obj")
                )
            if not collision_files:
                raise ValueError("registered table collision set must contain at least one OBJ")
            for j, collision_file in enumerate(collision_files):
                ET.SubElement(asset, "mesh", name=f"gw_desk_coll_{j}", file=collision_file,
                              scale=f"{self._desk_scale} {self._desk_scale} {self._desk_scale}")
                ET.SubElement(root.find("worldbody"), "geom", name=f"gw_desk_coll_{j}", type="mesh",
                              mesh=f"gw_desk_coll_{j}", pos=f"{tx} {ty} {zoff}", group="3",
                              contype="1", conaffinity="1", rgba="0.5 0.5 0.5 0")
            # Remove, rather than merely hide, TableArena's collision-enabled
            # infinite floor and flat table box only after an exact non-empty
            # CoACD set has been installed. Arena visuals are removed too: the
            # table / room appearance is already part of the 3DGS scene.
            remove_table_arena_fallback_geometries(root)
            print(f"[gwdesk] table collision: {len(collision_files)} validated CoACD parts "
                  "(contype=1) -> franka/table penetration measurable", flush=True)
            if contract_collision_files:
                print("[gwdesk] table visual omitted; collision comes from the 3DGS scene contract "
                      "(world-aligned, scale=1, pos=0)", flush=True)
            elif self.gw_desk_world_aligned:
                print(f"[gwdesk] desk visual omitted; collision-only world-aligned "
                      f"(asset={os.path.basename(self.gw_desk_obj)}, scale=1, pos=0)", flush=True)
            else:
                print(f"[gwdesk] desk visual omitted; collision-only {os.path.basename(self.gw_desk_obj)} "
                      f"xext={xext:.3f}m -> scale={self._desk_scale:.4f} "
                      f"(width {self.gw_desk_width_m}m) zoff={zoff:.3f} ttop={ttop:.3f}", flush=True)
