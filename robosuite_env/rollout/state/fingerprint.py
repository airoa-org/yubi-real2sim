# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Stable fingerprints for the compiled MuJoCo model."""

from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np


def compiled_model_signature(state):
    inventory = {
        "robot": state.ROBOT,
        "nq": int(state.m.nq),
        "nv": int(state.m.nv),
        "nu": int(state.m.nu),
        "nbody": int(state.m.nbody),
        "ngeom": int(state.m.ngeom),
        "joints": [
            mujoco.mj_id2name(state.m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            for i in range(state.m.njnt)
        ],
        "actuators": [
            mujoco.mj_id2name(state.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(state.m.nu)
        ],
        "joint_ranges": np.asarray(state.m.jnt_range, float).round(9).tolist(),
        "qpos0": np.asarray(state.m.qpos0, float).round(9).tolist(),
        "actuator_ctrlrange": np.asarray(
            state.m.actuator_ctrlrange, float
        ).round(9).tolist(),
        "actuator_forcerange": np.asarray(
            state.m.actuator_forcerange, float
        ).round(9).tolist(),
        "body_mass": np.asarray(state.m.body_mass, float).round(9).tolist(),
        "body_inertia": np.asarray(state.m.body_inertia, float).round(9).tolist(),
        "body_pos": np.asarray(state.m.body_pos, float).round(9).tolist(),
        "body_quat": np.asarray(state.m.body_quat, float).round(9).tolist(),
        "joint_pos": np.asarray(state.m.jnt_pos, float).round(9).tolist(),
        "joint_axis": np.asarray(state.m.jnt_axis, float).round(9).tolist(),
        "geom_type": np.asarray(state.m.geom_type, int).tolist(),
        "geom_size": np.asarray(state.m.geom_size, float).round(9).tolist(),
        "geom_pos": np.asarray(state.m.geom_pos, float).round(9).tolist(),
        "geom_quat": np.asarray(state.m.geom_quat, float).round(9).tolist(),
        "site_bodyid": np.asarray(state.m.site_bodyid, dtype=np.int64).tolist(),
        "site_pos": np.asarray(state.m.site_pos, float).round(9).tolist(),
        "site_quat": np.asarray(state.m.site_quat, float).round(9).tolist(),
        "camera_bodyid": np.asarray(state.m.cam_bodyid, dtype=np.int64).tolist(),
        "camera_pos": np.asarray(state.m.cam_pos, float).round(9).tolist(),
        "camera_quat": np.asarray(state.m.cam_quat, float).round(9).tolist(),
        "contact_exclude_signature": np.asarray(
            state.m.exclude_signature, dtype=np.int64
        ).tolist(),
    }
    raw = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def physical_model_fingerprint(state):
    """Hash compiled quantities that can change the settled physical state."""

    digest = hashlib.sha256()
    metadata = {
        "nq": int(state.m.nq),
        "nv": int(state.m.nv),
        "nu": int(state.m.nu),
        "na": int(state.m.na),
        "nbody": int(state.m.nbody),
        "njnt": int(state.m.njnt),
        "ngeom": int(state.m.ngeom),
        "nmesh": int(state.m.nmesh),
        "neq": int(state.m.neq),
        "timestep": float(state.m.opt.timestep),
        "gravity": [float(x) for x in state.m.opt.gravity],
        "integrator": int(state.m.opt.integrator),
        "solver": int(state.m.opt.solver),
        "iterations": int(state.m.opt.iterations),
        "ls_iterations": int(state.m.opt.ls_iterations),
        "tolerance": float(state.m.opt.tolerance),
        "cone": int(state.m.opt.cone),
        "jacobian": int(state.m.opt.jacobian),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    array_names = (
        "qpos0",
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_subtreemass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_gravcomp",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_bodyid",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_limited",
        "jnt_stiffness",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "dof_solref",
        "dof_solimp",
        "geom_bodyid",
        "geom_type",
        "geom_dataid",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
        "geom_condim",
        "geom_contype",
        "geom_conaffinity",
        "geom_priority",
        "geom_margin",
        "geom_gap",
        "mesh_vert",
        "mesh_face",
        "mesh_pos",
        "mesh_quat",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_dynprm",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_actrange",
        "actuator_actlimited",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_ctrllimited",
        "actuator_forcelimited",
        "pair_geom1",
        "pair_geom2",
        "pair_signature",
        "pair_dim",
        "pair_solref",
        "pair_solreffriction",
        "pair_solimp",
        "pair_margin",
        "pair_gap",
        "pair_friction",
        "exclude_signature",
        "tendon_type",
        "tendon_adr",
        "tendon_num",
        "tendon_matid",
        "tendon_width",
        "tendon_limited",
        "tendon_range",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_frictionloss",
        "tendon_springlength",
        "tendon_solref_lim",
        "tendon_solimp_lim",
        "tendon_solref_fri",
        "tendon_solimp_fri",
        "eq_type",
        "eq_obj1id",
        "eq_obj2id",
        "eq_active0",
        "eq_data",
        "eq_solref",
        "eq_solimp",
    )
    for name in array_names:
        if not hasattr(state.m, name):
            continue
        value = np.ascontiguousarray(np.asarray(getattr(state.m, name)))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


__all__ = ["compiled_model_signature", "physical_model_fingerprint"]
