# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Robosuite environment, MuJoCo model, and task-object construction."""

import json
import os

import mujoco
import numpy as np

from real2sim.config.task_objects import validate_task_objects
from robosuite_env.robots.fr3.initial_pose import (
    load_initial_arm_joint_pose,
    resolve_initial_grip_targets,
    validate_joint_ranges,
)
from robosuite_env.robots.fr3.runtime_ik import (
    AbsoluteFR3IK,
    effective_nullspace_gain,
)
from robosuite_env.robots.yubi.calibration import (
    GripperCalibrationError,
    apply_bilateral_gripper_calibration,
)
from robosuite_env.rollout.build.gs_server import setup_gs_server
from robosuite_env.rollout.setup.args import banner

def jid(state,n): return mujoco.mj_name2id(state.m,mujoco.mjtObj.mjOBJ_JOINT,n)

def bid(state,n): return mujoco.mj_name2id(state.m,mujoco.mjtObj.mjOBJ_BODY,n)

def aid(state,n): return mujoco.mj_name2id(state.m,mujoco.mjtObj.mjOBJ_ACTUATOR,n)

def _required_id(state,kind,name,lookup):
    value=lookup(state,name)
    if value < 0:
        raise SystemExit(f"[sim] {state.ROBOT} profile expected {kind} {name!r}, but it is absent from the compiled model")
    return value

def _free_task_object_binding(state,logical_name,obj,role):
    joint_names=tuple(getattr(obj,"joints",()) or ())
    if len(joint_names)!=1:
        raise RuntimeError(
            f"task object {logical_name!r} must expose exactly one joint; "
            f"got {list(joint_names)}"
        )
    joint_name=str(joint_names[0])
    joint_id=jid(state,joint_name)
    if joint_id<0:
        raise RuntimeError(
            f"task object {logical_name!r} joint is absent from the compiled "
            f"model: {joint_name!r}"
        )
    if int(state.m.jnt_type[joint_id])!=int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(
            f"task object {logical_name!r} joint {joint_name!r} is not free"
        )
    return {
        "name":str(logical_name),
        "role":str(role),
        "joint_name":joint_name,
        "joint_id":int(joint_id),
        "qpos_address":int(state.m.jnt_qposadr[joint_id]),
        "dof_address":int(state.m.jnt_dofadr[joint_id]),
    }

def _primary_object_bindings(state):
    """Resolve the task's target (``env.cube``) and receptacle (``env.plate``).

    Both come from the objects' own compiled joints, never from a fixed MJCF
    name. The box-pack env installs its target under its logical
    name (``tape``, ``pen``, ...), where a literal ``cube_joint0`` lookup
    returns -1 and numpy indexing then silently selects the model's last
    joint. Every failure to resolve raises. The receptacle is ``None`` for a
    task without one.
    """
    target=getattr(state.env,"cube",None)
    if target is None:
        raise RuntimeError("task env did not install a target object (env.cube)")
    plate=getattr(state.env,"plate",None)
    return (
        _free_task_object_binding(state,"cup",target,"target"),
        None if plate is None
        else _free_task_object_binding(state,"plate",plate,"receptacle"),
    )

def _task_binding(state):
    """Logical target and receptacle selected by the resolved task env."""

    objects=str(state.A.objects)
    if objects=="box_pack":
        target=str(getattr(state.env,"box_pack_target_object","") or "").strip()
        receptacle=str(getattr(getattr(state.env,"box",None),"name","") or "").strip()
    else:
        target="cup"
        receptacle=str(getattr(getattr(state.env,"plate",None),"name","") or "").strip()
    if not target or not receptacle:
        raise RuntimeError(
            f"task {objects!r} did not resolve a logical target/receptacle: "
            f"target={target!r} receptacle={receptacle!r}"
        )
    return {"target":target,"receptacle":receptacle}

def run(state):
    """Build the robosuite environment, MuJoCo model/data, and task objects."""
    try:
        validate_task_objects(state.A.objects, require_file=True)
    except ValueError as exc:
        raise SystemExit(f"[sim] {exc}") from exc
    banner(state)

    setup_gs_server(state)

    # ---------------- build env ----------------
    ROBOT = state.PROFILE.name
    state.ROBOT = ROBOT
    state.EMBODIMENT = state.PROFILE.embodiment
    from robosuite_env.robots.fr3 import model as fr3_robot
    state.fr3_robot = fr3_robot
    from robosuite_env.robots.yubi import model as _yubi_model  # noqa: F401
    if state.EMBODIMENT=="fr3":
        _joint_gain_overrides = {}
        if state.A.physics == "dynamic":
            if state.A.fr3_joint_kp is not None:
                _joint_gain_overrides["joint_kp"] = state.A.fr3_joint_kp
            if state.A.fr3_joint_kd is not None:
                _joint_gain_overrides["joint_kd"] = state.A.fr3_joint_kd
        ctrl = fr3_robot.load_fr3_controller_config(
            state.A.fr3_controller,
            **_joint_gain_overrides,
        )
        state.ctrl = ctrl
    else:
        ctrl = state.PROFILE.load_controller_config()
        state.ctrl = ctrl
    if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic":
        _joint_cfg = ctrl["body_parts"]["right"]
        state._joint_cfg = _joint_cfg
        _joint_ik_source_hz=(
            float(state.A.policy_action_hz)
            if float(state.A.policy_action_hz)>0.0 else float(state.A.frame_hz)
        )
        state._joint_ik_source_hz = _joint_ik_source_hz
        print(
            "[sim] FR3 ready joint servo (also used for joint_ik rollout): "
            f"{_joint_ik_source_hz:g}Hz whole-pose IK -> "
            f"{float(state.A.dynamic_control_hz):g}Hz JOINT_POSITION "
            f"joint_kp={float(state._joint_cfg['kp']):g} "
            f"vmax={float(state.A.joint_ik_max_velocity):g}rad/s "
            f"amax={float(state.A.joint_ik_max_acceleration):g}rad/s2 "
            f"jmax={float(state.A.joint_ik_max_jerk):g}rad/s3 "
            f"fail_hold={bool(state.A.joint_ik_fail_hold)} "
            f"desk_clearance={float(state.A.joint_ik_desk_collision_clearance_mm):g}mm",
            flush=True,
        )
    # objects positions
    if state.A.objects.endswith(".json") and os.path.isfile(state.A.objects):
        oj=json.load(open(state.A.objects)); os.environ["RS_CUP_XY"]=",".join(map(str,oj["cup_xy"])); os.environ["RS_PLATE_XY"]=",".join(map(str,oj["plate_xy"]))
        state.oj = oj
    if str(getattr(state.A, "box_pack_layout_json", "")).strip():
        _box_layout = str(state.A.box_pack_layout_json).strip()
        if not os.path.isfile(_box_layout):
            raise SystemExit(
                f"[sim] --box-pack-layout-json does not exist: {_box_layout}"
            )
        os.environ["RS_BOX_PACK_LAYOUT_JSON"] = _box_layout
        print(f"[sim] box_pack layout: {_box_layout}", flush=True)
    _BOX_PACK_TASK = (state.A.objects == "box_pack")
    state._BOX_PACK_TASK = _BOX_PACK_TASK
    if _BOX_PACK_TASK:
        from robosuite_env.tasks.box_pack import (
            box_pack_hand,
            box_pack_target_object,
            require_box_pack_instruction,
        )

        _index = int(state.A.box_pack_task_index)
        # This process knows its checkpoint only when it loads one itself. In a
        # batch run the policy server holds it and the worker is given an
        # address, so the strict question -- is this the wording that
        # checkpoint was trained on -- is the launcher's (preflight), and what
        # is checked here is that the prompt belongs to this task at all.
        require_box_pack_instruction(
            state.A.instruction, _index, checkpoint=getattr(state.A, "policy", None)
        )
        # The env and the criterion must agree on which object and which hand
        # the prompt named; both read these.
        os.environ["RS_BOX_PACK_OBJECT"] = box_pack_target_object(_index)
        os.environ["RS_BOX_PACK_HAND"] = box_pack_hand(_index)
        print(
            f"[sim] box pack task: task_index={_index} "
            f"object={box_pack_target_object(_index)!r} "
            f"hand={box_pack_hand(_index)!r} "
            f"instruction={state.A.instruction!r}",
            flush=True,
        )
    elif state.A.box_pack_task_index != 0:
        raise SystemExit(
            "[sim] --box-pack-task-index requires --objects box_pack"
        )
    # Hand-authored layouts: box resolves its own through
    # tasks/layout_selection.py -- the profile's flag, then the harness-only
    # RS_*_LAYOUT_JSON override, then the task default -- as the launcher
    # preflight, the init-state identity and the run summary do. The profile's
    # file reaches the environment under the override name.
    from robosuite_env.tasks.layout_selection import (
        LAYOUT_PARAMETERS,
        SELECTED_BY_PROFILE,
        LayoutError,
        select_task_layout,
    )

    for _layout_task, _layout_parameter in LAYOUT_PARAMETERS.items():
        if (
            str(getattr(state.A, _layout_parameter, "") or "").strip()
            and state.A.objects != _layout_task
        ):
            raise SystemExit(
                f"[sim] --{_layout_parameter.replace('_', '-')} requires "
                f"--objects {_layout_task}"
            )
    if state.A.box_pack_layout_index is not None and state.A.box_pack_layout_index < 0:
        raise SystemExit("[sim] --box-pack-layout-index must be non-negative")
    if state.A.box_pack_layout_index is not None and state.A.objects != "box_pack":
        raise SystemExit(
            "[sim] --box-pack-layout-index requires --objects box_pack"
        )
    if state.A.objects in LAYOUT_PARAMETERS:
        try:
            _layout_selection = select_task_layout(
                state.A.objects,
                getattr(state.A, LAYOUT_PARAMETERS[state.A.objects], ""),
                robot=state.ROBOT,
            )
        except LayoutError as exc:
            raise SystemExit(f"[sim] {exc}") from exc
        if not _layout_selection.path.is_file():
            raise SystemExit(
                f"[sim] {state.A.objects} layout does not exist: "
                f"{_layout_selection.path} (selected by "
                f"{_layout_selection.selected_by})"
            )
        if _layout_selection.selected_by == SELECTED_BY_PROFILE:
            os.environ[_layout_selection.override_env] = os.fspath(
                _layout_selection.path
            )
        print(
            f"[sim] {state.A.objects} layout: {_layout_selection.path} "
            f"(selected by {_layout_selection.selected_by})",
            flush=True,
        )
    if state.A.desk in {"robosuite", "native-table", "native_table"}:
        raise SystemExit(
            f"[sim] --desk {state.A.desk} was removed; the native TableArena "
            "environment is no longer supported. Pass a registered desk .obj "
            "or 'scene'."
        )
    if state.A.desk != "scene" and not os.path.isfile(state.A.desk):
        raise SystemExit(f"[sim] --desk does not exist: {state.A.desk}")
    if _BOX_PACK_TASK:
        from robosuite_env.tasks.envs.box_pack_registered import (
            BoxPackLiftGW as EnvCls,
        )
    else:
        from robosuite_env.tasks.envs.cup_plate_registered import (
            CupPlateLiftGW as EnvCls,
        )
    state.EnvCls = EnvCls
    if state.A.desk == "scene":
        if state.SCENE_CONTRACT_REPORT is None:
            raise SystemExit(
                "[sim] --desk scene requires --strict-assets and a validated "
                "schema-v2 --gs-background"
            )
        state.EnvCls.gw_desk_obj = None
        state.EnvCls.gw_desk_tex = None
        state.EnvCls.gw_table_collision_files = tuple(
            os.fspath(path)
            for path in state.SCENE_CONTRACT_REPORT["collision_files"]
        )
        # The launcher's scenes[].tag for this contract; hand-authored layouts
        # are checked against it (tasks/layout_selection.require_layout_scene).
        from robosuite_env.contracts.scene import scene_tag

        state.EnvCls.gw_scene_tag = scene_tag(state.SCENE_CONTRACT_REPORT["name"])
    else:
        state.EnvCls.gw_desk_obj=state.A.desk
        state.EnvCls.gw_desk_tex=state.A.desk_tex or (os.path.splitext(state.A.desk)[0]+".png")
        state.EnvCls.gw_table_collision_files=()
        state.EnvCls.gw_scene_tag=None
    state.EnvCls.table_z=state.A.table_z
    state.EnvCls.gw_desk_world_aligned=bool(state.A.desk_world_aligned)
    # Resolve the layout once, here, and hand the validated object to the env,
    # rather than through a module global in the task layer that would outlive
    # the build and be shared by every env in the process.
    from robosuite_env.robots.base_placement import resolve_robot_base_placement

    state.BASE_PLACEMENT = resolve_robot_base_placement(
        state.ROBOT, state.A.base_placement or None
    )
    state.EnvCls.base_placement = state.BASE_PLACEMENT
    print(
        f"[sim] {state.ROBOT} base placement: {state.BASE_PLACEMENT.source_path} "
        f"({state.BASE_PLACEMENT.source_sha256[:12]})",
        flush=True,
    )
    # NOTE: --world-offset is EE-target-only. Base/desk/cup/plate/GS stay FIXED (no scene shift here).
    _runtime_gripper_types=state.PROFILE.runtime_gripper_types(calibrated_yubi=True)
    state._runtime_gripper_types = _runtime_gripper_types
    state.env=state.EnvCls(robots=list(state.PROFILE.robosuite_robots),controller_configs=ctrl,gripper_types=_runtime_gripper_types,
               # Every output is produced by the explicit mujoco.Renderer
               # instances below. Keeping robosuite's implicit offscreen renderer
               # enabled creates an unused EGL context outside our GPU lock.
               has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,
               control_freq=int(state.A.dynamic_control_hz),
               ignore_done=(state.A.physics == "dynamic"),
               seed=(int(state.A.deterministic_seed) if int(state.A.deterministic_seed) >= 0 else None))
    state.env.reset(); state.m=state.env.sim.model._model; state.d=state.env.sim.data._data
    if getattr(state.A, "mujoco_timestep", None) is not None:
        timestep = float(state.A.mujoco_timestep)
        if not np.isfinite(timestep) or not 0 < timestep <= .002:
            raise ValueError("mujoco_timestep must be positive and at most 0.002 seconds")
        ticks = float(state.env.control_timestep) / timestep
        if not np.isclose(ticks, round(ticks), rtol=0, atol=1e-8):
            raise ValueError("mujoco_timestep must divide the environment control period")
        state.m.opt.timestep = timestep
        state.env.model_timestep = timestep  # noqa: V101  # robosuite counts physics steps using this clock.
    _INTEGRATOR_ENUMS = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    state._INTEGRATOR_ENUMS = _INTEGRATOR_ENUMS
    if state.A.mujoco_integrator != "keep":
        state.m.opt.integrator = _INTEGRATOR_ENUMS[state.A.mujoco_integrator]
        MUJOCO_INTEGRATOR_SELECTION = "explicit_diagnostic_override"
        state.MUJOCO_INTEGRATOR_SELECTION = MUJOCO_INTEGRATOR_SELECTION
    else:
        MUJOCO_INTEGRATOR_SELECTION = "model_default"
        state.MUJOCO_INTEGRATOR_SELECTION = MUJOCO_INTEGRATOR_SELECTION
    _INTEGRATOR_NAMES = {int(value): name for name, value in _INTEGRATOR_ENUMS.items()}
    state._INTEGRATOR_NAMES = _INTEGRATOR_NAMES
    MUJOCO_INTEGRATOR = _INTEGRATOR_NAMES.get(int(state.m.opt.integrator), str(int(state.m.opt.integrator)))
    state.MUJOCO_INTEGRATOR = MUJOCO_INTEGRATOR
    print(
        f"[sim] MuJoCo integrator={MUJOCO_INTEGRATOR} "
        f"selection={MUJOCO_INTEGRATOR_SELECTION} "
        f"timestep={float(state.m.opt.timestep):g}s",
        flush=True,
    )
    # Ambient-only headlight: view-independent wrist lighting, kept by the direct KB4 wrist images.
    state.m.vis.headlight.ambient[:]=[0.55,0.55,0.55]; state.m.vis.headlight.diffuse[:]=[0.0,0.0,0.0]; state.m.vis.headlight.specular[:]=[0.0,0.0,0.0]
    # Target and receptacle come from the objects the task env installed, not
    # from fixed MJCF names; see _primary_object_bindings.
    _target_binding,_plate_binding=_primary_object_bindings(state)
    state._TASK_BINDING=_task_binding(state)
    _cj=_target_binding["joint_id"]
    state._cj = _cj
    _cbid=state.m.jnt_bodyid[_cj]
    state._cbid = _cbid
    _CUP_MASS_XML=float(state.m.body_mass[_cbid]) if _cbid is not None and _cbid>=0 else None
    state._CUP_MASS_XML = _CUP_MASS_XML
    if state.A.cup_mass>0:   # Optional measured cup mass; inertia is scaled consistently.
        if _cbid is not None and _cbid>=0:
            _cur=float(state.m.body_mass[_cbid])
            state._cur = _cur
            if _cur>0:
                _sc=state.A.cup_mass/_cur
                state._sc = _sc
                _state_qpos=state.d.qpos.copy()
                state._state_qpos = _state_qpos
                _state_qvel=state.d.qvel.copy()
                state._state_qvel = _state_qvel
                _state_time=float(state.d.time)
                state._state_time = _state_time
                _state_ctrl=state.d.ctrl.copy()
                state._state_ctrl = _state_ctrl
                _state_act=state.d.act.copy() if state.m.na else None
                state._state_act = _state_act
                state.m.body_mass[_cbid]=state.A.cup_mass; state.m.body_inertia[_cbid]*=_sc
                mujoco.mj_setConst(state.m,state.d)   # refresh subtreemass etc.; this resets MjData
                state.d.qpos[:]=_state_qpos
                state.d.qvel[:]=_state_qvel
                state.d.time=_state_time
                state.d.ctrl[:]=_state_ctrl
                if state.m.na: state.d.act[:]=_state_act
                mujoco.mj_forward(state.m,state.d)
                print(f"[sim] cup mass override: {_cur*1000:.1f} g -> {state.A.cup_mass*1000:.1f} g (inertia x{_sc:.4f}) body={mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_BODY,_cbid)}",flush=True)
        else:
            print("[sim] --cup-mass set but no cup body found; skipped",flush=True)
    _CUP_MASS_EFFECTIVE=float(state.m.body_mass[_cbid]) if _cbid is not None and _cbid>=0 else None
    state._CUP_MASS_EFFECTIVE = _CUP_MASS_EFFECTIVE
    # Check every layout's recorded weight against the compiled body mass,
    # here, where the model exists. Reading body_mass is the reliable test:
    # base.xml sets inertiagrouprange="0 0", so compiling an asset XML on its
    # own reports a different number than the scene does.
    _LAYOUT_MASS_REPORTS={}
    state._LAYOUT_MASS_REPORTS = _LAYOUT_MASS_REPORTS
    # Resolve each task's layout the way the task itself does: the environment
    # variable when it is set, otherwise the module's default path, so runs
    # that do not override a layout are checked too.
    # Task-independent first: every measured object, checked against
    # whatever this model compiled. cup_plate places its objects from
    # RS_CUP_XY/RS_PLATE_XY and has no layout at all, so a layout-driven check
    # cannot reach it.
    from robosuite_env.tasks.layouts import assert_measured_masses
    # Name the renamed ones. robosuite compiles the task's selected object as
    # cube_main whatever it is, so the sweep cannot tell which asset it holds;
    # the object was built here from a known XML, so say so; this is how the
    # cup that cup_plate manipulates is checked.
    _RENAMED_BODIES={}
    for _logical,_obj in (("cube",getattr(state.env,"cube",None)),
                          ("plate",getattr(state.env,"plate",None))):
        if _obj is None:
            continue
        _file=getattr(_obj,"file",None)
        _asset=(os.path.splitext(os.path.basename(str(_file)))[0]
                if _file else None)
        if _asset:
            _RENAMED_BODIES[f"{getattr(_obj,'name',_logical)}_main"]=str(_asset)
    # --cup-mass is a supported override: it replaces the selected object's mass
    # on purpose, after the asset was compiled. Asserting the measured weight on
    # that same body would reject every run that used it, so the overridden
    # body is excluded and the override is stated instead of hidden.
    _RENAMED_BODIES_ALL=dict(_RENAMED_BODIES)
    _MASS_OVERRIDDEN=[]
    if float(getattr(state.A,"cup_mass",0.0) or 0.0)>0.0 and _cbid is not None \
            and _cbid>=0:
        _overridden_name=mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_BODY,_cbid)
        if _overridden_name:
            _MASS_OVERRIDDEN.append(_overridden_name)
            _RENAMED_BODIES.pop(_overridden_name,None)
            print(f"[sim] object masses: {_overridden_name} is excluded from the "
                  f"measured-mass check because --cup-mass overrode it to "
                  f"{float(state.A.cup_mass)*1000:.3f} g",flush=True)
    state._MASS_OVERRIDDEN = _MASS_OVERRIDDEN
    _MEASURED_MASS_REPORT=assert_measured_masses(state.m,mujoco=mujoco,
                                           extra_bodies=_RENAMED_BODIES,
                                           skip_bodies=_MASS_OVERRIDDEN)
    state._MEASURED_MASS_REPORT = _MEASURED_MASS_REPORT
    print(f"[sim] object masses: {len(_MEASURED_MASS_REPORT['checked'])} checked "
          f"against the measured masses",flush=True)
    for _row in _MEASURED_MASS_REPORT["checked"]:
        print(f"[sim]   {_row['body']}: {_row['measured_kg']*1000:.3f} g",flush=True)
    from robosuite_env.tasks.box_pack import BOX_PACK_LAYOUT_PATH
    for _layout_env,_layout_default,_layout_label in (
        ("RS_BOX_PACK_LAYOUT_JSON",BOX_PACK_LAYOUT_PATH,"box_pack"),
        # cup_plate's arrangement is --objects <json>, not an RS_* variable, so
        # it is listed here explicitly; its files carry top-level mass_kg.
        (None,(state.A.objects if str(state.A.objects).endswith(".json") else ""),
         "cup_plate"),
    ):
        _layout_path=((os.environ.get(_layout_env,"").strip() if _layout_env
                       else "") or (os.fspath(_layout_default)
                                    if _layout_default else ""))
        if not os.path.isfile(_layout_path):
            continue
        from robosuite_env.tasks.layouts import assert_layout_masses
        _payload=json.loads(open(_layout_path,encoding="utf-8").read())
        _report=assert_layout_masses(state.m,_payload,mujoco=mujoco,
                                     skip_bodies=_MASS_OVERRIDDEN,
                                     renamed_bodies=_RENAMED_BODIES_ALL,
                                     label=f"{_layout_label} {os.path.basename(_layout_path)}")
        _LAYOUT_MASS_REPORTS[_layout_label]={"layout":_layout_path,**_report}
        if _report.get("not_a_layout"):
            # This file was offered and is not a layout, so nothing here was
            # checked; say so rather than report it as clean.
            print(f"[sim] {_layout_label}: {os.path.basename(_layout_path)} carries "
                  f"no object weights ({_report['not_a_layout']}); the measured-mass "
                  f"table check above is what covers this task",flush=True)
            continue
        if _report.get("absent"):
            continue
        _checked=len(_report["checked"])
        _skipped=len(_report["unmeasured"])
        if not _report.get("measured"):
            # Present in this model, and not one recorded weight to check it
            # against. That is not a pass; it is an unprotected task, and it
            # says so rather than printing a reassuring count of zero.
            print(f"[sim] {_layout_label} layout masses: NOT PROTECTED -- "
                  f"{_skipped} objects present, none carries a measured weight "
                  f"({_layout_path})",flush=True)
        else:
            print(f"[sim] {_layout_label} layout masses: {_checked} checked "
                  f"against the layout, {_skipped} without a recorded weight",
                  flush=True)
        for _row in _report["checked"]:
            print(f"[sim]   {_row['object']}: {_row['layout_kg']*1000:.3f} g "
                  f"(compiled {_row['compiled_kg']*1000:.3f} g)",flush=True)
    ARM={}
    state.ARM = ARM
    for key,arm_profile in state.PROFILE.arms.items():
        _jids=np.array([_required_id(state,"joint",name,jid) for name in arm_profile.joint_names],dtype=int)
        state._jids = _jids
        _finger_jids=np.array([
            _required_id(state,"finger joint",name,jid)
            for name in arm_profile.finger_joint_names
        ],dtype=int)
        state._finger_jids = _finger_jids
        state.ARM[key]={
            "qi":np.array([state.m.jnt_qposadr[j] for j in _jids],dtype=int),
            "dof":np.array([state.m.jnt_dofadr[j] for j in _jids],dtype=int),
            "jr":np.array([state.m.jnt_range[j] for j in _jids],dtype=float),
            "eef":_required_id(state,"EEF body",arm_profile.eef_body,bid),
            "policy":_required_id(state,"policy hand-root body",arm_profile.policy_body,bid),
            "policy_offset":np.asarray(arm_profile.policy_offset_m,dtype=float),
            "link0":_required_id(state,"base body",arm_profile.base_body,bid),
            "fj":[state.m.jnt_qposadr[j] for j in _finger_jids],
            "fd":[state.m.jnt_dofadr[j] for j in _finger_jids],
            "finger_jids":_finger_jids,
            "fa":[
                _required_id(state,"finger actuator",name,aid)
                for name in arm_profile.finger_actuator_names
            ],
            "body_prefixes":arm_profile.body_prefixes,
            "gripper_prefix":arm_profile.gripper_prefix,
            "joint_names":arm_profile.joint_names,
        }
    _INITIAL_ARM_JOINT_POSE=None
    state._INITIAL_ARM_JOINT_POSE = _INITIAL_ARM_JOINT_POSE
    _INITIAL_ARM_JOINT_Q=None
    state._INITIAL_ARM_JOINT_Q = _INITIAL_ARM_JOINT_Q
    _INITIAL_GRIP_Q=None
    state._INITIAL_GRIP_Q = _INITIAL_GRIP_Q
    if state.EMBODIMENT=="fr3" and state.A.initial_arm_joints_json:
        try:
            _INITIAL_ARM_JOINT_POSE=load_initial_arm_joint_pose(
                state.A.initial_arm_joints_json
            )
            state._INITIAL_ARM_JOINT_POSE = _INITIAL_ARM_JOINT_POSE
            validate_joint_ranges(
                _INITIAL_ARM_JOINT_POSE.q_by_arm,
                {key:state.ARM[key]["jr"] for key in ("R","L")},
            )
            _INITIAL_GRIP_Q=resolve_initial_grip_targets(
                _INITIAL_ARM_JOINT_POSE,
                mimicgen_init_episode=state.A.mimicgen_init_episode,
            )
            state._INITIAL_GRIP_Q = _INITIAL_GRIP_Q
        except (OSError,ValueError) as exc:
            raise SystemExit(
                f"[sim] invalid --initial-arm-joints-json: {exc}"
            ) from exc
        _INITIAL_ARM_JOINT_Q={
            key:np.asarray(
                _INITIAL_ARM_JOINT_POSE.q_by_arm[key],dtype=np.float64
            ).copy()
            for key in ("R","L")
        }
        state._INITIAL_ARM_JOINT_Q = _INITIAL_ARM_JOINT_Q
        print(
            "[sim] specified initial arm joints: "
            f"source={_INITIAL_ARM_JOINT_POSE.source_path} "
            f"sha256={_INITIAL_ARM_JOINT_POSE.source_sha256} "
            f"schema_version={_INITIAL_ARM_JOINT_POSE.schema_version} "
            f"R={np.round(_INITIAL_ARM_JOINT_Q['R'],6).tolist()} "
            f"L={np.round(_INITIAL_ARM_JOINT_Q['L'],6).tolist()}",
            flush=True,
        )
        if _INITIAL_GRIP_Q is not None:
            print(
                "[sim] specified initial gripper (raw motor rad): "
                f"L={_INITIAL_GRIP_Q['L']:.6f} R={_INITIAL_GRIP_Q['R']:.6f}",
                flush=True,
            )
    _ABSOLUTE_JOINT_IK=None
    if state.EMBODIMENT=="fr3":
        _ABSOLUTE_JOINT_IK=AbsoluteFR3IK(
            state.m,
            {
                key:{
                    "qpos_indices":spec["qi"],
                    "dof_indices":spec["dof"],
                    "joint_ranges":spec["jr"],
                    "eef_body_id":spec["eef"],
                }
                for key,spec in state.ARM.items()
            },
            state.EE_OFF,
            iterations=100,
            damping=1e-5,
            max_step_deg=5.0,
            nullspace_gain=effective_nullspace_gain(state.A.fr3_nullspace, state.A.joint_ik_nullspace_gain),
            home_q=state.PROFILE.load_nullspace_home_qpos(),
        )
        state._ABSOLUTE_JOINT_IK = _ABSOLUTE_JOINT_IK
    try:
        _right_finger_jids=state.ARM["R"]["finger_jids"]
        state._right_finger_jids = _right_finger_jids
        _left_finger_jids=state.ARM["L"]["finger_jids"]
        state._left_finger_jids = _left_finger_jids
        GRIPPER_CALIBRATION_REPORT=apply_bilateral_gripper_calibration(
            state.m,
            right_joint_ids=_right_finger_jids,
            right_joint_names=[
                mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_JOINT,int(joint_id))
                for joint_id in _right_finger_jids
            ],
            left_joint_ids=_left_finger_jids,
            left_joint_names=[
                mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_JOINT,int(joint_id))
                for joint_id in _left_finger_jids
            ],
            calibration=state.GRIPPER_CALIBRATION,
            right_runtime_reference_xml=_yubi_model.YUBI_XML,
            left_runtime_reference_xml=_yubi_model.YUBI_XML_LEFT,
        )
        state.GRIPPER_CALIBRATION_REPORT = GRIPPER_CALIBRATION_REPORT
    except GripperCalibrationError as exc:
        raise SystemExit(f"[sim] could not apply bilateral gripper calibration: {exc}") from exc
    mujoco.mj_forward(state.m,state.d)
    print(
        "[sim] gripper calibration applied: "
        f"R={state.GRIPPER_CALIBRATION.right_offset_rad:+.12f}rad "
        f"L={state.GRIPPER_CALIBRATION.left_offset_rad:+.12f}rad "
        f"joints={len(state.GRIPPER_CALIBRATION_REPORT['joints'])} "
        f"sha256={state.GRIPPER_CALIBRATION.sha256[:12]}",
        flush=True,
    )
    # The manipulation target's free joint, resolved from env.cube's own
    # compiled joint by _primary_object_bindings (never a fixed MJCF name:
    # a literal "cube_joint0" lookup answers -1 on container/box-pack and numpy
    # then indexes the model's last joint).
    CUPa=state.m.jnt_qposadr[_cj]; cdof=state.m.jnt_dofadr[_cj]
    state.CUP_JOINT_NAME = _target_binding["joint_name"]
    state.CUPa = CUPa
    state.cdof = cdof
    _plate_jid=_plate_binding["joint_id"] if _plate_binding is not None else -1
    state._plate_jid = _plate_jid
    PLATEa=state.m.jnt_qposadr[_plate_jid] if _plate_jid>=0 else None
    state.PLATEa = PLATEa
    PLATEd=state.m.jnt_dofadr[_plate_jid] if _plate_jid>=0 else None
    state.PLATEd = PLATEd

    _TASK_OBJECT_BINDINGS={
        "cup":_free_task_object_binding(state,"cup",state.env.cube,"target"),
    }
    state._TASK_OBJECT_BINDINGS = _TASK_OBJECT_BINDINGS
    if state.env.plate is not None:
        _TASK_OBJECT_BINDINGS["plate"]=_free_task_object_binding(state,
            "plate",state.env.plate,"receptacle"
        )
    _TASK_OBJECT_TARGET="cup"
    state._TASK_OBJECT_TARGET = _TASK_OBJECT_TARGET
    # The ready pin, frame metrics and replay write CUPa / PLATEa, while
    # settle.py restores and gates _TASK_OBJECT_BINDINGS. If the two name
    # different joints, the pin moves an object that nothing puts back.
    _registry_plate=_TASK_OBJECT_BINDINGS.get("plate")
    if (
        _TASK_OBJECT_BINDINGS[_TASK_OBJECT_TARGET]["joint_id"]!=_cj
        or (-1 if _registry_plate is None else _registry_plate["joint_id"])!=_plate_jid
    ):
        raise RuntimeError(
            "task object registry disagrees with the resolved target/receptacle: "
            f"registry={_TASK_OBJECT_BINDINGS[_TASK_OBJECT_TARGET]['joint_name']!r}/"
            f"{None if _registry_plate is None else _registry_plate['joint_name']!r} "
            f"resolved={_target_binding['joint_name']!r}/"
            f"{None if _plate_binding is None else _plate_binding['joint_name']!r}"
        )
    _TASK_OBJECT_NAMES=tuple(_TASK_OBJECT_BINDINGS)
    state._TASK_OBJECT_NAMES = _TASK_OBJECT_NAMES
    _task_joint_ids=[state["joint_id"] for state in _TASK_OBJECT_BINDINGS.values()]
    state._task_joint_ids = _task_joint_ids
    if len(set(_task_joint_ids))!=len(_task_joint_ids):
        raise RuntimeError("task object registry contains duplicate compiled joints")
    OUTPUT_FPS=20
    state.OUTPUT_FPS = OUTPUT_FPS
    FRAME_DT_S=None
    state.FRAME_DT_S = FRAME_DT_S
    FRAME_HZ_EFFECTIVE=None
    state.FRAME_HZ_EFFECTIVE = FRAME_HZ_EFFECTIVE


__all__ = ["run"]
