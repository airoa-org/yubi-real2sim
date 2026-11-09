# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Controller cadence, contact parameters, and YUBI setup."""

import json
import os
import sys

import mujoco
import numpy as np

from robosuite_env.rollout.build.gripper import setup_yubi_cbpc
from robosuite_env.rollout.contact_overrides import (
    legacy_finger_contact_override_ids,
)
from robosuite_env.rollout.control.fr3_strategy import make_fr3_controller_strategy
from robosuite_env.rollout.state.fingerprint import (
    compiled_model_signature as _compiled_model_signature,
)
from robosuite_env.rollout.state.render_state import build_render_state_contract
from robosuite_env.rollout.timing import exact_state_replay_playback_clock

def _contact_vec(raw, name, n):
    vals = [float(x) for x in str(raw).split()]
    if len(vals) != n:
        raise SystemExit(f"[sim] --{name} expects {n} floats, got {raw!r}")
    return vals

def configure_finger_contact(state):
    base_keep = str(state.A.solref).lower() == "keep"
    grip_friction_keep = str(state.A.grip_friction).lower() == "keep"
    rubber_friction_keep = str(state.A.rubber_friction).lower() == "keep"
    rubber_solref_keep = str(state.A.rubber_solref).lower() == "keep"
    rubber_solimp_keep = str(state.A.rubber_solimp).lower() == "keep"
    sr = None if base_keep else _contact_vec(state.A.solref, "solref", 2)
    gf = None if grip_friction_keep else _contact_vec(state.A.grip_friction, "grip-friction", 3)
    rgf = None if rubber_friction_keep else _contact_vec(state.A.rubber_friction, "rubber-friction", 3)
    rsr = None if rubber_solref_keep else _contact_vec(state.A.rubber_solref, "rubber-solref", 2)
    rsi = None if rubber_solimp_keep else _contact_vec(state.A.rubber_solimp, "rubber-solimp", 3)
    n=0; nr=0; nf=0; nrf=0
    for gid in legacy_finger_contact_override_ids(state.m, mujoco):
        name=mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_GEOM,gid) or ""
        if not base_keep:
            state.m.geom_solref[gid]=sr
            state.m.geom_priority[gid]=int(state.A.grip_priority)
            n+=1
        if not grip_friction_keep:
            state.m.geom_friction[gid,:len(gf)]=gf
            nf+=1
        if "pad_collision" in name:
            if not rubber_friction_keep:
                state.m.geom_friction[gid,:len(rgf)]=rgf
                nrf+=1
            if not rubber_solref_keep:
                state.m.geom_solref[gid]=rsr
            if not rubber_solimp_keep:
                state.m.geom_solimp[gid,:len(rsi)]=rsi
            state.m.geom_priority[gid]=int(state.A.rubber_priority)
            nr+=1
    if base_keep:
        print("[sim] finger contact: keeping XML solref/priority for generic finger hulls", flush=True)
    else:
        print(f"[sim] finger contact override: solref={sr} priority={state.A.grip_priority} geoms={n}", flush=True)
    if grip_friction_keep:
        print("[sim] finger friction: keeping XML friction for generic finger hulls", flush=True)
    else:
        print(f"[sim] finger friction override: friction={gf} geoms={nf}", flush=True)
    if nr:
        print(f"[sim] rubber pad contact override: solref={rsr if not rubber_solref_keep else 'keep'} "
              f"solimp={rsi if not rubber_solimp_keep else 'keep'} priority={state.A.rubber_priority} geoms={nr}", flush=True)
    else:
        print("[sim] rubber pad contact override: no pad geoms matched", flush=True)
    if rubber_friction_keep:
        print("[sim] rubber pad friction: keeping XML friction", flush=True)
    elif nrf:
        print(f"[sim] rubber pad friction override: friction={rgf} geoms={nrf}", flush=True)
    else:
        print("[sim] rubber pad friction override: no pad geoms matched", flush=True)

def configure_gripper_actuator_torque(state):
    mult=float(state.A.grip_torque_mult)
    if abs(mult-1.0) <= 1e-9:
        print("[sim] gripper actuator torque: keeping XML force ranges", flush=True)
        return
    rows=[]
    for key in ("R","L"):
        for aa in state.ARM[key]["fa"]:
            if aa < 0:
                continue
            name=mujoco.mj_id2name(state.m,mujoco.mjtObj.mjOBJ_ACTUATOR,int(aa)) or f"act{aa}"
            old=state.m.actuator_forcerange[int(aa)].copy()
            if hasattr(state.m, "actuator_forcelimited") and not bool(state.m.actuator_forcelimited[int(aa)]):
                continue
            state.m.actuator_forcerange[int(aa)] = old * mult
            rows.append((name, [round(float(x),4) for x in old],
                         [round(float(x),4) for x in state.m.actuator_forcerange[int(aa)]]))
    print(f"[sim] gripper actuator torque override: mult={mult:g} actuators={len(rows)} ranges={rows}", flush=True)

def _pin_arm_q(state,key,q7):
    state.d.qpos[state.ARM[key]["qi"]]=np.asarray(q7,float)
    state.d.qvel[state.ARM[key]["dof"]]=0.0
    state.d.qacc[state.ARM[key]["dof"]]=0.0

def run(state):
    """Configure FR3 arm control, YUBI gripper control, and actuator overrides."""
    state.FR3_STRATEGY=make_fr3_controller_strategy(state.A.fr3_controller)
    MODEL_SIGNATURE=_compiled_model_signature(state)
    state.MODEL_SIGNATURE = MODEL_SIGNATURE
    _TASK_OBJECT_RENDER_INVENTORY=[
        {
            key:binding[key]
            for key in (
                "name","role","joint_name","qpos_address","dof_address"
            )
        }
        for binding in state._TASK_OBJECT_BINDINGS.values()
    ]
    state._TASK_OBJECT_RENDER_INVENTORY = _TASK_OBJECT_RENDER_INVENTORY
    _RENDER_STATE_CONTRACT=build_render_state_contract(
        model_signature=state.MODEL_SIGNATURE,
        nq=int(state.m.nq),
        target=state._TASK_OBJECT_TARGET,
        inventory=_TASK_OBJECT_RENDER_INVENTORY,
    )
    state._RENDER_STATE_CONTRACT = _RENDER_STATE_CONTRACT
    _STATE_LOG_SOURCE_SUMMARY=None
    state._STATE_LOG_SOURCE_SUMMARY = _STATE_LOG_SOURCE_SUMMARY
    _STATE_LOG_SOURCE_SUMMARY_PATH=None
    state._STATE_LOG_SOURCE_SUMMARY_PATH = _STATE_LOG_SOURCE_SUMMARY_PATH
    if state.A.state_log_replay:
        _STATE_LOG_SOURCE_SUMMARY_PATH=os.path.join(
            os.path.dirname(os.path.abspath(state.A.state_log_replay)),
            "sim_summary.json",
        )
        state._STATE_LOG_SOURCE_SUMMARY_PATH = _STATE_LOG_SOURCE_SUMMARY_PATH
        try:
            with open(state._STATE_LOG_SOURCE_SUMMARY_PATH,"r") as _stream:
                _STATE_LOG_SOURCE_SUMMARY=json.load(_stream)
                state._STATE_LOG_SOURCE_SUMMARY = _STATE_LOG_SOURCE_SUMMARY
        except (OSError,ValueError):
            pass
    _CALIBRATION_PROVENANCE_KEYS=(
        "enabled",
        "method",
        "sha256",
        "runtime_reference_xml_sha256",
        "runtime_reference_xml_sha256_right",
        "definition",
        "left_offset_rad",
        "right_offset_rad",
    )
    state._CALIBRATION_PROVENANCE_KEYS = _CALIBRATION_PROVENANCE_KEYS
    _expected_controller_action_dim = state.PROFILE.controller_action_dim_for(
        state.A.fr3_controller if state.EMBODIMENT == "fr3" else None
    )
    state._expected_controller_action_dim = _expected_controller_action_dim
    if int(state.env.action_dim) != int(_expected_controller_action_dim):
        raise SystemExit(
            f"[sim] {state.ROBOT} controller action_dim={state.env.action_dim}; "
            f"expected {_expected_controller_action_dim} for its embodiment controller"
        )
    print(f"[sim] env built. robot={state.ROBOT} topology=dual-single-arm "
          f"desk={('scene' if state.A.desk=='scene' else 'GW:'+os.path.basename(state.A.desk))} "
          f"nq={state.m.nq} model_signature={state.MODEL_SIGNATURE[:12]}",flush=True)
    if state.A.state_log_replay and state.A.frame_hz>0:
        # Saved state rows are already the completed source-action samples.  Their
        # display clock is exact metadata, not a physics interval to quantize onto
        # MuJoCo substeps.  Keeping one row per encoded frame prevents a 30 Hz log
        # from silently becoming 29 fps (and therefore time-stretched).
        try:
            _state_replay_clock=exact_state_replay_playback_clock(state.A.frame_hz)
            state._state_replay_clock = _state_replay_clock
        except ValueError as exc:
            raise SystemExit(f"[sim] invalid state replay --frame-hz: {exc}") from exc
        FRAME_HZ_EFFECTIVE=_state_replay_clock.frame_frequency_hz
        state.FRAME_HZ_EFFECTIVE = FRAME_HZ_EFFECTIVE
        FRAME_DT_S=_state_replay_clock.frame_dt_s
        state.FRAME_DT_S = FRAME_DT_S
        OUTPUT_FPS=_state_replay_clock.output_fps
        state.OUTPUT_FPS = OUTPUT_FPS
        print(
            "[sim] exact state-log playback cadence: "
            f"{state.FRAME_HZ_EFFECTIVE:g}Hz ({state.FRAME_DT_S*1000.0:.3f}ms/saved row)",
            flush=True,
        )
    elif state._FR3_DYNAMIC_SOURCE_HZ is not None:
        # Playback uses the declared source cadence. Physics holds are necessarily
        # quantized to complete controller ticks when the two rates do not divide;
        # the phase schedule below preserves cumulative source time.
        FRAME_HZ_EFFECTIVE=float(state._FR3_DYNAMIC_SOURCE_HZ)
        state.FRAME_HZ_EFFECTIVE = FRAME_HZ_EFFECTIVE
        FRAME_DT_S=1.0/state.FRAME_HZ_EFFECTIVE
        state.FRAME_DT_S = FRAME_DT_S
        OUTPUT_FPS=int(round(state.FRAME_HZ_EFFECTIVE))
        state.OUTPUT_FPS = OUTPUT_FPS
        print(
            "[sim] FR3 policy/action and playback cadence: "
            f"{state.FRAME_HZ_EFFECTIVE:g}Hz ({state.FRAME_DT_S*1000.0:.3f}ms/action)",
            flush=True,
        )
    elif state.A.frame_hz>0:
        if any(arg=="--substeps" or arg.startswith("--substeps=") for arg in sys.argv[1:]):
            print("[sim] warning: --frame-hz set with --substeps; frame-hz takes precedence",flush=True)
        _ts=float(state.m.opt.timestep)
        state._ts = _ts
        state.A.substeps=max(1,int(round(1.0/(float(state.A.frame_hz)*_ts))))
        FRAME_DT_S=float(state.A.substeps)*_ts
        state.FRAME_DT_S = FRAME_DT_S
        FRAME_HZ_EFFECTIVE=1.0/state.FRAME_DT_S
        state.FRAME_HZ_EFFECTIVE = FRAME_HZ_EFFECTIVE
        OUTPUT_FPS=max(1,int(round(state.FRAME_HZ_EFFECTIVE)))
        state.OUTPUT_FPS = OUTPUT_FPS
        print(f"[sim] frame-hz {float(state.A.frame_hz):g}: substeps -> {int(state.A.substeps)} "
              f"(frame_dt={state.FRAME_DT_S*1000.0:.1f}ms, effective {state.FRAME_HZ_EFFECTIVE:.2f}Hz)",flush=True)
    DYNAMIC_SUBFRAMES=1
    state.DYNAMIC_SUBFRAMES = DYNAMIC_SUBFRAMES
    DYNAMIC_CONTROL_REPEATS=1
    state.DYNAMIC_CONTROL_REPEATS = DYNAMIC_CONTROL_REPEATS
    DYNAMIC_SUBFRAMES_REQUESTED=state._DYNAMIC_TIMING.requested_subframes
    state.DYNAMIC_SUBFRAMES_REQUESTED = DYNAMIC_SUBFRAMES_REQUESTED
    if state.A.physics=="dynamic":
        DYNAMIC_CONTROL_REPEATS=state._DYNAMIC_TIMING.control_repeats
        state.DYNAMIC_CONTROL_REPEATS = DYNAMIC_CONTROL_REPEATS
        DYNAMIC_SUBFRAMES=state._DYNAMIC_TIMING.subframes
        state.DYNAMIC_SUBFRAMES = DYNAMIC_SUBFRAMES
        if state.DYNAMIC_SUBFRAMES_REQUESTED!=state.DYNAMIC_SUBFRAMES:
            print(f"[sim] dynamic-subframes clipped {state.DYNAMIC_SUBFRAMES_REQUESTED} -> {state.DYNAMIC_SUBFRAMES} "
                  f"(--dynamic-control-repeats={state.DYNAMIC_CONTROL_REPEATS})",flush=True)
        if state.DYNAMIC_SUBFRAMES>1:
            OUTPUT_FPS=max(1,int(state.OUTPUT_FPS)*int(state.DYNAMIC_SUBFRAMES))
            state.OUTPUT_FPS = OUTPUT_FPS
            print(f"[sim] dynamic subframe sampling: {state.DYNAMIC_SUBFRAMES} render/export samples per policy frame; "
                  f"output_fps={state.OUTPUT_FPS}",flush=True)
        if state._FR3_DYNAMIC_SOURCE_HZ is not None:
            _ticks_ratio=float(state.A.dynamic_control_hz)/float(state._FR3_DYNAMIC_SOURCE_HZ)
            state._ticks_ratio = _ticks_ratio
            print(
                "[sim] FR3 controller phase scheduler: "
                f"source={state._FR3_DYNAMIC_SOURCE_HZ:g}Hz "
                f"ticks/action={int(np.floor(_ticks_ratio))}..{int(np.ceil(_ticks_ratio))} "
                f"at {float(state.A.dynamic_control_hz):g}Hz "
                "(cumulative-nearest, half-up)",
                flush=True,
            )
        else:
            print(f"[sim] dynamic control repeats={state.DYNAMIC_CONTROL_REPEATS}: "
                  f"sim_dt_per_source_action={state.DYNAMIC_CONTROL_REPEATS/float(state.env.control_freq):.4f}s",flush=True)

    configure_finger_contact(state)

    configure_gripper_actuator_torque(state)

    state.YUBI=None
    state._SIM_TIME=0.0
    _FINGER_DOFS=np.array([fd for key in ("R","L") for fd in state.ARM[key]["fd"]], dtype=int)
    state._FINGER_DOFS = _FINGER_DOFS
    setup_yubi_cbpc(state)
    GRIP_CTRL_EFFECTIVE = (
        ("cbpc_qfrc_dynamic" if state.A.physics=="dynamic" else "cbpc_qfrc_kinematic") if state.YUBI is not None else
        ("kinematic_qpos_pin" if state.A.physics=="kinematic" else "mujoco_position_actuator")
    )
    state.GRIP_CTRL_EFFECTIVE = GRIP_CTRL_EFFECTIVE
    print(f"[sim] gripper control effective={state.GRIP_CTRL_EFFECTIVE} "
          f"(requested --grip-ctrl={state.A.grip_ctrl}, physics={state.A.physics})", flush=True)

    state.DYNAMIC_YUBI_HOOK=None
    if (
        state.A.physics=="dynamic"
        and state.YUBI is not None
        and state.A.grip_ctrl=="cbpc"
    ):
        from robosuite_env.robots.yubi.control.robosuite_hook import DynamicYubiQfrcHook
        state.DYNAMIC_YUBI_HOOK=DynamicYubiQfrcHook(
            state.env._pre_action,state.d,state.ARM,state.YUBI,expected_dt=float(state.m.opt.timestep)
        )
        state.env._pre_action=state.DYNAMIC_YUBI_HOOK
        print(f"[sim] dynamic YUBI CbPC hook installed: one torque update per {state.m.opt.timestep:.4f}s physics step",flush=True)


__all__ = [
    "_pin_arm_q",
    "run",
]
