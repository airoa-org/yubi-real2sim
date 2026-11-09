# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Finalize outputs, summaries, and resource cleanup."""

import atexit
import math
import os
import sys
import time

import numpy as np

from robosuite_env.policy import policy_io as pio
from robosuite_env.rendering.wrist_render_profiles import wrist_image_intrinsics
from robosuite_env.rollout.control.rule_grasp import _rule_grasp_status
from robosuite_env.rollout.outputs import (
    resolve_summary_paths,
    write_policy_chunk_records,
    write_rgb_video,
    write_summary_artifacts,
)
from robosuite_env.rollout.preflight import reported_gs_w2gs
from robosuite_env.rollout.state.render_state import build_render_state_summary_contract
from robosuite_env.rollout.task_layout import loaded_task_layout
from robosuite_env.rollout.timing import constant_control_ticks_per_source_action
from robosuite_env.robots.fr3.joint_gains import fr3_joint_gain_summary
from robosuite_env.robots.fr3.runtime_ik import controller_path_summary, controller_summary_name, effective_nullspace_gain
from robosuite_env.tasks.success import summarize_success_records


from robosuite_env.rollout.observe.policy_client import (
    _policy_display_transform_record,
    _raw_policy_display_transform_record,
)
from robosuite_env.rollout.render.composite import (
    iter_output_video_frames as _iter_output_video_frames,
)


def policy_wrist_intrinsics(state):
    """KB4 model of each rendered policy wrist image before the policy resize."""

    if not state.POLICY_SOURCE or state.A.policy_image_source != "sim":
        return None
    return {
        side: wrist_image_intrinsics(state.A.policy_yubi_render_profile, side).as_dict()
        for side in ("left", "right")
    }


# ---------------------------------------------------------------- arm controller


def arm_controller_fields(state):
    """Name and describe the arm controller that actually drove this run.

    Replays that bypass the online controller keep their replay name on every
    rig. FR3 and FR3 Duo use the FR3 formatter.
    """

    embodiment=state.EMBODIMENT
    physics=state.A.physics
    detail={"rig":state.ROBOT,"embodiment":embodiment,"physics":physics}
    if state.A.state_log_replay:
        detail["path"]="saved_state_log_qpos_rows_without_an_online_arm_controller"
        return {"arm_controller":"generation_state_log_qpos_replay","arm_controller_detail":detail}
    if embodiment=="fr3":
        detail["path"]=controller_path_summary(physics,state.A.fr3_controller)
        detail["fr3_controller"]=state.A.fr3_controller
        impedance = getattr(state, "FR3_IMPEDANCE", None)
        if impedance is not None:
            detail["cartesian_impedance"] = impedance.summary()

        return {
            "arm_controller":controller_summary_name(physics,state.A.fr3_controller,state.A.fr3_nullspace),
            "arm_controller_detail":detail,
        }
    raise ValueError(f"no arm-controller summary for embodiment {embodiment!r}")


# ---------------------------------------------------------------- object lift

_LEGACY_LIFT_CONSUMERS=(
    "robosuite_env/tools/evaluate_rollout_tree.py",
    "robosuite_env/tools/summarize_policy_rollouts.py",
    "robosuite_env/tools/plot_repeat_background_summary.py",
)


def _target_joint_name(state):
    """The joint whose qpos ``cup_pos`` was sampled from (``state.CUPa``)."""

    address=getattr(state,"CUPa",None)
    if address is None:
        return None
    for binding in (getattr(state,"_TASK_OBJECT_BINDINGS",None) or {}).values():
        if isinstance(binding,dict) and binding.get("qpos_address")==int(address):
            return binding.get("joint_name")
    model=getattr(state,"m",None)
    if model is not None:
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_qposadr[joint_id])==int(address):
                return model.joint(joint_id).name
    return None


def task_binding_fields(state):
    """Summary names from the task binding resolved when the model was built."""

    binding=getattr(state,"_TASK_BINDING",None)
    if not isinstance(binding,dict):
        raise RuntimeError(f"rollout has no resolved task binding: {binding!r}")
    values={role:binding.get(role) for role in ("target","receptacle")}
    if any(not isinstance(value,str) or not value for value in values.values()):
        raise RuntimeError(f"rollout has an invalid resolved task binding: {binding!r}")
    return {
        "task_target":values["target"],
        "task_receptacle":values["receptacle"],
    }


def legacy_target_lift_fields(state,cup_pos):
    """Keep ``cup_lift_mm`` / ``cup_displacement_mm`` and say what they measure.

    Both follow one fixed joint for every task and are relative to that
    joint's own pre-loop position, so they count a startup launch as a lift.
    They stay for the consumers named in the definition; ``task_lift_mm`` /
    ``task_lift`` are the task-aware, table-relative replacement.
    """

    has_rows=len(cup_pos)>0
    return {
        "cup_lift_mm":round(float((cup_pos[:,2].max()-state.cup[2])*1000.0),3) if has_rows else None,
        "cup_displacement_mm":round(float(np.linalg.norm(cup_pos-state.cup[None,:],axis=1).max()*1000.0),3) if has_rows else None,
        "cup_lift_mm_definition":{
            "fields":["cup_lift_mm","cup_displacement_mm"],
            "joint":_target_joint_name(state),
            "measure":(
                "cup_lift_mm: highest logged z of this free joint minus its reference z; "
                "cup_displacement_mm: largest 3-D distance from its reference position"
            ),
            "datum":(
                "the joint's own position captured before the frame loop (state.cup: "
                "after settle/ready, or the replay source's first target pose); "
                "not the table surface"
            ),
            "table_relative":False,
            "follows_task_stage":False,
            "counts_uncommanded_motion":True,
            "kept_for":list(_LEGACY_LIFT_CONSUMERS),
            "replacement":["task_lift_mm","task_lift"],
        },
    }


def _finite_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        return None
    value=float(value)
    return value if math.isfinite(value) else None


def _height_track(samples,*,with_contact):
    """Start/end/peak of ``(frame, height_mm, robot_touching)`` samples."""

    finite=[
        (frame,height,touching)
        for frame,raw,touching in samples
        for height in (_finite_number(raw),)
        if height is not None
    ]
    track={"evaluated_frames":len(samples),"finite_frames":len(finite)}
    if not finite:
        return track
    peak=max(finite,key=lambda sample:sample[1])
    track.update({
        "start_frame":finite[0][0],"start_mm":round(finite[0][1],3),
        "end_frame":finite[-1][0],"end_mm":round(finite[-1][1],3),
        "max_frame":peak[0],"max_mm":round(peak[1],3),
        "min_mm":round(min(sample[1] for sample in finite),3),
    })
    if with_contact:
        touched=[sample for sample in finite if sample[2]]
        touched_peak=max(touched,key=lambda sample:sample[1]) if touched else None
        track.update({
            "robot_contact_frames":len(touched),
            "max_while_robot_contact_frame":touched_peak[0] if touched_peak else None,
            "max_while_robot_contact_mm":round(touched_peak[1],3) if touched_peak else None,
        })
    return track


def _success_evaluations(records):
    return [
        (int(record.get("frame",index)),record["success"])
        for index,record in enumerate(records)
        if isinstance(record.get("success"),dict)
    ]


def _metric_task_lift(name,evaluations,description,*,metric,label,quantity,definition,extra=None):
    rows=[(frame,evaluation.get("metrics") or {}) for frame,evaluation in evaluations]
    table_relative=all("table_top_z_mm" in metrics for _,metrics in rows)
    datum=description.get("table_datum") if isinstance(description,dict) else None
    object_name=label(rows[-1][1])
    track=_height_track([(frame,metrics.get(metric),None) for frame,metrics in rows],with_contact=False)
    block={
        "schema_version":1,
        "criterion":name,
        "available":True,
        "quantity":quantity,
        "source":f"success.metrics.{metric}",
        "table_datum_source":datum.get("source") if isinstance(datum,dict) else None,
        "table_relative":table_relative,
        "task_lift_mm_definition":definition,
        "objects":{object_name:track},
    }
    if extra is not None:
        block.update(extra(rows))
    return track,block


def task_lift_fields(records,criterion_description=None):
    """Table-relative lift of the object(s) the task is actually about.

    The heights come from the success criterion's own per-frame metrics, so
    they share its table datum and its object identities; nothing here reads a
    fixed joint. A log whose criterion predates the table datum is reported
    with ``table_relative: false`` and a null ``task_lift_mm``.
    """

    evaluations=_success_evaluations(records)
    description=criterion_description if isinstance(criterion_description,dict) else None
    first=evaluations[0][1] if evaluations else {}
    name=(description or {}).get("name") or first.get("name") or first.get("task")
    if not evaluations:
        return {"task_lift_mm":None,"task_lift":{
            "schema_version":1,"criterion":name,"available":False,
            "reason":"no success evaluations were logged",
        }}
    if name=="object_in_box":
        track,block=_metric_task_lift(
            name,evaluations,description,
            metric="object_height_mm",
            label=lambda metrics:str(metrics.get("object")),
            quantity=(
                "named object's body origin height above the table datum under the "
                "box (criterion metric object_height_mm)"
            ),
            definition=(
                "highest logged object_height_mm of the prompt's object; null if the "
                "log has no table-relative heights"
            ),
            extra=lambda rows:{
                "box":_height_track(
                    [(frame,metrics.get("box_height_mm"),None) for frame,metrics in rows],
                    with_contact=False,
                ),
            },
        )
        return {
            "task_lift_mm":track.get("max_mm") if block["table_relative"] else None,
            "task_lift":block,
        }
    return {"task_lift_mm":None,"task_lift":{
        "schema_version":1,"criterion":name,"available":False,
        "reason":"this criterion reports no table-relative height for the object it scores",
    }}


def run(state):
    """Write videos, JSONL logs, and the final summary; release resources."""

    # ---------------- outputs ----------------
    _POLICY_CHUNKS_PATH=""
    if state.POLICY and state._POLICY_CHUNK_RECORDS:
        _POLICY_CHUNKS_PATH=write_policy_chunk_records(
            state.LOGDIR,state._POLICY_CHUNK_RECORDS
        )
        print(f"[sim] policy chunks -> {_POLICY_CHUNKS_PATH} ({len(state._POLICY_CHUNK_RECORDS)} chunks)",flush=True)
    if state.logf:
        state.logf.close()
        state.logf=None
        state.logf=state.logf
        jump_frames=[{"frame":r["frame"],"arm":s,"joint":r["arms"][s]["jump_joint"],"dqmax_deg":r["arms"][s]["dqmax_deg"]}
                     for r in state._RECS for s in ("R","L") if r["arms"][s].get("jump")]
        raw_jump_frames=[{"frame":r["frame"],"arm":s,"joint":r["arms"][s].get("raw_jump_joint",r["arms"][s]["jump_joint"]),
                          "dqmax_deg":r["arms"][s].get("raw_dqmax_deg",r["arms"][s]["dqmax_deg"]),"held":r["arms"][s].get("held",False)}
                         for r in state._RECS for s in ("R","L") if r["arms"][s].get("raw_jump")]
        state.njump=len(jump_frames)
        def _series(path,default=None):
            vals=[]
            for r in state._RECS:
                try:
                    x=r
                    for pth in path: x=x[pth]
                    vals.append(float(x))
                except Exception:
                    if default is not None: vals.append(default)
            return vals
        r_tcp=_series(("arms_metric","R","tcp_cup_dist_mm")); l_tcp=_series(("arms_metric","L","tcp_cup_dist_mm"))
        r_grip=_series(("grip","R")); l_grip=_series(("grip","L"))
        cup_pos=[]
        for r in state._RECS:
            if "cup_pos" in r:
                cup_pos.append(np.asarray(r["cup_pos"],float))
        cup_pos=np.asarray(cup_pos,float) if cup_pos else np.zeros((0,3))
        def _grasp_hold_diagnosis(side,near_limit_mm=120.0):
            rows=[r for r in state._RECS if isinstance(r.get("policy"),dict) and r["policy"].get("grasp_phase") in ("lift","hold")]
            rows=[r for r in rows if "cup_pos" in r and side in r.get("arms_metric",{})]
            if not rows:
                return {}
            cup_arr=np.asarray([r["cup_pos"] for r in rows],float)
            tcp_arr=np.asarray([r["arms_metric"][side]["tcp_pos"] for r in rows],float)
            rel=cup_arr-tcp_arr
            rel0=rel[0].copy()
            rel_delta=np.linalg.norm(rel-rel0[None,:],axis=1)
            tcp_dz=float(tcp_arr[-1,2]-tcp_arr[0,2])
            cup_dz=float(cup_arr[-1,2]-cup_arr[0,2])
            dists=np.asarray([float(r["arms_metric"][side]["tcp_cup_dist_mm"]) for r in rows],float)
            contact_rows=[r for r in rows if isinstance(r.get("grasp_contact"),dict)]
            contact_counts=[int(r.get("grasp_contact",{}).get(side,{}).get("count",0)) for r in rows]
            contact_forces=[float(r.get("grasp_contact",{}).get(side,{}).get("normal_force_sum_n",0.0)) for r in rows]
            has_contact_log=bool(contact_rows)
            contact_frames=sum(1 for x in contact_counts if x>0)
            follow_ratio=(cup_dz/tcp_dz) if abs(tcp_dz)>1e-9 else 0.0
            near_frames=int(np.sum(dists < float(near_limit_mm)))
            near_ratio=float(near_frames)/max(len(rows),1)
            rel_change_mm=float(np.linalg.norm(rel[-1]-rel0)*1000.0)
            rel_rms_mm=float(np.sqrt(np.mean(rel_delta**2))*1000.0)
            contact_ratio=float(contact_frames)/max(len(rows),1) if has_contact_log else None
            follows_z=bool(tcp_dz > 0.05 and cup_dz > 0.05 and 0.65 <= follow_ratio <= 1.35)
            stable_rel=bool(rel_change_mm <= 35.0 and rel_rms_mm <= 25.0)
            # How near the TCP has to stay is the object's own size: the cup
            # is gripped at its centre, the container at an end 121 mm from its
            # origin, and one threshold cannot describe both.
            near_ok=bool(near_ratio >= 0.8 and float(np.max(dists)) <= float(near_limit_mm))
            contact_ok=bool((not has_contact_log) or contact_ratio >= 0.5)
            held_like=bool(follows_z and stable_rel and near_ok and contact_ok)
            return {
                "phase":"lift+hold",
                "frames":len(rows),
                "held_like":held_like,
                "follows_z":follows_z,
                "stable_relative_pose":stable_rel,
                "near_tcp":near_ok,
                "contact_continuity":contact_ok if has_contact_log else None,
                "cup_dz_mm":round(cup_dz*1000.0,3),
                "tcp_dz_mm":round(tcp_dz*1000.0,3),
                "cup_dz_over_tcp_dz":round(float(follow_ratio),4),
                "relative_change_final_mm":round(rel_change_mm,3),
                "relative_rms_change_mm":round(rel_rms_mm,3),
                "tcp_cup_dist_minmax_mm":[round(float(np.min(dists)),3),round(float(np.max(dists)),3)],
                "near_limit_mm":round(float(near_limit_mm),3),
                "near_within_limit_frames":near_frames,
                "near_ratio":round(near_ratio,4),
                "has_grasp_contact_log":has_contact_log,
                "contact_frames":int(contact_frames) if has_contact_log else None,
                "contact_ratio":round(float(contact_ratio),4) if has_contact_log else None,
                "normal_force_sum_max_n":round(float(max(contact_forces)),4) if has_contact_log and contact_forces else None,
            }
        _joint_step_deg=[
            float(r["arms"][key].get("dqmax_deg",0.0))
            for r in state._RECS for key in ("R","L")
        ]
        _joint_vel=[
            float(r["arms"][key]["joint_velocity_max_rad_s"])
            for r in state._RECS for key in ("R","L")
            if r["arms"][key].get("joint_velocity_max_rad_s") is not None
        ]
        _joint_acc=[
            float(r["arms"][key]["joint_acceleration_max_rad_s2"])
            for r in state._RECS for key in ("R","L")
            if r["arms"][key].get("joint_acceleration_max_rad_s2") is not None
        ]
        _tcp_residual_mm=[
            float(r["arms"][key].get("res_pos_mm",0.0))
            for r in state._RECS for key in ("R","L")
        ]
        _joint_command_velocity=[
            float(command.get("velocity_max_rad_s",0.0))
            for r in state._RECS
            for sample in r.get("dynamic_grip_samples",[])
            for command in sample.get("arm_command",{}).values()
        ]
        _joint_command_acceleration=[
            float(command.get("acceleration_max_rad_s2",0.0))
            for r in state._RECS
            for sample in r.get("dynamic_grip_samples",[])
            for command in sample.get("arm_command",{}).values()
        ]
        _joint_command_jerk=[
            float(command.get("jerk_max_rad_s3",0.0))
            for r in state._RECS
            for sample in r.get("dynamic_grip_samples",[])
            for command in sample.get("arm_command",{}).values()
        ]
        _joint_command_average_jerk=[
            float(command.get("average_jerk_max_rad_s3",0.0))
            for r in state._RECS
            for sample in r.get("dynamic_grip_samples",[])
            for command in sample.get("arm_command",{}).values()
        ]
        _joint_command_discrete_velocity=[
            float(command.get(
                "discrete_position_step_velocity_max_rad_s",0.0
            ))
            for r in state._RECS
            for sample in r.get("dynamic_grip_samples",[])
            for command in sample.get("arm_command",{}).values()
        ]
        _joint_target_attempt_count=sum(
            state._JOINT_IK_TARGET_ATTEMPT_COUNT.values()
        )
        _joint_target_accept_count=(
            _joint_target_attempt_count-state._JOINT_IK_TARGET_REJECT_COUNT
        )
        _joint_target_stats_by_arm={
            key:{
                "attempted":int(state._JOINT_IK_TARGET_ATTEMPT_COUNT[key]),
                "accepted":int(
                    state._JOINT_IK_TARGET_ATTEMPT_COUNT[key]
                    -sum(state._JOINT_IK_TARGET_REJECT_REASONS[key].values())
                ),
                "rejected":int(
                    sum(state._JOINT_IK_TARGET_REJECT_REASONS[key].values())
                ),
                "reject_reasons":dict(sorted(
                    state._JOINT_IK_TARGET_REJECT_REASONS[key].items()
                )),
            }
            for key in ("R","L")
        }
        _joint_safety_stats_by_arm={}
        for _key in ("R","L"):
            _raw_stats=state._JOINT_IK_SAFETY_STATS[_key]
            _attempts=int(state._JOINT_IK_TARGET_ATTEMPT_COUNT[_key])
            _joint_safety_stats_by_arm[_key]={
                _name:(
                    None
                    if _name in (
                        "initial_min_clearance_mm",
                        "raw_min_clearance_mm",
                        "selected_min_clearance_mm",
                        "trajectory_preview_min_clearance_mm",
                    ) and not np.isfinite(float(_value))
                    else round(float(_value),6)
                    if _name in (
                        "initial_min_clearance_mm",
                        "raw_min_clearance_mm",
                        "selected_min_clearance_mm",
                        "trajectory_preview_min_clearance_mm",
                    )
                    else int(_value)
                )
                for _name,_value in _raw_stats.items()
            }
            _joint_safety_stats_by_arm[_key][
                "mean_scratch_collision_checks_per_target"
            ]=(
                round(
                    float(_raw_stats["scratch_collision_checks"])
                    /float(_attempts),
                    6,
                )
                if _attempts else 0.0
            )
            _decision_partition_count=sum(
                int(_raw_stats[_name])
                for _name in (
                    "full_target",
                    "ik_nonconverged_hold",
                    "collision_clipped",
                    "collision_hold",
                    "trajectory_collision_brake",
                    "trajectory_collision_hold",
                    "ruckig_reject",
                )
            )
            _joint_safety_stats_by_arm[_key][
                "decision_partition_count"
            ]=int(_decision_partition_count)
            _joint_safety_stats_by_arm[_key][
                "decision_partition_matches_attempts"
            ]=bool(_decision_partition_count==_attempts)
            if _decision_partition_count!=_attempts:
                raise RuntimeError(
                    "joint_ik safety decision counts do not partition targets "
                    f"for {_key}: decisions={_decision_partition_count} "
                    f"attempts={_attempts}"
                )
        _pass1_sec=time.time()-state._tloop
        _has_fr3_dynamic_action_schedule=bool(
            state.EMBODIMENT=="fr3"
            and state.A.physics=="dynamic"
            and not state.A.state_log_replay
            and state._FR3_DYNAMIC_SOURCE_HZ is not None
            and state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION
        )
        _fr3_dynamic_control_tick_total=(
            int(sum(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_control_tick_min=(
            int(min(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_control_tick_max=(
            int(max(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_control_tick_histogram=(
            {
                str(ticks):int(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION.count(ticks))
                for ticks in sorted(set(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            }
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_constant_control_repeats=(
            constant_control_ticks_per_source_action(
                float(state._FR3_DYNAMIC_SOURCE_HZ),
                control_frequency_hz=float(state.A.dynamic_control_hz),
            )
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_source_duration=(
            float(len(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            /float(state._FR3_DYNAMIC_SOURCE_HZ)
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_simulated_duration=(
            float(_fr3_dynamic_control_tick_total)/float(state.A.dynamic_control_hz)
            if _fr3_dynamic_control_tick_total is not None else None
        )
        _fr3_dynamic_boundary_phase_errors=(
            np.cumsum(
                np.asarray(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION,dtype=np.float64)
            )/float(state.A.dynamic_control_hz)
            -np.arange(
                1,len(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION)+1,dtype=np.float64
            )/float(state._FR3_DYNAMIC_SOURCE_HZ)
            if _has_fr3_dynamic_action_schedule else None
        )
        _fr3_dynamic_phase_error_min=(
            float(np.min(_fr3_dynamic_boundary_phase_errors))
            if _fr3_dynamic_boundary_phase_errors is not None else None
        )
        _fr3_dynamic_phase_error_max=(
            float(np.max(_fr3_dynamic_boundary_phase_errors))
            if _fr3_dynamic_boundary_phase_errors is not None else None
        )
        _fr3_dynamic_phase_error_max_abs=(
            float(np.max(np.abs(_fr3_dynamic_boundary_phase_errors)))
            if _fr3_dynamic_boundary_phase_errors is not None else None
        )
        _fr3_runtime_control_tick_total=(
            int(state._FR3_DYNAMIC_ENV_STEP_COUNT)
            if _has_fr3_dynamic_action_schedule else None
        )
        if (
            _fr3_dynamic_control_tick_total is not None
            and _fr3_runtime_control_tick_total!=_fr3_dynamic_control_tick_total
        ):
            raise RuntimeError(
                "FR3 runtime controller tick count differs from the source-action "
                "phase schedule: "
                f"runtime={_fr3_runtime_control_tick_total}, "
                f"scheduled={_fr3_dynamic_control_tick_total}"
            )
        _dynamic_action_sim_dt=(
            None
            if state.A.state_log_replay else
            float(state.A.substeps)*float(state.m.opt.timestep)
            if state.A.physics=="dynamic" and state.A.ik_replay else
            _fr3_dynamic_simulated_duration
            /float(len(state._FR3_DYNAMIC_CONTROL_TICKS_PER_ACTION))
            if _has_fr3_dynamic_action_schedule else
            float(state.DYNAMIC_CONTROL_REPEATS)*float(state.env.control_timestep)
            if state.A.physics=="dynamic" else None
        )
        _policy_action_clamp_telemetry=None
        if state.EMBODIMENT=="fr3" and state.POLICY_SOURCE:
            _chunk_clamp_telemetry=[]
            for _chunk_record in state._POLICY_CHUNK_RECORDS:
                _chunk_value=_chunk_record.get("policy_action_clamp_telemetry")
                if not isinstance(_chunk_value,dict):
                    raise RuntimeError(
                        "FR3 policy chunk is missing action-clamp telemetry: "
                        f"chunk_index={_chunk_record.get('chunk_index')}"
                    )
                _chunk_clamp_telemetry.append(_chunk_value)
            _policy_action_clamp_telemetry=(
                pio.aggregate_fr3_policy_action_clamp_telemetry(
                    _chunk_clamp_telemetry,
                    enabled=bool(state.A.policy_action_clamp),
                )
            )
        # Computed once, before the summary literal, so the task-lift block
        # reads the same criterion description the success fields publish.
        _success_summary=summarize_success_records(state._RECS, state._SUCCESS_CRITERION)
        summ={"robot":state.ROBOT,"model_signature":state.MODEL_SIGNATURE,
              "gripper_calibration":state.GRIPPER_CALIBRATION_REPORT,
              "robot_topology":"dual-single-arm",
              **arm_controller_fields(state),
              "controller_action_dim":int(state.env.action_dim),
              "physics":state.A.physics,"objects":state.A.objects,
              **task_binding_fields(state),
              # The layout file and sha256 the env loaded.
              "task_layout":loaded_task_layout(state),
              # The CLI default is "joint_ik" on every rig; only FR3 runs one.
              "fr3_controller":(state.A.fr3_controller if state.EMBODIMENT=="fr3" else None),
              "fr3_controller_path":(controller_path_summary(state.A.physics,state.A.fr3_controller) if state.EMBODIMENT=="fr3" else None),
              "dynamic_control_hz":int(state.A.dynamic_control_hz),
              **fr3_joint_gain_summary(
                  state._joint_cfg,
                  active=(state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik"),
                  joint_kp_override=state.A.fr3_joint_kp,
                  joint_kd_override=state.A.fr3_joint_kd,
              ),
              "fr3_absolute_ik_solve_count":(
                  int(state._JOINT_IK_SOLVE_COUNT) if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_absolute_ik_fail_count":(
                  int(state._JOINT_IK_FAIL_COUNT) if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_ik_home_q_rad":([float(value) for value in state.fr3_robot.FR3_HOME_QPOS]
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" else None),
              "fr3_ik_nullspace":(state.A.fr3_nullspace
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" else None),
              "fr3_ik_nullspace_gain":(effective_nullspace_gain(state.A.fr3_nullspace,state.A.joint_ik_nullspace_gain)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" else None),
              "fr3_ik_nullspace_gain_configured":(float(state.A.joint_ik_nullspace_gain)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" else None),
              "fr3_joint_trajectory_generator":(
                  f"ruckig_{state._JOINT_IK_TRAJECTORY['R'].ruckig_version}_online"
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_trajectory_state_carry":(
                  "q_qdot_qddot"
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_trajectory_position_extrema_gate":(
                  True if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_ik_nonconverged_fail_hold":(
                  bool(state.A.joint_ik_fail_hold)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_arm_desk_collision_gate_clearance_mm":(
                  float(state.A.joint_ik_desk_collision_clearance_mm)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_arm_desk_collision_gate_path_step_deg":(
                  2.0
                  if state._ARM_DESK_COLLISION_GATE is not None else None
              ),
              "fr3_arm_desk_collision_gate_retreat_deg":(
                  0.25
                  if state._ARM_DESK_COLLISION_GATE is not None else None
              ),
              "fr3_arm_desk_collision_gate_ruckig_preview_scope":(
                  "full_candidate_trajectory"
                  if state._ARM_DESK_COLLISION_GATE is not None else None
              ),
              "fr3_arm_desk_collision_gate_ruckig_preview_hz":(
                  float(state.env.control_freq)
                  if state._ARM_DESK_COLLISION_GATE is not None else None
              ),
              "fr3_arm_desk_collision_gate_initial_by_arm":(
                  state._JOINT_IK_INITIAL_SAFETY
                  if state._ARM_DESK_COLLISION_GATE is not None else None
              ),
              "fr3_joint_target_safety_stats_by_arm":(
                  _joint_safety_stats_by_arm
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_target_reject_count":(
                  int(state._JOINT_IK_TARGET_REJECT_COUNT)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_target_attempt_count":(
                  int(_joint_target_attempt_count)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_target_accept_count":(
                  int(_joint_target_accept_count)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_target_reject_ratio":(
                  round(
                      float(state._JOINT_IK_TARGET_REJECT_COUNT)
                      /float(_joint_target_attempt_count),6
                  )
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik"
                  and _joint_target_attempt_count else
                  0.0 if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_target_stats_by_arm":(
                  _joint_target_stats_by_arm
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_command_velocity_limit_rad_s":(
                  float(state.A.joint_ik_max_velocity)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_command_acceleration_limit_rad_s2":(
                  float(state.A.joint_ik_max_acceleration)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_command_jerk_limit_rad_s3":(
                  float(state.A.joint_ik_max_jerk)
                  if state.EMBODIMENT=="fr3" and state.A.physics=="dynamic" and state.A.fr3_controller=="joint_ik" else None
              ),
              "fr3_joint_command_velocity_max_rad_s":(
                  round(max(_joint_command_velocity),6)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" and _joint_command_velocity else None
              ),
              "fr3_joint_command_acceleration_max_rad_s2":(
                  round(max(_joint_command_acceleration),6)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" and _joint_command_acceleration else None
              ),
              "fr3_joint_command_jerk_max_rad_s3":(
                  round(max(_joint_command_jerk),6)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" and _joint_command_jerk else None
              ),
              "fr3_joint_command_average_jerk_max_rad_s3":(
                  round(max(_joint_command_average_jerk),6)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" and _joint_command_average_jerk else None
              ),
              "fr3_joint_command_discrete_step_velocity_max_rad_s":(
                  round(max(_joint_command_discrete_velocity),6)
                  if state.EMBODIMENT=="fr3" and state.A.fr3_controller=="joint_ik" and _joint_command_discrete_velocity else None
              ),
              "initial_arm_joints_json":(
                  state._INITIAL_ARM_JOINT_POSE.source_path
                  if state._INITIAL_ARM_JOINT_POSE is not None else None
              ),
              "initial_arm_joints_sha256":(
                  state._INITIAL_ARM_JOINT_POSE.source_sha256
                  if state._INITIAL_ARM_JOINT_POSE is not None else None
              ),
              "initial_arm_joint_targets_rad":(
                  {
                      key:[float(value) for value in state._INITIAL_ARM_JOINT_Q[key]]
                      for key in ("R","L")
                  }
                  if state._INITIAL_ARM_JOINT_Q is not None else None
              ),
              "initial_gripper_raw_rad":(
                  {
                      key:float(state._INITIAL_GRIP_Q[key])
                      for key in ("R","L")
                  }
                  if state._INITIAL_GRIP_Q is not None else None
              ),
              "initial_arm_joint_ready_gate":state._READY_GATE_REPORT,
              "dynamic_ready_requested_s":float(state.A.dynamic_ready_sec),
              "dynamic_ready_steps":(
                  int(state._DYNAMIC_READY_STEPS)
                  if state._DYNAMIC_READY_STEPS is not None else None
              ),
              "dynamic_ready_sim_dt_s":(
                  round(float(state._DYNAMIC_READY_STEPS) / float(state.env.control_freq), 6)
                  if state._DYNAMIC_READY_STEPS is not None else None
              ),
              "dynamic_ready_move_steps":(
                  int(state._READY_MOVE_STEPS)
                  if state.FR3_STRATEGY.supports_joint_ready
                  and state._READY_MOVE_STEPS is not None else None
              ),
              "dynamic_ready_hold_steps":(
                  int(state._READY_HOLD_STEPS)
                  if state.FR3_STRATEGY.supports_joint_ready
                  and state._READY_HOLD_STEPS is not None else None
              ),
              "dynamic_object_settle_enabled":bool(
                  float(state.A.dynamic_object_settle_max_sec)>0.0
              ),
              "dynamic_object_settle":state.OBJECT_SETTLE_REPORT,
              "dynamic_object_settle_requested":{
                  "min_sec":float(state.A.dynamic_object_settle_min_sec),
                  "max_sec":float(state.A.dynamic_object_settle_max_sec),
                  "stable_sec":float(state.A.dynamic_object_settle_stable_sec),
                  "linear_speed_mm_s":float(
                      state.A.dynamic_object_settle_linear_speed_mm_s
                  ),
                  "angular_speed_rad_s":float(
                      state.A.dynamic_object_settle_angular_speed_rad_s
                  ),
                  "linear_acceleration_m_s2":float(
                      state.A.dynamic_object_settle_linear_acceleration_m_s2
                  ),
                  "angular_acceleration_rad_s2":float(
                      state.A.dynamic_object_settle_angular_acceleration_rad_s2
                  ),
                  "position_drift_mm":float(
                      state.A.dynamic_object_settle_position_drift_mm
                  ),
                  "orientation_drift_deg":float(
                      state.A.dynamic_object_settle_orientation_drift_deg
                  ),
              },
              "grip_ctrl":state.A.grip_ctrl,"grip_ctrl_effective":state.GRIP_CTRL_EFFECTIVE,
              "dynamic_grip_cbpc_requested":bool(state.A.dynamic_grip_cbpc),
              "dynamic_gripper_control_tick_count":(
                  int(state.DYNAMIC_YUBI_HOOK.tick_count)
                  if state.DYNAMIC_YUBI_HOOK is not None else None
              ),
              "dynamic_gripper_control_target_updates":(
                  int(state.DYNAMIC_YUBI_HOOK.target_update_count)
                  if state.DYNAMIC_YUBI_HOOK is not None else None
              ),
              "dynamic_gripper_control_cadence_violations":(
                  int(state.DYNAMIC_YUBI_HOOK.cadence_violation_count)
                  if state.DYNAMIC_YUBI_HOOK is not None else None
              ),
              "dynamic_cbpc_tick_count":int(state.DYNAMIC_YUBI_HOOK.tick_count) if state.DYNAMIC_YUBI_HOOK is not None and state.A.grip_ctrl=="cbpc" else None,
              "dynamic_cbpc_target_updates":int(state.DYNAMIC_YUBI_HOOK.target_update_count) if state.DYNAMIC_YUBI_HOOK is not None and state.A.grip_ctrl=="cbpc" else None,
              "dynamic_cbpc_cadence_violations":int(state.DYNAMIC_YUBI_HOOK.cadence_violation_count) if state.DYNAMIC_YUBI_HOOK is not None and state.A.grip_ctrl=="cbpc" else None,
              "substeps":int(state.A.substeps),
              "dynamic_subframes_requested":int(state.DYNAMIC_SUBFRAMES_REQUESTED),
              "dynamic_subframes":int(state.DYNAMIC_SUBFRAMES),"rendered_frames":int(state._render_count),
              "output_fps":int(state.OUTPUT_FPS),
              "policy_action_hz_requested":float(state.A.policy_action_hz),
              "policy_action_hz_effective":(
                  float(state._FR3_DYNAMIC_SOURCE_HZ)
                  if state._FR3_DYNAMIC_SOURCE_HZ is not None else None
              ),
              "policy_action_dt_s":(
                  round(1.0/float(state._FR3_DYNAMIC_SOURCE_HZ),9)
                  if state._FR3_DYNAMIC_SOURCE_HZ is not None else None
              ),
              "policy_action_dt_nominal_s":(
                  round(1.0/float(state._FR3_DYNAMIC_SOURCE_HZ),9)
                  if state._FR3_DYNAMIC_SOURCE_HZ is not None else None
              ),
              "mujoco_timestep_s":float(state.m.opt.timestep),
              "mujoco_integrator":state.MUJOCO_INTEGRATOR,
              "mujoco_integrator_selection":state.MUJOCO_INTEGRATOR_SELECTION,
              "cup_mass_xml_kg":state._CUP_MASS_XML,
              "cup_mass_effective_kg":state._CUP_MASS_EFFECTIVE,
              "dynamic_control_repeats_requested":state.DYNAMIC_CONTROL_REPEATS_USER_REQUESTED,
              "dynamic_control_repeats_auto_resolved":(
                  bool(
                      state._FR3_DYNAMIC_SOURCE_HZ is not None
                      and state.A.physics=="dynamic"
                      and not state.A.state_log_replay
                      and state.DYNAMIC_CONTROL_REPEATS_USER_REQUESTED==0
                  )
              ),
              "dynamic_control_repeats":(
                  None
                  if state.A.ik_replay or state.A.state_log_replay
                  else _fr3_constant_control_repeats
                  if _has_fr3_dynamic_action_schedule
                  else int(state.DYNAMIC_CONTROL_REPEATS)
                  if state.A.physics=="dynamic"
                  else None
              ),
              "fr3_dynamic_tick_schedule":(
                  "source_to_controller_cumulative_nearest_half_up"
                  if _has_fr3_dynamic_action_schedule else None
              ),
              "fr3_dynamic_control_ticks_per_action_min":(
                  _fr3_dynamic_control_tick_min
              ),
              "fr3_dynamic_control_ticks_per_action_max":(
                  _fr3_dynamic_control_tick_max
              ),
              "fr3_dynamic_control_tick_histogram":(
                  _fr3_dynamic_control_tick_histogram
              ),
              "fr3_dynamic_control_tick_count_total":(
                  _fr3_dynamic_control_tick_total
              ),
              "fr3_dynamic_runtime_control_tick_count_total":(
                  _fr3_runtime_control_tick_total
              ),
              "fr3_dynamic_runtime_tick_count_matches_schedule":(
                  _fr3_runtime_control_tick_total==_fr3_dynamic_control_tick_total
                  if _fr3_dynamic_control_tick_total is not None else None
              ),
              "fr3_dynamic_source_duration_s":(
                  round(_fr3_dynamic_source_duration,9)
                  if _fr3_dynamic_source_duration is not None else None
              ),
              "fr3_dynamic_simulated_duration_s":(
                  round(_fr3_dynamic_simulated_duration,9)
                  if _fr3_dynamic_simulated_duration is not None else None
              ),
              "fr3_dynamic_timing_error_s":(
                  round(
                      _fr3_dynamic_simulated_duration
                      -_fr3_dynamic_source_duration,
                      9,
                  )
                  if (
                      _fr3_dynamic_simulated_duration is not None
                      and _fr3_dynamic_source_duration is not None
                  ) else None
              ),
              "fr3_dynamic_boundary_phase_error_s_min":(
                  round(_fr3_dynamic_phase_error_min,9)
                  if _fr3_dynamic_phase_error_min is not None else None
              ),
              "fr3_dynamic_boundary_phase_error_s_max":(
                  round(_fr3_dynamic_phase_error_max,9)
                  if _fr3_dynamic_phase_error_max is not None else None
              ),
              "fr3_dynamic_boundary_phase_error_s_max_abs":(
                  round(_fr3_dynamic_phase_error_max_abs,9)
                  if _fr3_dynamic_phase_error_max_abs is not None else None
              ),
              "fr3_dynamic_boundary_phase_error_bound_s":(
                  round(0.5/float(state.A.dynamic_control_hz),9)
                  if _has_fr3_dynamic_action_schedule else None
              ),
              "dynamic_action_sim_dt_s":(
                  round(_dynamic_action_sim_dt,9)
                  if _dynamic_action_sim_dt is not None else None),
              "playback_action_dt_s":round(float(state.DYNAMIC_SUBFRAMES)/float(state.OUTPUT_FPS),6),
              "dynamic_time_compression":(
                  round(_dynamic_action_sim_dt /
                        (float(state.DYNAMIC_SUBFRAMES)/float(state.OUTPUT_FPS)),6)
                  if _dynamic_action_sim_dt is not None else None),
              "replay_arm_source":(
                  "generation_joint_q_kinematic_pin" if state.A.ik_replay else
                  "generation_state_log" if state.A.state_log_replay else
                  "cartesian_pose" if state.A.replay else None),
              "replay_object_source":(
                  state.A.replay_object_source if state.A.replay else
                  "generation_state_log" if state.A.state_log_replay else
                  "physics" if state.A.ik_replay else None),
              "adopted_action_chunks":int(state.A.exec),"action_horizon":int(state.A.ah) if state.A.ah>0 else None,
              "policy_delta_anchor":state.A.policy_delta_anchor,
              "policy_pose_repr":state.A.policy_pose_repr if state.POLICY_SOURCE else None,
              "policy_action_clamp":bool(state.A.policy_action_clamp) if state.EMBODIMENT=="fr3" and state.POLICY_SOURCE else None,
              "policy_action_clamp_telemetry":_policy_action_clamp_telemetry,
              "rf_overlap":int(state.A.rf_overlap),
              "rf_overlap_sampler":state.A.rf_overlap_sampler,
              "rf_overlap_sampler_configured":state.A.rf_overlap_sampler,
              "rf_overlap_sampler_effective":"single",
              "rf_overlap_collapse_reason":None,
              "rf_overlap_prev_weight":None,
              "rf_overlap_prev_weight_configured":None,
              "rf_overlap_prev_weight_effective":None,
              "rf_overlap_prev_weight_dynamic":False,
              "rf_overlap_prev_weight_schedule":None,
              "rf_overlap_step_shift":int(state.A.rf_overlap_step_shift),
              "rf_overlap_step_shift_configured":int(state.A.rf_overlap_step_shift),
              "rf_overlap_step_shift_effective":0,
              "rf_overlap_sample_num_steps":None,
              "policy_replan_stride":int(state.A.exec),
              "policy_execution_slice_initial":[0,int(state.A.exec)],
              "policy_execution_slice_conditioned":(
                  [int(state.A.rf_overlap),int(state.A.rf_overlap+state.A.exec)]
                  if state.A.rf_overlap else None),
              "policy_chunks_jsonl":_POLICY_CHUNKS_PATH or None,
              "policy_chunks":len(state._POLICY_CHUNK_RECORDS),
              "rf_prefix_roundtrip_max_abs":max(
                  [float(r["prefix_roundtrip_max_abs"]) for r in state._POLICY_CHUNK_RECORDS
                   if r.get("prefix_roundtrip_max_abs") is not None],default=None),
              "rf_bridge_roundtrip_max_abs":max(
                  [float(r["bridge_roundtrip_max_abs"]) for r in state._POLICY_CHUNK_RECORDS
                   if r.get("bridge_roundtrip_max_abs") is not None],default=None),
              "rf_bridge_rebased_prev_distance_max_abs":max(
                  [float(r["bridge_rebased_prev_distance_max_abs"])
                   for r in state._POLICY_CHUNK_RECORDS
                   if r.get("bridge_rebased_prev_distance_max_abs") is not None],default=None),
              "grip_cfg":(
                  state.A.grip_cfg
                  if state.YUBI is not None and state.A.grip_ctrl=="cbpc" else None
              ),
              "torque_scale":float(state.A.torque_scale),
              "grip_torque_mult":float(state.A.grip_torque_mult),
              "grasp_contract":(
                  "mujoco_contact_dynamics_with_cbpc_no_latch"
                  if state.A.grasp_mode=="physics" and state.A.grip_ctrl=="cbpc" else
                  "mujoco_contact_dynamics_no_latch"
                  if state.A.grasp_mode=="physics" else
                  "inner_pad_contact_angle_rule_weld_equality"
              ),
              "physics_grasp_has_pose_latch":False if state.A.grasp_mode=="physics" else None,
              "cbpc_torque_limit_per_hand_nm":(
                  float(state.YUBI.cfg.resolve_torque_limit(opening=False))
                  if state.YUBI is not None and state.A.grip_ctrl=="cbpc" else None
              ),
              "cbpc_torque_limit_per_jaw_nm":(
                  float(state.YUBI.cfg.resolve_torque_limit(opening=False))/2.0
                  if state.YUBI is not None and state.A.grip_ctrl=="cbpc" else None
              ),
              "grip_torque_mult_affects_effective_controller":bool(
                  state.A.grip_ctrl != "cbpc"
              ),
              "egl_gpu_reservation":{
                  "guarded":bool(os.environ.get("REAL2SIM_EGL_GPU_LOCK")),
                  "gpu_id":os.environ.get("REAL2SIM_EGL_GPU_ID"),
                  "lock_path":os.environ.get("REAL2SIM_EGL_GPU_LOCK"),
                  "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "mujoco_egl_device_id":os.environ.get("MUJOCO_EGL_DEVICE_ID"),
              },
              "grasp_mode":state.A.grasp_mode,
              "rule_grasp_hand":state.A.rule_grasp_hand if state.A.grasp_mode=="rule" else None,
              "rule_grasp_hold_mode":"weld_equality" if state.A.grasp_mode=="rule" else None,
              "rule_grasp_predicate":"opposed_pinch_seated_v2" if state.A.grasp_mode=="rule" else None,
              "rule_grasp_inner_pad_only":bool(state.A.grasp_mode=="rule"),
              "rule_grasp_effort_eps":float(state.A.rule_grasp_effort_eps),
              "rule_grasp_policy_eps_rad":float(state.A.rule_grasp_policy_eps_rad),
              "rule_grasp_release_eps_rad":float(state.A.rule_grasp_release_eps_rad),
              "rule_grasp_events":(
                  state._STATE_LOG_SOURCE_SUMMARY.get(
                      "rule_grasp_events",state.RULE_GRASP["events"]
                  )
                  if state.A.state_log_replay
                  and isinstance(state._STATE_LOG_SOURCE_SUMMARY,dict)
                  and state.A.grasp_mode=="rule"
                  else state.RULE_GRASP["events"] if state.A.grasp_mode=="rule" else []
              ),
              "rule_grasp_final":(
                  state._STATE_LOG_SOURCE_SUMMARY.get(
                      "rule_grasp_final",_rule_grasp_status(state)
                  )
                  if state.A.state_log_replay
                  and isinstance(state._STATE_LOG_SOURCE_SUMMARY,dict)
                  and state.A.grasp_mode=="rule"
                  else _rule_grasp_status(state) if state.A.grasp_mode=="rule" else None
              ),
              "debug_grip_actuator":bool(state.A.debug_grip_actuator),
              "debug_grasp_contact":bool(state.A.debug_grasp_contact),
              "grasp_contact_tol_mm":float(state.A.grasp_contact_tol_mm),
              "overview_camera":{"azimuth":float(state.A.view_azimuth),"elevation":float(state.A.view_elevation),
                                 "distance":float(state.A.view_distance),"lookat":[round(float(x),5) for x in state._VIEW_LOOKAT]},
              "overview_render_pipeline":(
                  "deferred_isolated_egl_after_wrist_teardown"
                  if state._DEFER_OVERVIEW_RENDER else "inline_egl"
              ),
              "video_wrist_insets_enabled":bool(state._VIDEO_NEEDS_WRISTS),
              "video_wrist_inset_source":("off" if not state._VIDEO_NEEDS_WRISTS else "saved" if state._STATE_REPLAY_POLICY_INSETS is not None else "policy_observation" if state._POLICY_INSET_INPUT else "rendered"),
              "video_wrist_renderer_lifecycle":(
                  "per_frame_locked_create_capture_close"
                  if state._VIDEO_WRIST_EPHEMERAL
                  else "persistent"
                  if state._WILL_CREATE_VIDEO_WRIST_RENDERER
                  else None
              ),
              "video_wrist_renderer_stats":(
                  dict(state._VIDEO_WRIST_RENDERER_STATS)
                  if state._VIDEO_WRIST_EPHEMERAL else None
              ),
              "video_sim_panel_x_offset":int(640 if state.realcap is not None and not state.A.clean_video else 0),
              "video_wrist_inset_layout":{
                  "version":(
                      3 if state._STATE_REPLAY_POLICY_INSETS is not None
                      and state.A.state_replay_policy_obs_display=="raw224" else 2
                  ),
                  "left_xywh":[4,4,180,135],
                  "right_xywh":[188,4,180,135],
              },
              "video_wrist_display_contract":(
                  "raw_policy_tensor_full_canvas_resize_v1"
                  if state._STATE_REPLAY_POLICY_INSETS is not None
                  and state.A.state_replay_policy_obs_display=="raw224" else
                  "metadata_unpad_aspect_fit_v1" if state._POLICY_INSET_INPUT else
                  "source_aspect_fit_v1"
              ),
              "video_wrist_display_transforms":({
                  side:_raw_policy_display_transform_record()
                  for side in ("L","R")
              } if state._STATE_REPLAY_POLICY_INSETS is not None
                  and state.A.state_replay_policy_obs_display=="raw224" else {
                  side:_policy_display_transform_record(state.POLICY_OBS_DISPLAY_TRANSFORM[side])
                  for side in ("L","R")
              } if state._POLICY_INSET_INPUT else None),
              "state_replay_policy_obs_dir":(
                  os.path.abspath(state.A.state_replay_policy_obs_dir)
                  if state._STATE_REPLAY_POLICY_INSETS is not None else None
              ),
              "state_replay_policy_obs_unique_pairs":(
                  state._STATE_REPLAY_POLICY_INSETS.unique_pairs
                  if state._STATE_REPLAY_POLICY_INSETS is not None else None
              ),
              "deferred_pass1_compatibility_render_skipped":bool(
                  state._PASS1_OVERVIEW_RENDERER_RELEASED
              ),
              "pass1_overview_renderer_released_before_wrist":bool(
                  state._PASS1_OVERVIEW_RENDERER_RELEASED
              ),
              "finger_friction":state.A.grip_friction,"rubber_friction":state.A.rubber_friction,
              "finger_solref":state.A.solref,"finger_priority":int(state.A.grip_priority),
              "rubber_solref":state.A.rubber_solref,"rubber_solimp":state.A.rubber_solimp,
              "rubber_priority":int(state.A.rubber_priority),
              "frames":state.N,"loop_sec":None,"sec_per_frame":None,"rollout_pass1_sec":round(_pass1_sec,2),"max_pen_mm":round(state.maxpen,3),"frames_with_pen":state.npen,"ik_fail":state.nfail,
              "ik_jumps":state.njump,"jump_deg":state.A.jump_deg,"jump_frames":jump_frames,
              "arm_joint_jumps":state.njump,
              "arm_joint_step_max_deg":round(max(_joint_step_deg),3) if _joint_step_deg else 0.0,
              "arm_joint_velocity_max_rad_s":round(max(_joint_vel),5) if _joint_vel else None,
              "arm_joint_acceleration_max_rad_s2":round(max(_joint_acc),5) if _joint_acc else None,
              "arm_tcp_target_residual_max_mm":round(max(_tcp_residual_mm),3) if _tcp_residual_mm else None,
              "raw_ik_jumps":len(raw_jump_frames),"raw_jump_frames":raw_jump_frames,"ik_jump_reject":state.A.ik_jump_reject,
              "min_R_tcp_cup_dist_mm":round(min(r_tcp),3) if r_tcp else None,
              "min_L_tcp_cup_dist_mm":round(min(l_tcp),3) if l_tcp else None,
              "R_grip_minmax":[round(min(r_grip),5),round(max(r_grip),5)] if r_grip else None,
              "L_grip_minmax":[round(min(l_grip),5),round(max(l_grip),5)] if l_grip else None,
              # Legacy fixed-joint, pre-loop-relative lift (kept for its
              # consumers, labelled) beside the task-aware table-relative one.
              **legacy_target_lift_fields(state,cup_pos),
              **task_lift_fields(state._RECS,_success_summary.get("success_criterion")),
              "policy_freeze_left":bool(state.A.policy_freeze_left),
              "policy_batch_server":state.A.policy_batch_server if state.A.policy_batch_server else None,
              "policy_batch_client_id":state._cid if state.A.policy_batch_server and state._cid is not None else None,
              "policy_image_source":state.A.policy_image_source if state.POLICY_SOURCE else None,
              "policy_image_episode":state.A.policy_image_episode if state.POLICY_SOURCE and (state.A.policy_image_source=="lerobot" or state.A.policy_pose_source=="lerobot") else None,
              "policy_anchor_requested":str(state.A.policy_anchor),
              "policy_anchor_effective":state.POLICY_ANCHOR_EFFECTIVE,
              "policy_anchor_episode":state.POLICY_ANCHOR_EPISODE,
              "policy_image_index":state.A.policy_image_index if state.POLICY_SOURCE and (state.A.policy_image_source=="lerobot" or state.A.policy_pose_source=="lerobot") else None,
              "policy_image_loop":bool(state.A.policy_image_loop) if state.POLICY_SOURCE and (state.A.policy_image_source=="lerobot" or state.A.policy_pose_source=="lerobot") else None,
              "policy_pose_source":state.A.policy_pose_source if state.POLICY_SOURCE else None,
              # Sign of the relative-hand state quaternion, the norm stats it was
              # checked against (sha256, qw q01/q99) and what was actually sent.
              "policy_state_quat_hemisphere":getattr(state.A,"policy_state_quat_hemisphere",None) if state.POLICY_SOURCE else None,
              "policy_state_quat_hemisphere_provenance":(
                  {**state.POLICY_STATE_QUAT_HEMISPHERE,
                   "sent_observations":state.POLICY_STATE_QUAT_ENCODER.summary()}
                  if state.POLICY_SOURCE and getattr(state,"POLICY_STATE_QUAT_HEMISPHERE",None) is not None
                  else None),
              "policy_image_resize":state.A.policy_image_resize if state.POLICY_SOURCE else None,
              "policy_yubi_render_profile":state.A.policy_yubi_render_profile,  # the display wrist insets use it too
              "policy_wrist_backend":state.A.policy_wrist_backend,
              "gs_kb4_near_cull":state.A.gs_kb4_near_cull,
              "gs_kb4_near_m":(state.gsc.kb4_near_meters() if getattr(state,"gsc",None) is not None else None),
              "policy_wrist_intrinsics":policy_wrist_intrinsics(state),
              "policy_wrist_renderer_lifecycle":(
                  "per_observation_locked_create_capture_close"
                  if state._POLICY_WRIST_EPHEMERAL
                  else "persistent"
                  if state.POLICY and state.A.policy_image_source=="sim"
                  else None
              ),
              "policy_wrist_renderer_stats":(
                  dict(state._POLICY_WRIST_RENDERER_STATS)
                  if state._POLICY_WRIST_EPHEMERAL
                  else None
              ),
              "policy_wrist_contrast":float(state.A.policy_wrist_contrast) if state.POLICY_SOURCE else None,
              "policy_wrist_preprocess_order":state._POLICY_WRIST_PREPROCESS_ORDER if state.POLICY_SOURCE else None,
              "policy_obs_dir":state.POLICY_OBS_LOGDIR if state.POLICY_SOURCE else None,
              "policy_obs_saved":len(state.POLICY_OBS_LOGGED) if state.POLICY_SOURCE else None,
              "dump_obs":state.A.dump_obs if state.POLICY_SOURCE and state.A.dump_obs else None,
              "init_state_in":state.A.init_state_in or None,
              "init_state_out":state.A.init_state_out or None,
              "mimicgen_init":state.MIMICGEN_INIT_REPORT,
              "deterministic_seed":int(state.A.deterministic_seed),
              "state_dump_dir":state.A.state_dump_dir or None,
              "state_dump_frames":sorted(int(x) for x in state.STATE_DUMP_FRAMES),
              "state_dump_on_rule_grasp":bool(state.A.state_dump_on_rule_grasp),
              "state_dump_on_success":bool(state.A.state_dump_on_success),
              "state_log_replay":state.A.state_log_replay,
              "execution_class":state.EXECUTION_CLASS,
              "official_run_manifest":state.OFFICIAL_RUN_MANIFEST_PATH,
              "official_worker_contract":state.OFFICIAL_WORKER_CONTRACT_PATH,
              "asset_contract_sha256":(
                  state.ASSET_CONTRACT_REPORT["contract_digest"]
                  if state.ASSET_CONTRACT_REPORT is not None else None
              ),
              "scene_contract":({
                  "schema_version":state.SCENE_CONTRACT_REPORT["schema_version"],
                  "id":state.SCENE_CONTRACT_REPORT["id"],
                  "name":state.SCENE_CONTRACT_REPORT["name"],
                  "tag":state.A.official_scene_tag,
                  "sha256":state.SCENE_CONTRACT_REPORT["contract_digest"],
                  "manifest":os.fspath(state.SCENE_CONTRACT_REPORT["manifest_path"]),
                  "table_appearance":"embedded_in_3dgs",
                  "table_collision_parts":len(state.SCENE_CONTRACT_REPORT["collision_files"]),
              } if state.SCENE_CONTRACT_REPORT is not None else None),
              "replay":state.A.replay,"policy":state.A.policy,"instruction":state.A.instruction,"desk":state.A.desk,"gs":state.A.gs,
              "gs_background":state.A.gs_background,"gs_background_source":state.A.gs_background_source,
              "gs_composite_mode":state.A.gs_composite_mode,
              "obs_gs_only":bool(state.A.obs_gs_only),
              "gs_port":state.A.gs_port,"gs_rot":state.A.gs_rot,
              "gs_w2gs":reported_gs_w2gs(state.A),
              "gs_calibration_world_offset_m":[float(x) for x in state.A.gs_calibration_world_offset],
              "gs_w2gs_raw":state.GS_RAW_MAT_W2GS.tolist() if state.GS_RAW_MAT_W2GS is not None else None,
              "gs_w2gs_effective":state.GS_EFFECTIVE_MAT_W2GS.tolist() if state.GS_EFFECTIVE_MAT_W2GS is not None else None,
              "gs_color_gain_rgb":[float(x) for x in state.A.gs_color_gain_rgb],
              "gs_color_gamma_rgb":[float(x) for x in state.A.gs_color_gamma_rgb],
              "gs_render_timeout_sec":float(state.A.gs_render_timeout_sec) if state.gsc is not None else None,
              "gs_render_max_attempts":int(state.A.gs_render_attempts) if state.gsc is not None else None,
              "gs_render_retry_delay_sec":float(state.A.gs_render_retry_delay_sec) if state.gsc is not None else None,
              "gs_render_strict":bool(state.gsc.strict) if state.gsc is not None else None,
              "gs_overview_attempts":int(state.GS_OVERVIEW_ATTEMPTS) if state.gsc is not None and not state.A.no_render else None,
              "gs_render_attempt_count":int(state.gsc.attempt_count) if state.gsc is not None else None,"gs_render_request_count":int(state.gsc.request_count) if state.gsc is not None else None,
              "gs_render_control_request_count":int(state.gsc.control_request_count) if state.gsc is not None else None,
              "gs_render_failures":int(state.gsc.fail_count) if state.gsc is not None else None}
        # Locked distributions this worker verified before importing them
        # (OpenPI client side for policy runs); None when nothing was checked.
        summ["locked_imports"]=getattr(state,"LOCKED_IMPORTS",None)
        summ["render_state_log"]=build_render_state_summary_contract(
            state._RENDER_STATE_CONTRACT,
            rows_written=len(state._RECS),
            complete=bool(len(state._RECS)==state.N),
        )
        summ["render_state_replay"]=(
            {
                "source_summary":state._STATE_LOG_SOURCE_SUMMARY_PATH,
                "rows_validated":int(state._STATE_RENDER_ROWS_VALIDATED),
                "rows_applied":int(state._STATE_RENDER_ROWS_APPLIED),
                "max_qpos_apply_error":float(state._STATE_RENDER_MAX_APPLY_ERROR),
                "target_baseline_source":state._STATE_REPLAY_TARGET_BASELINE_SOURCE,
                "all_rendered_rows_applied":bool(
                    state._STATE_RENDER_ROWS_APPLIED==state.N
                ),
            }
            if state.A.state_log_replay else None
        )
        summ.update(_success_summary)
        if state.A.frame_hz>0 or state._FR3_DYNAMIC_SOURCE_HZ is not None:
            summ["frame_dt_s"]=round(float(state.FRAME_DT_S),6)
            summ["frame_hz_effective"]=round(float(state.FRAME_HZ_EFFECTIVE),6)
        if state.A.max_wall_sec>0:
            summ["max_wall_sec"]=float(state.A.max_wall_sec)
            summ["wall_stopped_early"]=bool(state.WALL_STOPPED)
        if state.A.grasp_test:
            summ["grasp_test"] = {"hand": state.GH, "grasp_z": state.A.grasp_z, "lift_h": state.A.lift_h,
                                  "grasp_xy": [round(state._gx,5), round(state._gy,5)], "yaw": state.A.grasp_yaw,
                                  "close_grip": state.A.close_grip, "open_grip": state.A.open_grip,
                                  "phase_bounds": state._PB}
            if state.GH=="both":
                if state._GRASP_SPAN is not None:
                    summ["grasp_test"]["span_m"]=round(float(state._GRASP_SPAN),6)
                summ["grasp_test"]["grasp_by_hand"]={
                    hand:[round(float(v),6) for v in poses["grasp"]]
                    for hand,poses in state._P_BY_HAND.items()
                }
                # Each hand grips an end of the object, so "near" is measured
                # against the reach from its origin to that end plus the
                # gripper's own standoff, not the cup's 120 mm.
                _near_limit_mm=1000.0*float(
                    max(
                        np.linalg.norm(np.asarray(poses["grasp"],dtype=float)
                                       - np.asarray(state.cup,dtype=float))
                        for poses in state._P_BY_HAND.values()
                    )
                ) + 40.0
                summ["grasp_hold_diagnosis"]={
                    hand:_grasp_hold_diagnosis(hand,near_limit_mm=_near_limit_mm)
                    for hand in ("R","L")
                }
            else:
                summ["grasp_hold_diagnosis"]=_grasp_hold_diagnosis(state.GH)
            summ["grasp_test"]["approach"]=state.A.approach
            if state.A.approach=="side":
                summ["grasp_test"].update({"approach_azim":state.A.approach_azim,"side_standoff":state.A.side_standoff,"grasp_reach":state.A.grasp_reach})
        _,_summary_path=resolve_summary_paths(state.LOGDIR,state.A.summary_path)
        summ["log_jsonl"]=os.path.join(state.LOGDIR,"sim_log.jsonl"); summ["logdir"]=state.LOGDIR
        summ["summary_path"]=_summary_path
        # What the episode started from, so a batch of zeros can be told apart
        # from a batch that never had the objects where the layout says.
        _start_state=getattr(state,"_start_state_report",None)
        if _start_state is not None:
            summ["start_state"]=_start_state.as_dict()
        print(f"[sim] arm joint jumps (Δjoint>{state.A.jump_deg}deg/source-frame): {state.njump}"+(f" -> {jump_frames}" if jump_frames else " (none)"),flush=True)
    _video_pipeline_started=time.time()
    try:
        if state._render_count:
            _written_frame_count=write_rgb_video(
                _iter_output_video_frames(state),state.A.out,state.OUTPUT_FPS
            )
            if _written_frame_count!=state._render_count:
                raise RuntimeError(
                    "video writer frame count mismatch: "
                    f"written={_written_frame_count} expected={state._render_count}"
                )
            print(f"[sim] wrote {state.A.out} ({_written_frame_count} frames)",flush=True)
        else:
            print("[sim] --no-render: skipped video/render (simulation+logs only)",flush=True)
    finally:
        # The same idempotent stack is also registered with atexit immediately
        # after environment creation, so pass-1 capture/spool failures take this
        # exact cleanup path instead of leaking EGL contexts until __del__.
        _cleanup_had_active_exception=sys.exc_info()[0] is not None
        _render_cleanup_failures=state._ROLLOUT_CLEANUP.run()
        atexit.unregister(state.at_exit_cleanup)
        if _render_cleanup_failures:
            _render_cleanup_text="; ".join(
                str(failure) for failure in _render_cleanup_failures
            )
            print(
                "[sim] rollout cleanup warning: " + _render_cleanup_text,
                file=sys.stderr,
                flush=True,
            )
            if not _cleanup_had_active_exception:
                raise RuntimeError("rollout cleanup failed: " + _render_cleanup_text)
    if state._SIM_LOG_ENABLED:
        _total_sec=time.time()-state._tloop
        summ["loop_sec"]=round(_total_sec,2); summ["sec_per_frame"]=round(_total_sec/max(state.N,1),4)
        summ["video_pipeline_sec"]=round(time.time()-_video_pipeline_started,2)
        write_summary_artifacts(state.LOGDIR,_summary_path,summ,state._RECS)
        print("[sim] SUMMARY",summ,flush=True)
        print(f"[sim] IK json -> {state.LOGDIR}/ik_results.json ; penetration json -> {state.LOGDIR}/penetration_results.json",flush=True)
        print(f"[sim] IK+penetration logs -> {state.LOGDIR}/sim_log.jsonl , sim_summary.json",flush=True)


__all__ = ["run"]
