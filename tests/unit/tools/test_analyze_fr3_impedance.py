# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from scripts.analyze_fr3_impedance import audit, joint_bound_metrics


def test_bounds_detect_single_fast_joint_and_soft_position_penetration():
    limits = dict(lower=np.full(7, -1.), upper=np.full(7, 1.), velocity=np.full(7, 2.62))
    q, dq = np.zeros((100, 7)), np.zeros((100, 7))
    q[50, 2], dq[50, 2] = -1.032, -4.607
    result = joint_bound_metrics(q, dq, limits)
    assert abs(result['position_overshoot_rad'] - .032) < 1e-12
    assert result['velocity_peak_rad_s'][2] == 4.607
    assert result['velocity_exceeding_arm_samples'] == 1


def test_success_with_active_grasp_is_separate_from_official_score(tmp_path):
    logs = tmp_path/'scene'/'run_00'/'logs'
    logs.mkdir(parents=True)
    (logs/'sim_summary.json').write_text(json.dumps(dict(frames=1, new_final_success=True)))
    (logs/'sim_log.jsonl').write_text(json.dumps(dict(frame=0, rule_grasp=dict(active=True)))+'\n')
    result = audit(tmp_path, 1)
    assert result['successes'] == 1
    row = result['rows'][0]
    assert row['success_while_still_held']
    assert row['quality_findings'] == ['official success while rule grasp remains active']
    assert not row['errors']
    # A self-consistent short summary must not pass a longer configured budget.
    (tmp_path/'resolved_rollout_config.json').write_text(json.dumps(dict(runner=dict(frames=2))))
    assert audit(tmp_path, 1)['rows'][0]['errors'] == ['frame count: 1, configured 2']


@pytest.mark.parametrize('corruption, expected', [
    ('cadence', 'cadence: 1'), ('command', 'command_limit: 1'),
    ('motor', 'motor_limit: 1'), ('nonfinite', 'nonfinite: 1'), ('lead', 'lead: 1'),
])
def test_audit_detects_corrupted_physics_samples_across_frame_boundaries(tmp_path, corruption, expected):
    logs = tmp_path/'scene'/'run_01'/'logs'
    logs.mkdir(parents=True)
    arm = dict(torque_nm=[1.4]*7, torque_raw_nm=[2.]*7,
               command_torque_without_gravity_nm=[1.]*7, gravity_torque_nm=[.4]*7,
               q=[0.]*7, dq=[0.]*7, position_error_m=[.03, 0, 0], rotation_error_rad=[0.]*3,
               saturated=True, motor_saturated=False, position_clamped=False,
               rotation_clamped=False, command_stale=False)
    frames = [dict(frame=i, fr3_impedance_microsteps=[
        dict(sim_time_s=(2*i+j)*.001, arms={'R': json.loads(json.dumps(arm))})
        for j in range(2)]) for i in range(2)]
    detail = dict(physics_dt_s=.001, ticks=4, saturated_arm_ticks=4, motor_saturated_arm_ticks=0,
                  position_clamped_arm_ticks=0, rotation_clamped_arm_ticks=0, stale_arm_ticks=0,
                  config=dict(tau_limit=[1.]*7, max_pose_offset_lin=.05),
                  torque_limits_nm={'R': [[-2., 2.]]*7})
    (logs/'sim_summary.json').write_text(json.dumps(dict(frames=2,
        arm_controller_detail=dict(cartesian_impedance=detail))))
    path = logs/'sim_log.jsonl'
    path.write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    assert not audit(tmp_path, 1, frame_hz=500)['rows'][0]['errors']
    tick = frames[1]['fr3_impedance_microsteps'][0]
    sample = tick['arms']['R']
    if corruption == 'cadence':
        tick['sim_time_s'] += .01
        frames[1]['fr3_impedance_microsteps'][1]['sim_time_s'] += .01
    elif corruption == 'command':
        sample['command_torque_without_gravity_nm'][0] = 1.1
    elif corruption == 'motor':
        sample['torque_nm'][0] = 2.1
    elif corruption == 'nonfinite':
        sample['dq'][0] = float('nan')
    else:
        sample['position_error_m'][0] = .051
    path.write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    assert expected in audit(tmp_path, 1, frame_hz=500)['rows'][0]['errors']


def test_audit_reads_summaries_with_and_without_retired_source_revisions(tmp_path):
    # Summaries written before #85 carry two source-revision fields that later
    # summaries omit. The audit uses neither, so both give the same row.
    arm = dict(torque_nm=[1.4]*7, torque_raw_nm=[2.]*7,
               command_torque_without_gravity_nm=[1.]*7, gravity_torque_nm=[.4]*7,
               q=[0.]*7, dq=[0.]*7, position_error_m=[.03, 0, 0], rotation_error_rad=[0.]*3,
               saturated=True, motor_saturated=False, position_clamped=False,
               rotation_clamped=False, command_stale=False)
    frames = [dict(frame=i, fr3_impedance_microsteps=[
        dict(sim_time_s=(2*i+j)*.001, arms={'R': arm}) for j in range(2)]) for i in range(2)]
    rows = {}
    for legacy in (False, True):
        detail = dict(physics_dt_s=.001, ticks=4, saturated_arm_ticks=4, motor_saturated_arm_ticks=0,
                      position_clamped_arm_ticks=0, rotation_clamped_arm_ticks=0, stale_arm_ticks=0,
                      config=dict(tau_limit=[1.]*7, max_pose_offset_lin=.05),
                      torque_limits_nm={'R': [[-2., 2.]]*7})
        if legacy:
            detail.update(deploy_commit='0'*40, controller_source_commit='1'*40)
        root = tmp_path/('legacy' if legacy else 'current')
        logs = root/'scene'/'run_00'/'logs'
        logs.mkdir(parents=True)
        (logs/'sim_summary.json').write_text(json.dumps(dict(frames=2,
            arm_controller_detail=dict(cartesian_impedance=detail))))
        (logs/'sim_log.jsonl').write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
        row = dict(audit(root, 1, frame_hz=500)['rows'][0])
        assert not row.pop('errors')
        row.pop('path')
        rows[legacy] = json.dumps(row, sort_keys=True, default=str)
    assert rows[False] == rows[True]
