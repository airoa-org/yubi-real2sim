#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Audit FR3 logs without treating incomplete rollouts as policy failures.

The default ``--joint-limits`` file, ``third_party/franka_description/robots/fr3/joint_limits.yaml``,
exists after the bootstrap step clones franka_description; pass another path otherwise.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def read_joint_limits(path):
    limits = yaml.safe_load(Path(path).read_text())
    return {name: np.array([limits[f'joint{i}']['limit'][name] for i in range(1, 8)])
            for name in ('lower', 'upper', 'velocity')}


def joint_bound_metrics(q, dq, limits):
    q, dq = np.asarray(q), np.asarray(dq)
    return dict(position_overshoot_rad=float(max(0., np.max(limits['lower']-q), np.max(q-limits['upper']))),
                velocity_peak_rad_s=np.max(np.abs(dq), axis=0),
                velocity_exceeding_arm_samples=int(np.sum(np.any(np.abs(dq)>limits['velocity']+1e-8, axis=1))))


def audit(root, expected, frame_hz=10., joint_limits=None):
    rows = []
    config_path = root / 'resolved_rollout_config.json'
    expected_frames = (json.loads(config_path.read_text())['runner']['frames']
                       if config_path.exists() else None)
    for path in sorted(root.glob('*/run_*/logs/sim_summary.json')):
        s = json.loads(path.read_text())
        detail = s.get('arm_controller_detail', {}).get('cartesian_impedance')
        dt = detail['physics_dt_s'] if detail else None
        previous_time = None
        reference_errors, rotation_errors, velocities, changes = [], [], [], []
        counters = dict(cadence=0, nonfinite=0, motor_limit=0, command_limit=0, gravity_sum=0, lead=0)
        nframes = nticks = narms = 0
        saturation = motor_saturation = pos_clamp = rot_clamp = stale = 0
        previous_torque = {}
        frame_ids, score_history = [], []
        bounds = dict(position_overshoot_rad=0., velocity_peak_rad_s=np.zeros(7),
                      velocity_exceeding_arm_samples=0, arm_samples=0)
        final_grasp_active = False
        for line in path.with_name('sim_log.jsonl').open():
            record = json.loads(line)
            nframes += 1
            frame_ids.append(record['frame'])
            score_history.append(record.get('success', {}))
            final_grasp_active = bool(record.get('rule_grasp', {}).get('active', False))
            if joint_limits is not None:
                ticks = record.get('fr3_impedance_microsteps', [])
                if ticks:
                    arms = [arm for tick in ticks for arm in tick['arms'].values()]
                    q, dq = [a['q'] for a in arms], [a['dq'] for a in arms]
                else:
                    arms = list(record['arms'].values())
                    q, dq = [a['q'] for a in arms], [a['joint_velocity_rad_s'] for a in arms]
                current = joint_bound_metrics(q, dq, joint_limits)
                bounds['position_overshoot_rad'] = max(bounds['position_overshoot_rad'], current['position_overshoot_rad'])
                bounds['velocity_peak_rad_s'] = np.maximum(bounds['velocity_peak_rad_s'], current['velocity_peak_rad_s'])
                bounds['velocity_exceeding_arm_samples'] += current['velocity_exceeding_arm_samples']
                bounds['arm_samples'] += len(arms)
            ticks = record.get('fr3_impedance_microsteps', [])
            if not ticks:
                continue
            times = np.asarray([tick['sim_time_s'] for tick in ticks])
            differences = np.diff(times if previous_time is None else np.r_[previous_time, times])
            counters['cadence'] += int(np.count_nonzero(np.abs(differences - dt) > 1e-8))
            previous_time = float(times[-1])
            nticks += len(ticks)
            entries = [(side, arm) for tick in ticks for side, arm in tick['arms'].items()]
            if not entries:
                continue
            sides = np.asarray([side for side, arm in entries])
            arms = [arm for side, arm in entries]
            def values(name):
                return np.asarray([arm[name] for arm in arms])
            narms += len(arms)
            tau, raw = values('torque_nm'), values('torque_raw_nm')
            command, gravity = values('command_torque_without_gravity_nm'), values('gravity_torque_nm')
            numeric = np.concatenate((tau, raw, command, gravity, values('q'), values('dq'),
                                      values('position_error_m'), values('rotation_error_rad')), axis=1)
            counters['nonfinite'] += int(np.count_nonzero(~np.isfinite(numeric).all(axis=1)))
            limits = np.asarray([detail['torque_limits_nm'][side] for side in sides])
            source_limits = np.asarray(detail['config']['tau_limit'])
            counters['motor_limit'] += int(np.count_nonzero(np.any(
                (tau < limits[:,:,0]-1e-9) | (tau > limits[:,:,1]+1e-9), axis=1)))
            counters['command_limit'] += int(np.count_nonzero(~np.isclose(
                command, np.clip(raw, -source_limits, source_limits), atol=1e-9).all(axis=1)))
            counters['gravity_sum'] += int(np.count_nonzero(~np.isclose(
                tau, np.clip(command+gravity, limits[:,:,0], limits[:,:,1]), atol=1e-9).all(axis=1)))
            error = np.linalg.norm(values('position_error_m'), axis=1)
            counters['lead'] += int(np.count_nonzero(error > detail['config']['max_pose_offset_lin'] + 1e-9))
            for side in set(sides):
                commands = command[sides == side]
                if side in previous_torque:
                    commands = np.vstack((previous_torque[side], commands))
                changes.extend((np.abs(np.diff(commands, axis=0)).max(axis=1)/dt).tolist())
                previous_torque[side] = commands[-1]
            reference_errors.extend((error * 1000).tolist())
            rotation_errors.extend(np.rad2deg(np.linalg.norm(values('rotation_error_rad'), axis=1)).tolist())
            velocities.extend(np.abs(values('dq')).max(axis=1).tolist())
            saturation += int(values('saturated').sum())
            motor_saturation += int(values('motor_saturated').sum())
            pos_clamp += int(values('position_clamped').sum())
            rot_clamp += int(values('rotation_clamped').sum())
            stale += int(values('command_stale').sum())
        errors = []
        if frame_ids != list(range(s['frames'])):
            errors.append('frame sequence/count differs from summary')
        if expected_frames is not None and nframes != expected_frames:
            errors.append(f'frame count: {nframes}, configured {expected_frames}')
        if detail:
            if nticks != detail['ticks']:
                errors.append('logged microstep count differs from summary')
            expected_ticks = round(nframes / frame_hz / dt)
            if nticks != expected_ticks or not nticks:
                errors.append(f'physics tick count: {nticks}, expected {expected_ticks}')
            for name, count in counters.items():
                if count:
                    errors.append(f'{name}: {count}')
            for name, count in (('saturated_arm_ticks', saturation), ('motor_saturated_arm_ticks', motor_saturation),
                                ('position_clamped_arm_ticks', pos_clamp), ('rotation_clamped_arm_ticks', rot_clamp),
                                ('stale_arm_ticks', stale)):
                if count != detail[name]:
                    errors.append(f'{name}: log/summary mismatch')
        def percentile(values, q):
            return float(np.percentile(values, q)) if values else None
        quality_findings = []
        if bounds['velocity_exceeding_arm_samples']:
            quality_findings.append('static FR3 joint velocity limit exceeded')
        if bounds['position_overshoot_rad'] > 1e-6:
            quality_findings.append('FR3 joint position range exceeded (MuJoCo limits are soft)')
        held_success = bool(s.get('new_final_success') and final_grasp_active)
        if held_success:
            quality_findings.append('official success while rule grasp remains active')
        bounds['velocity_peak_rad_s'] = bounds['velocity_peak_rad_s'].tolist()
        bounds['sampling_hz'] = 1/dt if detail else frame_hz
        bounds['static_limits'] = {key: value.tolist() for key, value in joint_limits.items()} if joint_limits else None
        rows.append(dict(path=str(path), controller=s.get('arm_controller'), frames=nframes,
            success=s.get('new_final_success'), ever_success=s.get('success_ever'),
            max_pen_mm=s.get('max_pen_mm'), ik_fail=s.get('ik_fail'), impedance_ticks=nticks,
            reference_error_p95_mm=percentile(reference_errors,95),
            orientation_error_p95_deg=percentile(rotation_errors,95),
            actual_joint_velocity_max_rad_s=max(velocities,default=None),
            command_torque_rate_max_nm_s=max(changes,default=None),
            saturated_fraction=saturation/narms if narms else None,
            motor_saturated_fraction=motor_saturation/narms if narms else None,
            position_clamped_fraction=pos_clamp/narms if narms else None,
            rotation_clamped_fraction=rot_clamp/narms if narms else None,
            stale_fraction=stale/narms if narms else None,
            success_final=s.get('success_final'), errors=errors,
            joint_bounds=bounds if joint_limits else None,
            success_while_still_held=held_success, quality_findings=quality_findings))
    return dict(root=str(root), expected=expected, expected_frames=expected_frames,
                completed=len(rows), complete=len(rows)==expected,
                successes=sum(row['success'] is True for row in rows), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--expected', type=int, default=10)
    p.add_argument('--frame-hz', type=float, default=10.)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--joint-limits', type=Path, default=Path(__file__).resolve().parents[1]
                   / 'third_party/franka_description/robots/fr3/joint_limits.yaml')
    p.add_argument('--fail-on-quality-findings', action='store_true',
                   help='Return 2 for physical/scoring findings; structural failures always return 1.')
    a = p.parse_args()
    limits = read_joint_limits(a.joint_limits)
    report = [audit(root, a.expected, a.frame_hz, limits) for root in a.roots]
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    for r in report:
        print(r['root'], f"complete={r['completed']}/{r['expected']} success={r['successes']}/{r['completed']}")
        for row in r['rows']:
            print(Path(row['path']).parents[1].name, 'success=', row['success'], 'ticks=', row['impedance_ticks'],
                  'p95_ref_mm=', row['reference_error_p95_mm'], 'errors=', row['errors'],
                  'quality_findings=', row['quality_findings'])
    if any(not r['complete'] or any(row['errors'] for row in r['rows']) for r in report):
        return 1
    if a.fail_on_quality_findings and any(row['quality_findings'] for r in report for row in r['rows']):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
