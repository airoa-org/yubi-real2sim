# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Relative-hand state quaternion hemisphere: encoding, resolution, cross-check.

Recorded inputs are the frame-0 execution anchors and the rel_pose logged by a
ten-rollout Duo run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite_env.policy import policy_io as pio

# label -> (L position, L gripper quat wxyz, R position, R gripper quat wxyz, logged rel_pose)
RECORDED_OBS0 = {
    "cup": (
        [0.2699467507634215, -0.4737008610848523, 0.23468703240654046],
        [0.3581776889011723, 0.8260130005045072, -0.3713612340270935, -0.22694074123618588],
        [-0.27484776806058964, -0.4625862361760874, 0.2247332219507028],
        [0.38877410102852994, 0.8085550069463917, 0.3673036509268633, 0.24531923513490417],
        [0.3508510887622833, -0.3459559679031372, 0.23289841413497925, -0.4275670051574707,
         -0.023780206218361855, 0.6620665788650513, 0.6150518655776978],
    ),
    "second_positive_anchor": (
        [0.2077564711228707, -0.739434502267965, 0.2401053365575784],
        [0.36278837528675234, 0.8414077154482096, -0.3601573728450476, -0.17522647609986186],
        [-0.25032840731675915, -0.7554702521986901, 0.23363967194977492],
        [0.3966188803324321, 0.8235662440715842, 0.3618624579537233, 0.18299635765052874],
        [0.2669173777103424, -0.29914042353630066, 0.2222854495048523, -0.4652002453804016,
         -0.03243832290172577, 0.5724098682403564, 0.6744504570960999],
    ),
    "negative_anchor": (
        [0.2077560646224257, -0.7394337606150798, 0.24010515543669067],
        [0.36278867712615076, 0.841406425427895, -0.3601603929017196, -0.17522583822766424],
        [-0.2503285399771395, -0.7554704270667751, 0.23363968498659243],
        [0.3966189821477245, 0.823566244817486, 0.3618625180642354, 0.18299601475862176],
        [0.2669179141521454, -0.29913774132728577, 0.2222878783941269, -0.46520259976387024,
         -0.03243699669837952, 0.5724101662635803, 0.6744486093521118],
    ),
}
POSITIVE_ANCHORS = ("cup", "second_positive_anchor")


@pytest.fixture(scope="module")
def q_fc():
    """The worker's robot-to-canonical quaternion (state._Q_FC for FR3/duo)."""

    from robosuite_env.rollout.setup.frames import build_frame_constants

    return build_frame_constants(SimpleNamespace(A=SimpleNamespace(world_offset=[0, 0, 0])))[6]


def _legacy_relative_hand_state(pL, qL, pR, qR, q_fc):
    """Frozen copy of relative_hand_state before the hemisphere mode existed."""

    qc_l = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(qL, q_fc))
    qc_r = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(qR, q_fc))
    dp = pio.quat_wxyz_to_matrix(qc_l).T @ (np.asarray(pR, float) - np.asarray(pL, float))
    qr = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(pio.quat_conj_wxyz(qc_l), qc_r))
    if qr[0] < 0.0:
        qr = -qr
    return np.concatenate([dp, pio.wxyz_to_xyzw(qr)]).astype(np.float32)


def _rel_rotation_about_z(theta_deg):
    """Left/right gripper quaternions whose relative rotation is theta about canonical z."""

    half = np.radians(theta_deg) / 2.0
    q_rel = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    q_left_canon = pio.normalize_quat_wxyz(np.array([0.9, 0.1, -0.2, 0.3]))
    q_right_canon = pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q_left_canon, q_rel))
    q_cf = pio.Q_CANON_TO_FRANKA_WXYZ
    return (
        pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q_left_canon, q_cf)),
        pio.normalize_quat_wxyz(pio.quat_mul_wxyz(q_right_canon, q_cf)),
    )


# ------------------------------------------------------------------ encoding


@pytest.mark.parametrize("label", POSITIVE_ANCHORS)
def test_positive_mode_is_byte_identical_on_recorded_obs0(label, q_fc):
    pL, qL, pR, qR, logged = RECORDED_OBS0[label]
    legacy = _legacy_relative_hand_state(pL, qL, pR, qR, q_fc)
    default = pio.relative_hand_state(pL, qL, pR, qR, q_franka_to_canon_wxyz=q_fc)
    explicit = pio.relative_hand_state(
        pL, qL, pR, qR, q_franka_to_canon_wxyz=q_fc, quat_hemisphere="positive"
    )
    encoded = pio.PolicyStateQuatEncoder("positive", q_franka_to_canon_wxyz=q_fc)(pL, qL, pR, qR)
    for got in (default, explicit, encoded):
        assert got.dtype == np.float32
        assert got.tobytes() == legacy.tobytes()
        assert got.tobytes() == np.asarray(logged, np.float32).tobytes()


@pytest.mark.parametrize("theta", [179.8, 179.99, 180.0, 180.01, 180.2])
def test_positive_mode_keeps_the_fixed_rule_near_qw_zero(theta):
    qL, qR = _rel_rotation_about_z(theta)
    p = [0.0, 0.0, 0.0]
    encoder = pio.PolicyStateQuatEncoder("positive")
    for _ in range(3):  # repeated calls carry no continuity in positive mode
        got = encoder(p, qL, p, qR)
        assert got.tobytes() == _legacy_relative_hand_state(
            p, qL, p, qR, pio.quat_conj_wxyz(pio.Q_CANON_TO_FRANKA_WXYZ)
        ).tobytes()


def test_negative_mode_picks_nonpositive_qw_at_the_first_observation(q_fc):
    pL, qL, pR, qR, logged = RECORDED_OBS0["negative_anchor"]
    assert logged[pio.POLICY_STATE_REL_QW_INDEX] > 0.0  # what the fixed rule sent
    got = pio.PolicyStateQuatEncoder("negative", q_franka_to_canon_wxyz=q_fc)(pL, qL, pR, qR)
    assert got[pio.POLICY_STATE_REL_QW_INDEX] <= 0.0
    np.testing.assert_array_equal(got[:3], np.asarray(logged, np.float32)[:3])
    np.testing.assert_array_equal(got[3:7], -np.asarray(logged, np.float32)[3:7])


def _sign_jumps(quats):
    return sum(float(np.dot(a, b)) < 0.0 for a, b in zip(quats[:-1], quats[1:]))


def test_negative_mode_never_flips_through_a_qw_zero_crossing():
    p = [0.0, 0.0, 0.0]
    thetas = np.linspace(150.0, 210.0, 25)  # relative qw crosses zero at 180 deg
    encoder = pio.PolicyStateQuatEncoder("negative")
    sent = [encoder(p, *_rel_rotation_about_z(t)[:1], p, _rel_rotation_about_z(t)[1]) for t in thetas]
    quats = [np.asarray(s[3:7], float) for s in sent]
    assert sent[0][pio.POLICY_STATE_REL_QW_INDEX] <= 0.0
    assert any(s[pio.POLICY_STATE_REL_QW_INDEX] > 0.0 for s in sent)  # the crossing happened
    assert _sign_jumps(quats) == 0
    assert encoder.summary()["sign_jumps_between_consecutive_observations"] == 0
    fixed = [pio.relative_hand_state(p, *_rel_rotation_about_z(t)[:1], p, _rel_rotation_about_z(t)[1])[3:7] for t in thetas]
    assert _sign_jumps([np.asarray(q, float) for q in fixed]) == 1  # the raw sign jump the encoder removes


def test_negative_mode_continuity_resets_per_episode():
    p = [0.0, 0.0, 0.0]
    late = _rel_rotation_about_z(210.0)  # raw qw < 0, but continuity from 150 deg sends qw > 0
    encoder = pio.PolicyStateQuatEncoder("negative")
    for t in np.linspace(150.0, 210.0, 25):
        carried = encoder(p, _rel_rotation_about_z(t)[0], p, _rel_rotation_about_z(t)[1])
    assert carried[pio.POLICY_STATE_REL_QW_INDEX] > 0.0
    encoder.reset()
    assert encoder(p, late[0], p, late[1])[pio.POLICY_STATE_REL_QW_INDEX] <= 0.0
    fresh = pio.PolicyStateQuatEncoder("negative")  # a new rollout
    assert fresh(p, late[0], p, late[1])[pio.POLICY_STATE_REL_QW_INDEX] <= 0.0


def test_unknown_hemisphere_and_positive_continuity_are_refused():
    with pytest.raises(ValueError):
        pio.PolicyStateQuatEncoder("auto")
    with pytest.raises(ValueError):
        pio.orient_policy_state_quat_wxyz(np.array([1.0, 0, 0, 0]), "positive", np.array([0, 0, 0, 1.0]))


def test_qw_index_matches_the_yubi_data_config():
    yubi_policy = pytest.importorskip("robosuite_env.policy.yubi_openpi.yubi_policy")
    rel = np.arange(7, dtype=np.float32) + 10.0
    obs = {
        "observation.image.left": np.zeros((224, 224, 3), np.uint8),
        "observation.image.right": np.zeros((224, 224, 3), np.uint8),
        pio.POLICY_STATE_REL_POSE_KEY: rel,
        "observation.joint_states": np.array([20.0, 21.0], np.float32),
        "prompt": "x",
    }
    state = yubi_policy.YubiInputs()(obs)["state"]
    assert state[pio.POLICY_STATE_REL_QW_INDEX] == rel[6]
    q = pio.relative_hand_state([0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0.5, 0.5, 0.5])
    assert q[pio.POLICY_STATE_REL_QW_INDEX] == pytest.approx(1.0)  # identity: w is last


# ------------------------------------------------------------------ resolution


def _write_norm_stats(checkpoint: Path, qw_q01: float, qw_q99: float, *, asset="dataset", extra_dim=False):
    q01 = [0.0] * 32
    q99 = [0.0] * 32
    for i in range(9):
        q01[i], q99[i] = -0.5, 0.5
    q01[6], q99[6] = qw_q01, qw_q99
    if extra_dim:
        q01[12], q99[12] = -1.0, 1.0
    path = checkpoint / "assets" / asset / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"norm_stats": {"state": {"q01": q01, "q99": q99}, "actions": {}}}))
    return path


@pytest.mark.parametrize(
    "qw_q01,qw_q99,classification,hemisphere",
    [
        (-0.986, -0.150, "negative", "negative"),
        (0.499, 0.965, "positive", "positive"),
        (-0.988, 0.970, "mixed", "positive"),
        (-0.5, 0.0, "mixed", "positive"),
    ],
)
def test_auto_resolution_from_norm_stats(tmp_path, qw_q01, qw_q99, classification, hemisphere):
    path = _write_norm_stats(tmp_path, qw_q01, qw_q99, asset="yubi/nested/asset")
    got = pio.policy_state_quat_hemisphere_from_norm_stats(path)
    assert (got["classification"], got["hemisphere"]) == (classification, hemisphere)
    assert got["norm_stats_path"] == str(path)
    assert got["qw_index"] == 6 and got["qw_q01"] == qw_q01 and got["qw_q99"] == qw_q99
    assert len(got["norm_stats_sha256"]) == 64


def _rewrite(path, text):
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "setup,match",
    [
        (lambda c: _rewrite(_write_norm_stats(c, -1, -0.1), "{not json"), "not valid JSON"),
        (lambda c: _rewrite(_write_norm_stats(c, -1, -0.1), '{"norm_stats": {}}'), "no norm_stats.state"),
        (lambda c: _write_norm_stats(c, -1, -0.1, extra_dim=True), "qw index is ambiguous"),
        (lambda c: _write_norm_stats(c, -3.0, -0.1), "not a unit-quaternion component"),
    ],
)
def test_auto_resolution_fails_closed(tmp_path, setup, match):
    path = setup(tmp_path)
    with pytest.raises(pio.PolicyStateQuatHemisphereError, match=match):
        pio.policy_state_quat_hemisphere_from_norm_stats(path)


def test_launcher_renders_the_flag_only_for_negative(tmp_path):
    from real2sim.config._types import ConfigError
    from real2sim.config.parameters import (
        POLICY_STATE_QUAT_HEMISPHERE_FLAG as FLAG,
        resolve_policy_state_quat_hemisphere_parameter as resolve,
    )

    base = ["--physics", "dynamic"]
    config_asset = "pi05_cup"
    resolver_args = {
        "config_name": "pi05_cup",
        "config_source": "deploy",
        "action_horizon": 32,
        "repo_root": Path(__file__).resolve().parents[3],
    }
    neg, pos, mixed = tmp_path / "neg", tmp_path / "pos", tmp_path / "mixed"
    _write_norm_stats(neg, -0.986, -0.150, asset=config_asset)
    _write_norm_stats(pos, 0.499, 0.965, asset=config_asset)
    _write_norm_stats(mixed, -0.988, 0.970, asset=config_asset)
    args, record = resolve(
        {"policy_state_quat_hemisphere": "auto"},
        base,
        checkpoint=neg,
        **resolver_args,
    )
    assert args == base + [FLAG, "negative"] and record["source"] == "checkpoint_norm_stats"
    for checkpoint in (pos, mixed):
        args, record = resolve(
            {"policy_state_quat_hemisphere": "auto"},
            base,
            checkpoint=checkpoint,
            **resolver_args,
        )
        assert args == base and record["hemisphere"] == "positive"
    args, record = resolve(
        {"policy_state_quat_hemisphere": "negative"},
        base,
        checkpoint=tmp_path / "none",
        **resolver_args,
    )
    assert args == base + [FLAG, "negative"] and record["source"] == "profile"
    # Absent stats (placeholder checkpoint): unresolved, no flag, no exception.
    args, record = resolve(
        {"policy_state_quat_hemisphere": "auto"},
        base,
        checkpoint=tmp_path / "none",
        **resolver_args,
    )
    assert args == base and record["source"] == "unresolved" and record["hemisphere"] is None
    # Present but broken stats still fail closed at resolution.
    broken = tmp_path / "broken"
    _write_norm_stats(
        broken, -0.986, -0.150, asset=config_asset
    ).write_text("{not json")
    with pytest.raises(ConfigError, match="auto cannot be resolved"):
        resolve(
            {"policy_state_quat_hemisphere": "auto"},
            base,
            checkpoint=broken,
            **resolver_args,
        )
    with pytest.raises(ConfigError, match="rendered by the launcher"):
        resolve(
            {"policy_state_quat_hemisphere": "auto"},
            base + [FLAG, "negative"],
            checkpoint=neg,
            **resolver_args,
        )


# ------------------------------------------------------------------ worker cross-check


def _binding_state(requested, metadata, *, batch=True, pose_source="sim"):
    return SimpleNamespace(
        A=SimpleNamespace(
            policy_state_quat_hemisphere=requested,
            policy_batch_server="127.0.0.1:1" if batch else "",
            policy_pose_source=pose_source,
        ),
        policy=SimpleNamespace(metadata=metadata),
    )


def _published(hemisphere):
    resolution = {
        "hemisphere": hemisphere, "classification": hemisphere, "rule": "r",
        "norm_stats_path": "/ckpt/assets/a/norm_stats.json", "norm_stats_sha256": "f" * 64,
        "qw_index": 6, "qw_q01": -0.9, "qw_q99": -0.1,
    }
    return {"policy_state_quat_hemisphere": hemisphere, "policy_state_quat_hemisphere_resolution": resolution}


def test_worker_refuses_a_hemisphere_that_disagrees_with_the_server():
    from robosuite_env.rollout.observe.policy_client import _PolicyBinding

    with pytest.raises(RuntimeError, match="hemisphere mismatch"):
        _PolicyBinding(_binding_state("positive", _published("negative")), None)
    with pytest.raises(RuntimeError, match="hemisphere mismatch"):
        _PolicyBinding(_binding_state("negative", _published("positive")), None)


def test_worker_refuses_a_server_that_publishes_no_hemisphere():
    from robosuite_env.rollout.observe.policy_client import _PolicyBinding

    with pytest.raises(RuntimeError, match="did not publish"):
        _PolicyBinding(_binding_state("positive", {"norm_stats_sha256": "x"}), None)


def test_worker_records_an_agreeing_server_and_encodes_with_it():
    from robosuite_env.rollout.observe import policy_client

    state = _binding_state("negative", _published("negative"))
    binding = policy_client._PolicyBinding(state, None)
    assert state.POLICY_STATE_QUAT_HEMISPHERE["verified"] is True
    assert state.POLICY_STATE_QUAT_HEMISPHERE["norm_stats_sha256"] == "f" * 64
    assert state.POLICY_STATE_QUAT_ENCODER.quat_hemisphere == "negative"
    assert binding._state_quat is state.POLICY_STATE_QUAT_ENCODER


def test_worker_grants_no_unverified_exemption_by_transport_name():
    # No transport name grants an exemption, including the removed portable
    # WebSocket bridge.
    # No retained server sets ``policy_transport``; metadata that still names it
    # is refused like any other server that publishes no hemisphere.
    from robosuite_env.rollout.observe.policy_client import _PolicyBinding

    for hemisphere in ("positive", "negative"):
        state = _binding_state(hemisphere, {"policy_transport": "portable_websocket"})
        with pytest.raises(RuntimeError, match="did not publish"):
            _PolicyBinding(state, None)
        assert not hasattr(state, "POLICY_STATE_QUAT_ENCODER")
