# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Measure bounded FR3 wall contact, command timeout and commanded withdrawal."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from robosuite_env.robots.fr3.impedance import CartesianImpedanceHook, ImpedanceConfig


def probe(output, render=True):
    import mujoco

    assets = Path(__file__).resolve().parents[1] / "assets"
    root = ET.parse(assets / "fr3_robot.xml").getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(assets / mesh.get("file")))
    hand = root.find(".//body[@name='right_hand']")
    ET.SubElement(hand, "geom", name="contact_probe", type="sphere", size="0.012",
                  contype="4", conaffinity="2", rgba="0.9 0.3 0.1 1")
    ET.SubElement(root.find("worldbody"), "geom", name="wall", type="box",
                  size="0.015 0.10 0.10", pos="2 0 0", contype="2", conaffinity="4",
                  solref="0.005 1", rgba="0.3 0.5 0.8 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[:] = [0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398]
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    position = data.xpos[body].copy()
    model.geom_pos[wall_id] = position + [0.045, 0, 0]
    mujoco.mj_forward(model, data)
    state = SimpleNamespace(m=model, d=data, A=SimpleNamespace(fr3_nullspace="home"),
        ARM={"R": {"qi": np.arange(7), "dof": np.arange(7), "eef": body,
                    "policy": body, "policy_offset": np.zeros(3)}},
        env=SimpleNamespace(_pre_action=lambda action, policy_step: None))
    config = ImpedanceConfig(null_space_q_right=tuple(data.qpos))
    hook = CartesianImpedanceHook(state, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writer = renderer = None
    if render:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(output / "wall.mp4", fps=25)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = position
    camera.distance, camera.azimuth, camera.elevation = 1.2, 135, -25  # noqa: V101  # mujoco.MjvCamera field
    rows = []
    try:
        for tick in range(12000):
            t = float(data.time)
            phase = "press" if t < 4 else "stale_hold" if t < 7 else "withdraw" if t < 11 else "settle"
            if phase in {"press", "withdraw", "settle"}:
                speed = 0.02 if phase == "press" else -0.02 if phase == "withdraw" else 0.0
                hook.set_twist("R", [speed, 0, 0], [0, 0, 0])
            before_q, before_v = data.qpos.copy(), data.qvel.copy()
            hook(None)
            if not (np.array_equal(before_q, data.qpos) and np.array_equal(before_v, data.qvel)):
                raise RuntimeError("controller changed physical state")
            sample = hook.drain_samples()[-1]["arms"]["R"]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            contact_force = penetration = 0.0
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if wall_id not in (contact.geom1, contact.geom2):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                contact_force += max(0.0, float(force[0]))
                penetration = max(penetration, max(0.0, -float(contact.dist)))
            rows.append({"time_s": t, "phase": phase, "contact_force_n": contact_force,
                         "penetration_m": penetration, "tcp_x_m": float(data.xpos[body, 0]),
                         "reference_x_m": sample["reference_position"][0],
                         "lead_m": float(np.linalg.norm(sample["position_error_m"])),
                         "command_stale": sample["command_stale"]})
            if writer is not None and tick % 40 == 0:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    hold = [row for row in rows if 6 < row["time_s"] < 7]
    final = rows[-500:]
    summary = {
        "controller": hook.summary(), "ticks": len(rows),
        "probe": "12 mm sphere at URDF hand_tcp; isolated wall contact, no YUBI fingers",
        "peak_contact_force_n": max(row["contact_force_n"] for row in rows),
        "hold_mean_force_n": float(np.mean([row["contact_force_n"] for row in hold])),
        "max_penetration_mm": 1000 * max(row["penetration_m"] for row in rows),
        "max_lead_mm": 1000 * max(row["lead_m"] for row in rows),
        "final_mean_force_n": float(np.mean([row["contact_force_n"] for row in final])),
        "withdrawal_mm": 1000 * (np.mean([row["tcp_x_m"] for row in hold])
                                 - np.mean([row["tcp_x_m"] for row in final])),
        "automatic_force_retreat": False,
    }
    summary["passed"] = bool(
        summary["hold_mean_force_n"] > 10 and summary["hold_mean_force_n"] < 80
        and summary["max_lead_mm"] <= 50.00001 and summary["peak_contact_force_n"] < 100
        and summary["final_mean_force_n"] < 0.1 and summary["withdrawal_mm"] > 20
        and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    (output / "trace.json").write_text(json.dumps(rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    times = [row["time_s"] for row in rows]
    axes[0].plot(times, [row["contact_force_n"] for row in rows])
    axes[0].set_ylabel("Wall force (N)")
    for field, label in [("tcp_x_m", "TCP"), ("reference_x_m", "Equilibrium")]:
        axes[1].plot(times, [1000 * (row[field] - position[0]) for row in rows], label=label)
    axes[1].set_ylabel("x displacement (mm)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].legend()
    for axis in axes:
        for boundary in (4, 7, 11):
            axis.axvline(boundary, color="gray", linestyle="--", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output / "wall.png", dpi=150)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    summary = probe(args.output, render=not args.no_video)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
