# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""A claw-machine bench for the rule grasp: descend, close, lift, hold.

The unit tests around the rule grasp write joint angles by hand and call the
functions directly. That is enough to pin the bookkeeping and nothing else:
they never run the solver, so they cannot say whether the ratchet drives
penetration, whether the weld holds the object where the jaws are, or whether a
pen caught across the jaws is squeezed straight. This bench answers those
questions.

So: one real YUBI gripper on a vertical slide, one real object with a free
joint, a floor, and the four moves of a claw machine.

    descend   the open hand comes down over the object
    close     the jaws drive shut; the rule grasp latches somewhere in here
    lift      the hand rises
    hold      it stops, and we see whether the object is still there

Everything a comparison needs is recorded per step: jaw angles, contact
count, peak normal force, deepest penetration, object height and tilt, and
which phase the grasp was in. Runs on different trees can be diffed directly.

    python tests/tools/ufo_catcher.py --object pen --out /tmp/pen
    python tests/tools/ufo_catcher.py --object pen --out /tmp/pen --render

The object is placed lying across the jaws on purpose -- the tilt column is
the point of the exercise, not a detail.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "robosuite_env" / "assets"

#: Objects the tasks actually pick up, and how they are laid out. ``tilt_deg``
#: is about the hand's approach axis: a pen dropped at 35 degrees is a pen the
#: jaws have to roll upright, which is the behaviour under test.
#: ``yaw_deg`` is how far the object's long axis is turned away from square to
#: the jaws. The jaws close along world x, so a yaw of zero is an object lying
#: along y with its narrow side presented -- the easy case. Anything else is
#: the case under test: can the closing jaws roll it square?
#: Which way up an object lies is chosen by measurement, not written here: of
#: the three axis-aligned orientations, the one whose world height is smallest
#: is the one it is lying down in.
#:
#: Resting height and how far the hand comes down are measured too. ``yaw_deg``
#: is the one placement choice left, and it is the point of the exercise: how
#: far off square to the jaws the object starts.
OBJECTS = {
    "pen": {"xml": "pen.xml", "yaw_deg": 35.0},
    "tape": {"xml": "tape.xml", "yaw_deg": 25.0},
    "lan": {"xml": "lan.xml", "yaw_deg": 30.0},
}

#: Quaternions that bring each local axis onto world z, plus the identity.
_LAY_QUATS = {
    "z": np.array([1.0, 0.0, 0.0, 0.0]),
    "x": np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]),
    "y": np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0]),
}

#: The claw machine's script, in seconds.
DESCEND_SEC = 0.6
CLOSE_SEC = 1.2
LIFT_SEC = 1.0
HOLD_SEC = 1.2


def _flattest_axis(object_xml):
    """Which local axis to stand upright so the object lies as flat as it can.

    Reads the collision meshes out of the object's own model and compares the
    three axis-aligned orientations. The smallest resulting height is the one
    the object is lying down in.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(object_xml))
    verts = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) != 0:
            continue
        mesh = int(model.geom_dataid[g])
        if mesh < 0:
            continue
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        verts.append(np.array(model.mesh_vert[start:start + count], dtype=float))
    if not verts:
        return "z"
    extent = np.vstack(verts).ptp(axis=0)
    # Standing local axis i upright makes the world height that axis' extent.
    return ("x", "y", "z")[int(np.argmin(extent))]


def _object_world_z(model, data, state):
    """The object's lowest and highest points right now, in world z."""

    low, high = float("inf"), float("-inf")
    for g in state.GRASPABLE_GEOMS:
        mesh = int(model.geom_dataid[g])
        if mesh < 0:
            continue
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        verts = np.array(model.mesh_vert[start:start + count], dtype=float)
        mat = np.array(data.geom_xmat[g], dtype=float).reshape(3, 3)
        world = verts @ mat.T + np.array(data.geom_xpos[g], dtype=float)
        low = min(low, float(world[:, 2].min()))
        high = max(high, float(world[:, 2].max()))
    return low, high


def _pad_world_z(model, data, state):
    """The vertical span of the inner pads -- the part that does the gripping."""
    pads = (state.INNER_PAD_SIDE_GEOMS["R"]["left"]
            | state.INNER_PAD_SIDE_GEOMS["R"]["right"])
    low, high = float("inf"), float("-inf")
    for g in pads:
        mesh = int(model.geom_dataid[g])
        if mesh < 0:
            continue
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        verts = np.array(model.mesh_vert[start:start + count], dtype=float)
        mat = np.array(data.geom_xmat[g], dtype=float).reshape(3, 3)
        world = verts @ mat.T + np.array(data.geom_xpos[g], dtype=float)
        low = min(low, float(world[:, 2].min()))
        high = max(high, float(world[:, 2].max()))
    return low, high


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _merge(into: ET.Element, other: ET.Element, tag: str) -> None:
    """Copy one section of another model in, creating it if absent."""
    source = other.find(tag)
    if source is None:
        return
    target = into.find(tag)
    if target is None:
        target = ET.SubElement(into, tag)
    for child in source:
        target.append(copy.deepcopy(child))


def build_model_xml(object_name: str) -> str:
    """A floor, a YUBI right gripper on a slide, and one object on a free joint."""
    spec = OBJECTS[object_name]
    gripper = ET.parse(ASSETS / "yubi_right.xml").getroot()
    obj = ET.parse(ASSETS / spec["xml"]).getroot()

    root = ET.Element("mujoco", {"model": f"ufo_catcher_{object_name}"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.002", "integrator": "implicitfast", "gravity": "0 0 -9.81"},
    )
    for section in ("asset", "equality", "sensor"):
        _merge(root, gripper, section)
    _merge(root, obj, "asset")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "640", "offheight": "480"})

    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        {"name": "grid", "type": "2d", "builtin": "checker",
         "rgb1": ".2 .25 .3", "rgb2": ".3 .35 .4", "width": "256", "height": "256"},
    )
    ET.SubElement(
        asset, "material",
        {"name": "grid_mat", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.05"},
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0.3 -0.3 0.8", "dir": "-0.3 0.3 -0.8",
                                   "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "1 1 0.05",
                                  "material": "grid_mat", "condim": "3",
                                  "friction": "1 0.05 0.0001"})
    ET.SubElement(world, "camera", {"name": "side", "pos": "0.42 -0.30 0.16",
                                    "xyaxes": "0.6 0.8 0 -0.25 0.19 0.95"})

    # The gripper hangs from a vertical slide -- the claw machine's one axis.
    mount = ET.SubElement(world, "body", {"name": "mount", "pos": "0 0 0.30"})
    ET.SubElement(mount, "joint", {"name": "mount_z", "type": "slide", "axis": "0 0 1",
                                   "range": "-0.30 0.10", "damping": "60"})
    ET.SubElement(mount, "geom", {"name": "mount_geom", "type": "box",
                                  "size": "0.02 0.02 0.01", "contype": "0",
                                  "conaffinity": "0", "rgba": "0.25 0.3 0.35 1",
                                  "mass": "0.5"})
    gripper_body = gripper.find("worldbody").find("body")
    placed = copy.deepcopy(gripper_body)
    # Point the jaws down at the floor.
    placed.set("pos", "0 0 0")
    placed.set("quat", "0 1 0 0")
    mount.append(placed)

    yaw = math.radians(spec["yaw_deg"])
    lay = _LAY_QUATS[_flattest_axis(ASSETS / spec["xml"])]
    turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    w = _quat_mul(turn, lay)
    quat = " ".join(f"{v:.9f}" for v in w)
    holder = ET.SubElement(
        world, "body",
        # Lifted onto the floor in `run` once the compiled geometry says how
        # tall it is in this orientation.
        {"name": "target", "pos": "0 0 0.20", "quat": quat},
    )
    ET.SubElement(holder, "freejoint", {"name": "target_free"})
    for child in obj.find("worldbody").find("body"):
        if child.tag == "body":
            for geom in child:
                holder.append(copy.deepcopy(geom))

    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(actuator, "position", {"name": "a_mount", "joint": "mount_z",
                                         "kp": "4000", "ctrlrange": "-0.30 0.10"})
    for side in ("left", "right"):
        ET.SubElement(
            actuator, "position",
            {"name": f"a_finger_{side}", "joint": f"finger_joint_{side}",
             "kp": "12", "ctrlrange": "0 0.7854"},
        )
    # The same two inactive welds the tasks declare, so the rule grasp has
    # something to retarget. MuJoCo fixes the equality count at compile time.
    from robosuite_env.tasks.equalities import add_rule_grasp_weld_equalities

    add_rule_grasp_weld_equalities(root, placeholder_bodies=("mount", "target"))
    return ET.tostring(root, encoding="unicode")


def make_state(model, data, args_overrides=None):
    """The slice of rollout state the rule grasp reads, built for this model."""
    import mujoco
    from types import SimpleNamespace

    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    finger_joints = [jid("finger_joint_left"), jid("finger_joint_right")]
    pads = {"left": set(), "right": set()}
    fingers = {"left": set(), "right": set()}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for side in ("left", "right"):
            if name.startswith(f"finger_{side}"):
                fingers[side].add(g)
                if "pad_collision" in name:
                    pads[side].add(g)

    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    graspable = {g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) == target_body
                 and int(model.geom_group[g]) == 0}

    args = SimpleNamespace(
        grasp_mode="rule",
        rule_grasp_hand="R",
        grasp_contact_tol_mm=3.0,
        rule_grasp_policy_eps_rad=0.001,
        rule_grasp_release_eps_rad=0.05,
        rule_grasp_effort_eps=0.001,
        state_dump_on_rule_grasp=False,
    )
    for key, value in (args_overrides or {}).items():
        setattr(args, key, value)

    from robosuite_env.rollout.control.rule_grasp import _new_rule_grasp_state

    return SimpleNamespace(
        m=model,
        d=data,
        A=args,
        YUBI=None,
        ARM={"R": {
            "eef": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eef"),
            "fj": [int(model.jnt_qposadr[j]) for j in finger_joints],
            "fd": [int(model.jnt_dofadr[j]) for j in finger_joints],
            "fa": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                   for n in ("a_finger_left", "a_finger_right")],
        }},
        GRASPABLE_GEOMS=graspable,
        INNER_PAD_SIDE_GEOMS={"R": pads},
        FINGER_SIDE_GEOMS={"R": fingers},
        RULE_GRASP=_new_rule_grasp_state(),
    )


def run(object_name, render=False, out_dir=None, rule_grasp=True):
    import mujoco
    from robosuite_env.rollout.control.rule_grasp import _rule_grasp_update

    # Written into the asset directory rather than compiled from a string:
    # the YUBI and object models name their meshes by paths relative to it
    # (``yubi_right/...``, ``fr3_yubi_right_registered/...``,
    # ``trellis_office/...``), and letting MuJoCo resolve them from disk is
    # less brittle than reproducing that map in a dict.
    scratch = ASSETS / f".ufo_catcher_{object_name}.xml"
    scratch.write_text(build_model_xml(object_name), encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(scratch))
    finally:
        scratch.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    state = make_state(model, data)

    dt = float(model.opt.timestep)
    mount_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_mount")
    finger_a = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in ("a_finger_left", "a_finger_right")]
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_qpos = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")])

    data.ctrl[mount_a] = 0.0
    for a in finger_a:
        data.ctrl[a] = 0.7854
    mujoco.mj_forward(model, data)

    # Seat the object on the floor and aim the hand at its middle, both read
    # off the compiled model in this orientation rather than guessed.
    low, high = _object_world_z(model, data, state)
    data.qpos[target_qpos + 2] -= low
    mujoco.mj_forward(model, data)
    low, high = _object_world_z(model, data, state)
    # Aim the middle of the gripping surface at the middle of the object. The
    # pads sit forward of the finger joints, so using a finger body or the pad
    # geom origin leaves the hand short -- take the pads' own vertical span.
    pad_low, pad_high = _pad_world_z(model, data, state)
    depth = max(0.0, 0.5 * (pad_low + pad_high) - 0.5 * (low + high))
    # ...but do not drive the hand through the table. A flat object is gripped
    # at its edge, not by burying the pads; without this the tape run spends
    # itself pressing 116 N into the floor and never latches.
    depth = min(depth, max(0.0, pad_low - 0.0015))
    start_yaw = _yaw_off_square_deg(data.xquat[target_body])

    renderer = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)

    rows = []
    schedule = [("descend", DESCEND_SEC), ("close", CLOSE_SEC),
                ("lift", LIFT_SEC), ("hold", HOLD_SEC)]
    t = 0.0
    frame_every = max(1, int(round(1.0 / 30.0 / dt)))
    step_index = 0
    for phase, duration in schedule:
        for _ in range(int(round(duration / dt))):
            if phase == "descend":
                data.ctrl[mount_a] = -depth * min(1.0, (t / DESCEND_SEC))
            elif phase == "close":
                for a in finger_a:
                    data.ctrl[a] = 0.0
            elif phase == "lift":
                data.ctrl[mount_a] = -depth + 0.18 * min(1.0, (t - DESCEND_SEC - CLOSE_SEC) / LIFT_SEC)
            mujoco.mj_step(model, data)
            if rule_grasp:
                _rule_grasp_update(state, {"R": float(data.ctrl[finger_a[0]])})
            rows.append(_sample(model, data, state, phase, target_body, target_qpos))
            if renderer is not None and step_index % frame_every == 0:
                renderer.update_scene(data, camera="side")
                frames.append(renderer.render().copy())
            t += dt
            step_index += 1

    summary = _summarise(rows, start_yaw)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{object_name}_rows.json").write_text(json.dumps(rows), encoding="utf-8")
        (out / f"{object_name}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        if frames:
            _write_video(frames, out / f"{object_name}.mp4")
    return summary, rows, frames


def _yaw_off_square_deg(quat):
    """How far the object's long axis is from square to the jaws, in degrees.

    The jaws close along world x, so square is along world y. Zero means the
    jaws have it across their narrow dimension, which is the grip that holds.
    The models put their long axis on local z.
    """
    import mujoco

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    axis = mat.reshape(3, 3)[:, 2]
    planar = np.array([axis[0], axis[1]])
    norm = float(np.linalg.norm(planar))
    if norm < 1e-9:
        return 90.0
    planar = planar / norm
    # Undirected: a pen turned 175 degrees is 5 degrees off square, not 175.
    return float(math.degrees(math.acos(min(1.0, abs(planar[1])))))


def _long_axis_elevation_deg(quat):
    """How far the object's long axis has tipped out of horizontal.

    Without this the squareness angle is ambiguous: a pen hanging vertically
    has almost no horizontal component, so its projected heading is noise.
    """
    import mujoco

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    axis = mat.reshape(3, 3)[:, 2]
    return float(math.degrees(math.asin(max(-1.0, min(1.0, abs(axis[2]))))))


def _sample(model, data, state, phase, target_body, target_qpos):
    import mujoco

    peak_force = 0.0
    deepest_mm = 0.0
    pad_contacts = 0
    inner = state.INNER_PAD_SIDE_GEOMS["R"]["left"] | state.INNER_PAD_SIDE_GEOMS["R"]["right"]
    force6 = np.zeros(6)
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 in state.GRASPABLE_GEOMS and g2 in inner)
                or (g2 in state.GRASPABLE_GEOMS and g1 in inner)):
            continue
        pad_contacts += 1
        mujoco.mj_contactForce(model, data, ci, force6)
        peak_force = max(peak_force, float(force6[0]))
        deepest_mm = max(deepest_mm, max(0.0, -float(c.dist) * 1000.0))
    grasp = state.RULE_GRASP
    return {
        "t": round(float(data.time), 5),
        "phase": phase,
        "jaw_l": round(float(data.qpos[state.ARM["R"]["fj"][0]]), 6),
        "jaw_r": round(float(data.qpos[state.ARM["R"]["fj"][1]]), 6),
        "obj_z_mm": round(float(data.xpos[target_body][2]) * 1000.0, 3),
        "obj_yaw_off_deg": round(_yaw_off_square_deg(data.xquat[target_body]), 3),
        "obj_elev_deg": round(_long_axis_elevation_deg(data.xquat[target_body]), 3),
        "pad_contacts": pad_contacts,
        "peak_force_n": round(peak_force, 4),
        "pen_mm": round(deepest_mm, 4),
        "active": bool(grasp.get("active")),
        "welded": grasp.get("weld_eq_id") is not None,
    }


def latch_row_is_capped(rows):
    """Whether the weld fired on the one-second cap rather than on a stall."""
    try:
        from robosuite_env.rollout.control.rule_grasp import RULE_GRASP_CLOSING_MAX_SEC
    except ImportError:
        # A tree without this constant welds at latch and has no closing phase.
        return False

    latch = next((r for r in rows if r["active"]), None)
    weld = next((r for r in rows if r["welded"]), None)
    if latch is None or weld is None:
        return False
    return (weld["t"] - latch["t"]) >= RULE_GRASP_CLOSING_MAX_SEC - 1e-6


def _summarise(rows, start_yaw):
    lifted = [r for r in rows if r["phase"] in ("lift", "hold")]
    held = [r for r in rows if r["active"]]
    weld_row = next((r for r in rows if r["welded"]), None)
    weld_reason = ("time_cap" if weld_row and latch_row_is_capped(rows) else "stall")
    latch_row = next((r for r in rows if r["active"]), None)
    final = rows[-1]
    return {
        "latched": latch_row is not None,
        "latch_t": latch_row["t"] if latch_row else None,
        "welded": weld_row is not None,
        "weld_t": weld_row["t"] if weld_row else None,
        "closing_sec": (round(weld_row["t"] - latch_row["t"], 4)
                        if weld_row and latch_row else None),
        "weld_reason": weld_reason if weld_row else None,
        "jaw_at_latch": [latch_row["jaw_l"], latch_row["jaw_r"]] if latch_row else None,
        "jaw_at_weld": [weld_row["jaw_l"], weld_row["jaw_r"]] if weld_row else None,
        "jaw_closed_after_latch_rad": (
            round((latch_row["jaw_l"] + latch_row["jaw_r"])
                  - (weld_row["jaw_l"] + weld_row["jaw_r"]), 6)
            if weld_row and latch_row else None),
        "start_yaw_off_deg": round(start_yaw, 3),
        "final_yaw_off_deg": final["obj_yaw_off_deg"],
        "final_elev_deg": final["obj_elev_deg"],
        "yaw_squared_up_deg": round(start_yaw - final["obj_yaw_off_deg"], 3),
        "max_obj_z_mm": round(max(r["obj_z_mm"] for r in rows), 3),
        "final_obj_z_mm": final["obj_z_mm"],
        "lifted": bool(lifted and max(r["obj_z_mm"] for r in lifted) > 40.0),
        "peak_force_n": round(max(r["peak_force_n"] for r in rows), 4),
        "max_pen_mm": round(max(r["pen_mm"] for r in rows), 4),
        "held_steps": len(held),
        "steps": len(rows),
    }


def _write_video(frames, path):
    import subprocess

    height, width = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "30", "-i", "-",
         "-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "4M", str(path)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=sorted(OBJECTS), default="pen")
    parser.add_argument("--out", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-rule-grasp", action="store_true")
    arguments = parser.parse_args()

    summary, _, _ = run(
        arguments.object,
        render=arguments.render,
        out_dir=arguments.out,
        rule_grasp=not arguments.no_rule_grasp,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
