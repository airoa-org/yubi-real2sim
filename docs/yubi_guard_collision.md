# Shared YUBI exterior collision contract

The bilateral YUBI exterior collision is a shared robot asset.  Its source of
truth is
`robosuite_env/tools/build_yubi_full_collision.py`; the machine-readable
review record is `robosuite_env/assets/yubi_guard_collision_manifest.json`.
The generator updates both handed MJCFs and writes deterministic proxy meshes
below `yubi_right/guard_collision/` and `yubi_left/guard_collision/`.

## Coverage and inventory

Each handed asset has 60 colored YUBI visual parts.  The four moving
`geometry6` / `geometry7` parts already had reviewed, per-hull collision for
the structural fingertip and rubber pad.  Those geoms and their contact values
remain unchanged.  The generator covers all other 56 visual parts with 102
new guards per hand:

| Side | Fixed guards | Moving guards | New guards | Final collision geoms, including mount and existing fingertip/pad |
| --- | ---: | ---: | ---: | ---: |
| Right | 70 | 32 | 102 | 178 |
| Left | 70 | 32 | 102 | 179 |

The default reduction is one conservative convex proxy per source part.
Reviewed concave housing and finger parts use four or eight deterministic
spatial clusters to avoid filling the finger aperture with one large convex
hull.  All source collision-face vertices are inside the generated proxy
union.  Detailed visual-face vertices are inside the union to within 0.1 mm,
well inside the 1 mm contact margin.  A 17-sample sweep from finger joint
`q=0` through `q=0.7854 rad` has no guard self-contact; the 0.1 mm center-axis
probe also stays clear over the reviewed grasp corridor.  The exact numerical
clearance and volume audit is stored in the generated manifest.

The guard geoms have `mass=0`, so adding them does not change YUBI body mass or
inertia.  They reuse the corresponding visual geom transform.  Collision masks
`contype=2`, `conaffinity=1` preserve contact with ordinary environment geoms
while preventing guard-to-guard self-collision.

## Contact behavior and limits

Every new exterior guard has this shared contact contract:

| Parameter | Value |
| --- | --- |
| `condim` | `3` |
| `friction` | `0.1 0.005 0.0001` |
| `solref` | `0.004 1` |
| `solimp` | `0.99 0.9999 0.0001` |
| `priority` | `1` |
| `margin`, `gap` | `0.001`, `0` m |

At the production MuJoCo timestep of 2 ms, the 4 ms time constant is the
stiffest value allowed by the safe-reference rule (`timeconst >= 2 *
timestep`).  The 1 mm margin activates the constraint before proxy surfaces
meet.  Regression probes at 0.01 m/s and at the FR3 Cartesian limit of
0.4 m/s must retain positive surface separation, and the test verifies that
the first effective contact actually uses the priority-1 friction, `solref`,
and `solimp` values above.

This is a stiff, low-friction discrete-time contact contract, not a
mathematical zero-penetration guarantee for arbitrary impact speeds.  MuJoCo
contacts are soft constraints; velocities above the reviewed 0.4 m/s envelope,
a larger timestep, or a changed solver can transiently penetrate the physical
proxy surface.  Runtime penetration monitoring must therefore remain enabled
where safety depends on it.

The low-friction guarantee also assumes the contacting task/world geom keeps
priority below 1, as the current asset inventory does.  A future external geom
with priority 1 or greater can change MuJoCo's contact-parameter combination
and must be reviewed separately.  Existing YUBI rubber pads deliberately keep
their high-friction priority-1 contract.

`robosuite_env.rollout.build.controllers.configure_finger_contact()` applies
command-line overrides only to the existing fingertip/pad set.  Generated
`guard_*` geoms are explicitly excluded, so a rollout cannot silently soften
or increase the friction of this contract.

## Regeneration and checks

Run the bilateral generator from the repository root:

```bash
.venv/bin/python robosuite_env/tools/build_yubi_full_collision.py --write
.venv/bin/python robosuite_env/tools/build_yubi_full_collision.py --check
```

The generator imports no first-party package, so it needs no `PYTHONPATH`.

`--check` performs no writes and fails if an MJCF, proxy OBJ, manifest entry,
or stale generated mesh differs from the deterministic output.  The focused
regression suite is:

```bash
.venv/bin/python -m pytest -q tests/unit/robots/test_yubi_guard_collision.py
```

It checks bilateral generation, visual and collision surface coverage,
compiled contact values, unchanged mass/inertia, runtime-override isolation,
the joint sweep, the 0.4 m/s contact envelope, warnings, and a bounded compile
and step cost.
