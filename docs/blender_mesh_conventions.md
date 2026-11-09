# Blender mesh assembly conventions

This document is the repository source of truth for static Blender assemblies
of Franka FR3, YUBI, and mounting hardware.

## Why these checks are required

GLB exports produced from simulation state can be misleading in several ways:

1. A node named `*_visual` can still contain the low-resolution collision
   mesh. Object names are not provenance.
2. Blender's glTF import preserves node transforms as object-local rotation.
   The model can look correct while every object retains an unexpected local
   rotation, which causes problems in later editing and export.
3. `rotation_euler` can already report zero while the imported rotation still
   exists in `matrix_basis`.
4. A raw source STL and a mesh inside an exported GLB do not necessarily share
   the same immediate local coordinate frame, even when their topology is
   identical. The exporter and Blender's glTF axis conversion can split the
   effective transform between mesh data and node matrices.

Treat source paths, topology counts, and post-save transform checks as the
authority instead of object names or a viewport-only inspection.

## Franka FR3 mesh sources

Visual geometry:

- directory: `robosuite_env/assets/fr3_visual/`
- manifest: `robosuite_env/assets/fr3_visual/manifest.json`
- upstream: official Franka Robotics `franka_description`
- pinned source commit: `02afaae282d4a8e10d7d2f781b23b3515c303ce5`
- upstream path: `meshes/robots/fr3/visual/`

Collision geometry:

- directory: `robosuite_env/assets/fr3_collision/`
- files: `link0.stl` through `link7.stl`
- not tracked in git: the asset downloader installs it from the `fr3_collision`
  entry in `artifacts/manifest.yaml`

The visual meshes are split by source material. Do not replace them with one
collision STL per link merely because an exported GLB calls those nodes
`robot0_linkN_visual`.

## Required topology verification

| Link | Visual objects | Visual triangles | Collision triangles |
| ---: | ---: | ---: | ---: |
| 0 | 7 | 63,170 | 374 |
| 1 | 1 | 8,028 | 236 |
| 2 | 1 | 8,028 | 228 |
| 3 | 2 | 21,566 | 270 |
| 4 | 2 | 21,566 | 262 |
| 5 | 3 | 27,145 | 266 |
| 6 | 8 | 44,837 | 323 |
| 7 | 4 | 37,958 | 314 |
| **Total** | **28** | **232,298** | **2,273** |

A Franka assembly with eight objects and 2,273 triangles is the collision
model, regardless of its object names.

## Required Blender collection layout

Use the following Outliner structure for a combined static assembly:

```text
FR3_YUBI_Assembly
├── visual_mesh
│   ├── Franka_Official_Visual
│   ├── YUBI
│   └── Flange
└── collision_mesh                  # hidden in viewport and render
    └── Franka_Collision
```

Keep collision geometry in the file when useful for reference, but never mix
it into the visual collection. Collision geometry must be hidden by default.

## YUBI physical root and hand_root

For the FR3/YUBI assembly, do not apply the `hand_root` translation to the
physical YUBI mesh root. The `chain` in
`robosuite_env/assets/fr3_yubi_right_registered/manifest.json` records two
distinct frames:

- `fr3_link8 -> YUBI_physical_root` (`link8_to_yubi_physical_root`): zero
  translation (the physical YUBI root is at the flange origin) and
  `rpy = [0, 0, -3pi/4]`;
- `YUBI_physical_root -> YUBI_hand_root` (`yubi_physical_root_to_hand_root`):
  approximately `[0, 0.0015, 0.051] m` with `rpy = [-pi/2, -pi/2, 0]`.

The `+51 mm` value is a child reference-frame offset, not a physical mounting
translation. A Blender assembly should expose `YUBI_physical_root` and
`YUBI_hand_root` as separate reference objects and verify their relative
translation after reopening the file.

## Registered YUBI mounting geometry

Do not reconstruct the mount from raw CAD STL files: those files are
not repository assets. The registered, metre-scale source of truth is
`robosuite_env/assets/fr3_yubi_right_registered/manifest.json` and the OBJ
files it names:

| Mesh | Path | Triangles |
| --- | --- | ---: |
| tool flange | `robosuite_env/assets/fr3_yubi_right_registered/visual/right_yubi_flange.obj` | 3,884 |
| FR3 attachment | `robosuite_env/assets/fr3_yubi_right_registered/visual/right_yubi_attachment.obj` | 12,884 |

Their transforms and scale are already baked in the
`YUBI_physical_root, metres` coordinate frame. Both `yubi_right.xml` and
`yubi_left.xml` reference these registered visuals and their CoACD collision
hulls; use the manifest rather than reapplying a CAD pre-rotation
or fitting a standalone flange.

For a static initial-pose assembly, apply the manifest chain exactly. It records
`fr3_link8 -> YUBI_physical_root` as zero translation and yaw
`-2.356194490192345` radians, and
`YUBI_physical_root -> YUBI_hand_root` as approximately
`[0, 0.0015, 0.051] m` with roll/pitch `[-pi/2, -pi/2]`. The registered colored
YUBI translation in the physical-root frame is approximately
`[0, -0.012, -0.0165] m`. Do not subtract 51 mm from mesh positions: that is a
child reference-frame offset, not a mount correction.

## Transform policy for static assemblies

glTF/GLB node matrices may appear in Blender as non-zero object-local
rotations. For a static assembly:

1. preserve each object's world-space vertex positions;
2. bake rotation and scale into the mesh data;
3. keep the world translation on the object;
4. set local rotation to `(0, 0, 0)` and scale to `(1, 1, 1)`; and
5. remove object parenting.

Set `matrix_basis` explicitly to a translation-only matrix after baking. Do
not rely on assigning zero to `rotation_euler`: if that property already reads
zero, Blender may retain the non-decomposed glTF rotation in `matrix_basis`.

When a raw STL must inherit a pose from a GLB, use a known corresponding mesh
(for example the FR3 collision mesh) to fit the rigid source-to-world
transform. Apply that verified transform to both the raw visual and collision
assets. Do not apply the imported GLB node matrix directly to raw STL vertices
without proving that their local coordinate frames match.

Do not use this policy for an articulated rig unless explicitly requested;
articulated models need meaningful link-local transforms and hierarchy.

## Validation

Reopen the saved file in a fresh Blender process and confirm:

- `Franka_Official_Visual` has 28 objects and 232,298 triangles;
- `Franka_Collision` has 8 objects and 2,273 triangles;
- `visual_mesh` is visible;
- `collision_mesh` is hidden in both viewport and render;
- every static mesh object's local rotation is zero;
- the 3x3 portion of every static mesh object's `matrix_basis` is identity;
- every static mesh object's local scale is one;
- no static mesh object has an object parent; and
- YUBI, flange, Franka visual, and Franka collision geometry retain their
  intended world-space alignment.

Validate alignment by comparing world-space mesh vertices. Do not use
`Object.bound_box` as the sole post-bake check because Blender can retain a
stale bounding-box cache after mesh-data transforms. Reopen the saved file and
repeat the vertex check in a fresh process.
