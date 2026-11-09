# FR3 visual meshes

These material-split meshes are derived from the official Franka Robotics
`franka_description` FR3 visual meshes (`link0.dae` through `link7.dae`).

- Source: https://github.com/frankarobotics/franka_description
- Source commit: `02afaae282d4a8e10d7d2f781b23b3515c303ce5`
- Source path: `meshes/robots/fr3/visual/`
- Conversion: each COLLADA scene-node transform was baked into a binary STL;
  material colors are recorded in `manifest.json` and applied by the MJCF.
- Units: metres

The original Apache License 2.0 text is included as `LICENSE`. Collision
geometry is the unmodified upstream FR3 collision set (`fr3_collision/`,
installed from the Hugging Face folder `assets/fr3_collision/`), not these
visual meshes.
