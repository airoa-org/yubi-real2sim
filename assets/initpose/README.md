# Initial object positions

Where each object starts in a release condition, one file per robot,
scene and task:

```
assets/initpose/<robot>/<scene>/<task>.json
assets/initpose/<robot>/<scene>/<task>_<checkpoint>.json
```

The release has one file, `fr3/flat/cup_cup1trim.json`: the cup and plate
positions of the FR3 Flat cup task for the `pi05_cup` checkpoint (`cup1trim`).
Where the positions depend on the checkpoint, the checkpoint follows the task.

Both release profiles name it in `task.objects`. It carries the object
positions the published run read;
`assets/release/fr3/flat/cup/MANIFEST.json` records where it came from and its
sha256.

The `source` block of a file records how the placement was made before this
repository was published. Files marked "(not kept in the repository)" and
tools marked "(development tool)" are not part of this repository; the same
holds for a `MANIFEST.json` `source` marked "(not kept in the repository)".

A `placed_by` note that says "manual" marks a placement made by hand on the
placement board; "automated" marks one made by a scripted position sweep.

Only the provenance text of these files was edited after the published run.
For such a file the condition `MANIFEST.json` pins the tracked bytes as
`sha256` and keeps the digest that run read as `recorded_sha256`; the object
positions are unchanged.
