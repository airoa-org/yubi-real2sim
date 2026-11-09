# Architecture

## Production flow

```text
.env + schema-v2 YAML
        │
        ▼
real2sim.cli.main                         real2sim-run console entry point
  ├─ real2sim.config.*                    parse, validate, resolve
  ├─ real2sim.preflight                   runtime checks
  ├─ real2sim.launch_provenance           seal config/source provenance
  └─ real2sim.runner.execute
        │
        ▼
scripts/run_policy_batch.sh               process/GPU lifecycle
  ├─ OpenPI batch policy server
  ├─ one 3DGS server per worker
  └─ python -m robosuite_env.sim           worker entry point
        │
        ▼
robosuite_env.rollout.worker
  validate → build → controllers → grasp/GS → settle → render → frames → output
        │
        ▼
outputs/<run>/                             logs, videos, manifests, summaries
```

By default, this entire flow runs as child processes of `real2sim-run` in the
current interactive shell. Optional [`runner.shards`](rollout_shards.md)
places one shell-runner process in each operator-supplied, existing Slurm
allocation and rejoins them before finalization. Resource provisioning stays
outside the repository; no component submits or cancels allocations or
discovers job IDs.

`src/real2sim/launcher.py` is a thin entry-point module: it re-exports the
public entry point but does not own launcher behavior. Argument parsing and
orchestration live in `src/real2sim/cli.py`. Likewise, `robosuite_env/sim.py`
is a thin entry-point module that dispatches to the structured rollout package;
it contains no simulator implementation.

## Launcher ownership

```text
src/real2sim/
├── cli.py                       command parsing and top-level orchestration
├── launcher.py                  compatibility imports only
├── config/
│   ├── schema.py                raw load and resolved plan
│   ├── keys.py                  accepted schema keys
│   ├── parameters.py            simulation-parameter validation
│   ├── controllers.py           controller compatibility rules
│   ├── paths.py                 templates and resolved paths
│   └── reproducibility.py       source, lock, environment, support checks
├── artifacts.py                manifest loading and byte verification
├── artifacts_binding.py        profile-to-artifact binding
├── preflight.py                runtime and production gates
├── runner/
│   ├── argv.py                  exact worker arguments
│   ├── environment.py           allowlisted worker environment
│   └── execute.py               shell-runner execution
├── reporting/
│   ├── dry_run.py               resolved-plan display
│   ├── completion.py            completion contract
│   ├── summary.py               aggregate result files
│   └── posthoc_provenance.py    rerender/source verification
├── launch_provenance.py         pre-launch provenance and run manifest
├── provenance.py                repository/dependency identity
└── runtime_manifest.py          runtime source selection and seal
```

Machine-local paths belong in `.env`; experiment behavior belongs in YAML.
The launcher resolves both before the shell runner starts and records the
result. The shell runner owns child-process cleanup, render locks, explicit GPU
assignment, and exact-B policy waves—not experiment defaults.

## Simulator ownership

```text
robosuite_env/
├── contracts/       fail-closed repository, scene, and worker contracts
├── policy/          observation transforms and batch policy client/server
├── rendering/       wrist rendering and appearance calibration
├── robots/          FR3/YUBI models and controllers
├── rollout/         one rollout's lifecycle and outputs
├── tasks/           environments, object placement, and success criteria
├── tools/           operator-facing builders and diagnostics
├── paths.py         canonical repository resources
└── sim.py           compatibility entry point
```

Root-level contract and policy-server modules that remain are import/CLI
compatibility shims. New code should import the package that owns the behavior.

The rollout implementation is decomposed around a `RolloutSession` shared by
the ordered stages:

```text
rollout/
├── worker.py                  small ordered-stage coordinator
├── session.py                 shared live rollout state
├── setup/                     argument and frame setup
├── build/                     model, geometry, controllers, gripper, GS server
├── stages/                    settle, renderer init, frame loop, finalization
├── control/                   IK, trajectories, substeps, grasp intent/latch
├── observe/                   policy client, deploy anchor, wrists, OBS log
├── render/                    GS client, compositing, overview, video capture
├── state/                     MuJoCo state, settling, cache, fingerprints
├── replay/ and sources/       recorded-input adapters
├── grasp/                     explicit grasp-diagnostic scripting
├── contacts.py and metrics.py contact/success measurements
├── outputs.py                 per-rollout records and videos
├── policy_chunks.py           action-chunk lineage
├── resource_lifecycle.py      child/resource teardown
└── timing.py                  FR3 timing contracts
```

`worker.py` creates the session and invokes, in order, `validate_args`,
`build_model_env`, `setup_controllers`, `setup_grasp_and_gs`,
`settle_and_ready`, `init_policy_render`, `run_frames`, and
`finalize_outputs`. Pure calculations and independently testable I/O belong in
the smaller modules rather than being added back to the coordinator.

## Supported production profile

| Configuration | Scene | Rollouts | Frames | Policy batch |
| --- | --- | ---: | ---: | ---: |
| `config/release/fr3/flat/a_room_cup_cup1trim.yaml` (environment check) | `flat_a_room` | 10 | 300 | 2 |
| `config/release/fr3/flat/reading_room_cup_cup1trim.yaml` | `flat_reading_room` | 10 | 300 | 2 |

The two release profiles use schema v2. `fr3/flat/a_room_cup_cup1trim.yaml` is
the environment check a new host runs first and uses production reproducibility
mode; `fr3/flat/reading_room_cup_cup1trim.yaml` uses fixture mode. Both select
visible GPU slot `0` for the policy, 3DGS, and MuJoCo. An optional inherited
`CUDA_VISIBLE_DEVICES` value defines the ordered visible-device boundary;
without it, slots are host ordinals.

## Controller axes

| YAML key | Values | Scope |
| --- | --- | --- |
| `simulation.fr3_controller` | `joint_ik`, `cartesian_impedance` | FR3 arm motion ([FR3 Cartesian impedance](fr3_impedance.md)) |
| `simulation.fr3_nullspace` | `home`, `off` | `joint_ik` posture objective |
| `simulation.yubi_controller` | `cbpc`, `kinematic` | YUBI fingers |

`fr3_nullspace: off` forces the effective null-space gain to zero; `home` with
a configured zero gain is rejected so disabling the objective stays explicit.

## Output boundary

Before workers start, an official launch writes the original and resolved
configuration plus `run_manifest.json`. Completion requires every expected
worker to exit successfully, reach the configured frame count, stay within the
wall-clock limit, and emit any video required by the resolved render contract.
The launcher runs the aggregate CSV/JSON/PNG summarizer after the shell runner,
then reports success only when the completion contract and expected summary
count both pass. A direct `sim.py` invocation remains useful for development
but is recorded as `execution_class: diagnostic`, not an official evaluation.
