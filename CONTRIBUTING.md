# Contributing

We are not accepting external pull requests yet. Bug reports and questions via Issues on
[airoa-org/yubi-real2sim](https://github.com/airoa-org/yubi-real2sim) are welcome. Report
security problems privately as described in [`SECURITY.md`](SECURITY.md), not in a public issue.
This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

The rest of this file is for maintainers.

## Licence of contributions

The code in this repository is licensed under the [Apache License 2.0](LICENSE). By submitting a
contribution you agree that it is licensed under the same terms (inbound = outbound), as section 5
of the licence provides. Do not submit code, data or models that you may not license this way;
third-party material keeps its own licence and must be listed in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). In particular, the Apache License 2.0 does not cover
the 3DGS renderer in `third_party/GaussianWrapping`: it stays under the Inria/MPII
Gaussian-Splatting License, which allows non-commercial research and evaluation use only. Changes
to the YUBI gripper meshes listed in THIRD_PARTY_NOTICES A.4 stay under CERN-OHL-W-2.0.

## Development environment

Follow the [README](README.md) for the host prerequisites. Tests and lint do not need a GPU. For
CPU-only development, sync the locked environment without the CUDA extensions:

```bash
uv sync --locked --no-group cuda-extensions
```

`uv` must be version 0.12.3 (`pyproject.toml` requires it exactly). The GPU workflow (bootstrap,
inputs and rollouts) is in the README and [`docs/rtx_environment.md`](docs/rtx_environment.md).

## Tests

The `pytest` markers are defined in `pyproject.toml`:

- `gpu`: needs a GPU or EGL rendering context;
- `heavy`: needs a runtime dependency that `requirements-lock.ci.txt` does not install
  (MuJoCo, robosuite, OpenPI, ...);
- `assets`: needs git-ignored local inputs that a clean checkout lacks.

CI's `locked-runtime` job runs the CPU contracts with this marker selection in the environment it
syncs from the lock file. To run the same selection locally:

```bash
uv run --locked --no-group cuda-extensions python -m pytest -q -m "not gpu and not heavy and not assets"
```

Run the tests of the files you change first; add a test that fails without your change when you
fix a bug. A test that needs a GPU, a heavy dependency or local inputs must carry the matching
marker.

## Lint checks

CI's `lint` job runs ruff with the version it pins:

```bash
uvx ruff@0.17.0 check --no-fix . robosuite_env/rollout/build
```

and the vulture checks described under [Lint checks](README.md#lint-checks).
Do not commit local paths, home directories or machine names.

## Pull requests (maintainers)

- Keep one change per pull request, and explain in its description what changes and how you
  checked it.
- Update the documentation and configuration tables that describe the behaviour you change.
- `git diff --check` must pass.
