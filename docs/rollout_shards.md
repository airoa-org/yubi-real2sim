# Rollout shards

`runner.shards` can split one logical rollout run across existing Slurm job
allocations. It is optional: when the key is absent, `real2sim-run` uses the
original single-process launcher path and passes the original environment and
argument vector to `scripts/run_policy_batch.sh`.

The launcher does not submit jobs. Every configured allocation must already
exist and remain active for generation and finalization. It starts one
overlapping step per shard with:

```text
srun --jobid=<id> --overlap --cpu-bind=none [--cpus-per-task=<n>] scripts/run_policy_batch.sh
```

All steps write their absolute `run_NN` directories into the same run root.
Shard-local service files live below `.shards/<name>/`, and the complete
stdout/stderr stream is written to `shard_<name>.log`. If one shard fails, the
launcher terminates the remaining steps and does not finalize the run. After
all shards succeed, it uses the first shard's allocation for one aggregation
step (`aggregate.log`), then runs the launcher completion check once. The
aggregation step creates `outputs.csv`, summaries, post-hoc videos, and the
rollout grid from the complete shared run root.

## YAML form

```yaml
runner:
  repeats: 100
  policy_batch_size: 25
  shards:
    - name: a
      repeats: [1, 50]
      slurm_job_id: "1001"
      gs_cuda_devices: "0,1,2,3"
      sim_cuda_devices: "0,1,2,3"
      policy_cuda_device: "0"
      cpus_per_task: 28
      policy_batch_size: 25
      server_port: 8871
      gs_port_base: 10201
    - name: b
      repeats: [51, 100]
      slurm_job_id: "1002"
      gs_cuda_devices: "0,1"
      sim_cuda_devices: "0,1"
      policy_cuda_device: "0"
      policy_batch_size: 10
```

`name`, `slurm_job_id`, `cpus_per_task`, `policy_batch_size`, `server_port`,
and `gs_port_base` are optional. Names default to `01`, `02`, and so on. An
omitted job ID uses the launcher's `SLURM_JOB_ID`. Batch size defaults to
`runner.policy_batch_size`. Policy and 3DGS ports are assigned from the
runner's resolved runtime ports while avoiding every other shard's assigned
policy port and 3DGS port range.

Configuration resolution fails before launch when `shards` is empty, names
or ranges are invalid, the inclusive ranges overlap or leave a gap in
`1..runner.repeats`, a batch is too large or cannot form complete policy
waves, a job/GPU selector is invalid, or any explicit/automatic port range
collides.

## Command-line override

Use `--shards` to replace (not extend) YAML `runner.shards` without editing the
profile. Quote the whole value. Each whitespace-separated entry has the form
`JOB:GPU[,GPU...]:START-STOP`. For a profile whose `runner` block matches the
YAML above (`repeats: 100`, `policy_batch_size: 25`; `PROFILE.yaml` stands for
that profile, and the job IDs are placeholders):

```bash
python -m real2sim.launcher \
  --config PROFILE.yaml \
  --shards '1001:0,1,2,3:1-50 1002:0,1:51-100'
```

The compact form uses the GPU list for both GS and simulation, uses its first
selector for policy inference, and inherits batch sizes and ports from the
profile. Use YAML when a shard needs a distinct CPU count, batch size, policy
GPU, or explicit ports. `--dry-run` shows the fully resolved shard list.

## Direct range execution

The lower-level shell runner accepts inclusive absolute ranges through
`--repeat-start` / `REPEAT_START` and `--repeat-stop` / `REPEAT_STOP`:

```text
scripts/run_policy_batch.sh --repeats 100 --repeat-start 51 --repeat-stop 100
```

This produces `run_51` through `run_100`; numbering never restarts at one. A
partial range validates and records only its own rollouts and deliberately
defers aggregate summaries, post-hoc rendering, the grid, poster, and grid
manifest. Normal operators should use `runner.shards`, whose final aggregation
performs those deferred whole-run checks.
