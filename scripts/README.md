# Running image-gen on Pegasus

## Confirmed cluster values

Observed on Pegasus, September 2026. Re-check with the command in the right
column if something stops working.

| What | Value | Verify with |
|---|---|---|
| Account | `ei-external` | `sacctmgr show assoc user=$USER format=account,partition,qos%30` |
| GPU request syntax | `--gres=gpu:N` | `sinfo -o "%20P %12l %18G"` — GRES reads `gpu:8`, no type name |
| Partition (recommended) | `RTXA6000` — 48 GB, 3-day limit, 13 nodes | `sinfo -o "%20P %12l %18G %8D %T"` |
| Other usable partitions | `A100-80GB`, `A100-40GB`, `RTX3090`, `V100-32GB` | as above |
| Interactive limits | `--time ≤ 8h`, `--immediate ≤ 3600`, both **required** | the `cli_filter` error message |
| Batch time limit | 3 days (A100/RTX), 1 day (H100/H200/B200) | `sinfo` TIMELIMIT column |
| Scratch | `/netscratch/$USER`, BeeGFS, ~248 T free | `df -h /netscratch/$USER` |
| Login node Python | 3.12.3 — but see the numpy trap below | `python3 --version` |
| Package index | local mirror at `http://pypi-cache/index`, very fast | pip output |
| Container images | `ls -la /enroot/*.sqsh`; `cuda12.4python3.10.sqsh` is a candidate | |

SDXL needs ~12 GB of VRAM, so any of those partitions works. `RTXA6000` is the
pick because it has by far the most nodes, and queue time is what actually costs
you.

## Do you need a container?

Not so far. The login node has Python 3.12.3, and the venv on `/netscratch` is
readable from compute nodes, so `pegasus_setup.sh` runs without Enroot and jobs
reuse the same venv.

A container becomes worth it if a compute node's CUDA driver disagrees with the
pip-installed torch, or you need a Python that the login node lacks. In that
case build the venv **inside** the container, since a venv hardcodes the
interpreter that created it:

```bash
srun --partition=RTXA6000 --gres=gpu:1 --account=ei-external \
  --time=02:00:00 --immediate=600 \
  --container-image=/enroot/<IMAGE>.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,"$PWD":"$PWD" \
  --container-workdir="$PWD" \
  --pty bash
```

## Step 1 — set up the environment (login node, has internet)

```bash
cd /netscratch/$USER
git clone <your-repo> image-gen-service
cd image-gen-service

./scripts/pegasus_setup.sh                        # core only, no torch
./scripts/pegasus_setup.sh --with-gpu --with-weights   # + torch, + ~13 GB of weights
```

The script refuses to put the venv or `HF_HOME` under `$HOME` — that 10 GB cap is
the most common way to lose an afternoon here.

Run it **without** `--with-gpu` first. That proves the boring layer — Python
version, dependency install, entry point, `/netscratch` writes — with four small
pure-Python dependencies instead of 2.5 GB of torch. When you then add torch,
anything that breaks is torch's fault, not a mystery.

Weights must be fetched here: **compute nodes have no internet**. The pins land
in `configs/revisions.lock.json` — commit it, the revisions are part of the
reproducibility claim.

## Step 2 — prompt iteration (interactive, has a GPU)

Interactive jobs cap at 8 h — plenty for iterating, no use for a long-lived
service.

A `cli_filter` plugin rejects interactive jobs that omit these: `--pty` requires
`--time` of at most **8 hours** and `--immediate` of at most **3600** seconds.
`--immediate` is how long srun waits for an allocation before giving up.

```bash
srun --partition=RTXA6000 --gres=gpu:1 --account=ei-external \
  --time=02:00:00 --immediate=600 \
  --container-image=/enroot/<IMAGE>.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER \
  --pty bash

source /netscratch/$USER/venvs/image-gen/bin/activate
export HF_HOME=/netscratch/$USER/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

image-gen gen --kind skybox --prompt "a forest with blue trees" --seed 1234
```

Or keep the model warm behind the HTTP API and drive it from a browser:

```bash
uvicorn image_gen.api.app:app --host 0.0.0.0 --port 8000
```

```bash
# from your own machine -- find the node with: squeue -u $USER -o "%N"
ssh -J <you>@<LOGIN-HOST> -L 8000:<COMPUTE-NODE>:8000 <you>@<COMPUTE-NODE>
#   then open http://localhost:8000/docs
```

First request pays the ~30–60 s model load; every one after is ~20–60 s. Change
a word, resend, compare. **Whether Pegasus permits SSH to compute nodes is a
local policy question** — it is how Jupyter is normally run, but confirm it. If
it is blocked, use the CLI above and `scp` the PNGs back.

## Step 3 — production batches (sbatch)

Not written yet — build step 6. The shape it will take:

```bash
#SBATCH --array=0-9
#SBATCH --output=logs/%A_%a.out

image-gen run queue.json \
  --shard-index $SLURM_ARRAY_TASK_ID \
  --shard-count $SLURM_ARRAY_TASK_COUNT
```

Each array task owns a disjoint set of scenes, so per-scene manifests need no
locking. A dependent merge job then combines them:

```bash
sbatch --dependency=afterany:$ARRAY_JOB_ID scripts/pegasus_merge.sbatch
```

**`afterany`, not `afterok`** — if some tasks fail, the manifest must still be
built, or the failures become invisible.

## Slurm quick reference

| | |
|---|---|
| Submit | `sbatch job.sbatch` |
| Queue | `squeue -u $USER` |
| Which node is my job on | `squeue -u $USER -o "%.18i %.9P %.8j %.2t %.10M %N"` |
| Outcome + exit codes | `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` |
| Cancel | `scancel <jobid>` (or `scancel -u $USER`) |
| Live log | `tail -f logs/<jobid>_<task>.out` |

`image-gen` exit codes: `0` all ok, `1` fatal, `2` completed with some assets
failed. `sacct` surfaces these per array task.

## Things that bite on this cluster

- **numpy and torch cannot be imported on the login node.** `login1` is a VM
  with a conservative CPU model that masks SSE4.2, so it does not meet
  **x86-64-v2** — the baseline numpy 2.x wheels are built against:

  ```
  RuntimeError: NumPy was built with baseline optimizations:
  (X86_V2) but your machine doesn't support: (X86_V2).
  ```

  Confirm with `grep -oE 'sse4_2|popcnt|cx16' /proc/cpuinfo | sort -u` — it
  comes back empty on the login node and populated on any compute node. This is
  not something to fix: install and download on the login node (both are pure
  Python), and run anything that imports torch on a compute node. The same venv
  on `/netscratch` works there, because it is the CPU that differs, not the
  wheel. A first failed import can also surface as the misleading
  `ImportError: cannot load module more than once per process`; re-run with
  `python -E` to see the real cause.

- **`$HOME` is capped at 10 GB.** Venvs, weights and outputs all go on
  `/netscratch`. `pegasus_setup.sh` enforces this.
- **Compute nodes have no internet.** Anything needing a download happens on the
  login node. Set `HF_HUB_OFFLINE=1` in jobs so a missing file fails loudly
  instead of hanging on a connection attempt.
- **Interactive jobs cap at 8 h** and must pass `--time` and `--immediate`
  explicitly, or `cli_filter` rejects them. Batch partitions allow far longer
  (3 days on the A100/RTX partitions), so a long-running server is still not the
  main path — batch is.
- **Queue time is unpredictable.** Submitting is not starting. This is why
  generation cannot sit on a participant's critical path.
- **Shell scripts need LF line endings.** Edited on Windows, a `.sh` file can
  pick up CRLF and fail with `bad interpreter: /usr/bin/env bash^M`. The
  repo's `.gitattributes` forces LF; if you hit it anyway, `dos2unix` the file.
