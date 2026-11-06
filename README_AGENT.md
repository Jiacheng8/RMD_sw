# Runbook: the two K512 Z-Image experiments on a fresh machine

This file is for a coding agent (Codex, Claude Code, …) asked to run the two experiments on a
new Linux GPU machine, from an empty disk to two finished training runs. Follow the steps in
order. Every command is safe to re-run: finished work is detected and skipped.

## What you are running

Two separate distillation runs of Z-Image-Turbo into a 2-step, 1024 px student. Both start
from the original pretrained weights and train for 500 updates with the same data, seed,
window (K=512, B=64) and MMD encoders (DINOv3-L + Qwen3-VL-4B):

| group | name on the command line | config | run directory name |
|---|---|---|---|
| 1 | `mmd-only` | `z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_mmd_only_from0.yaml` | `zimage-2step-1024-dino-qwenvl-k512-b64-mmd-only-from0` |
| 2 | `mmd-gan` | `z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_fixed_from0.yaml` | `zimage-2step-1024-dino-qwenvl-k512-b64-fixed-from0` |

Group 2 adds a fixed-weight (`5e-5`) GAN term on top of the same MMD objective. Group 2 must
**not** resume from Group 1. The script always starts each group fresh.

All the work goes through one script: `z-image/scripts/k512_two_groups.sh`.

## Rules

- **Do not edit the two YAML configs.** Hyperparameters are the experiment. The script passes
  machine-specific values (paths, attention backend, micro-batch) as `--set` overrides.
- **Do not use `ops/h20-run-k512-b64-fixed.sh`.** It only works on the original H20 server.
- **Never delete or overwrite a run directory, checkpoint or downloaded data** to get past an
  error. If the script refuses to start because a run directory already holds a run, stop and
  report it to the user.
- **Do not kill processes you did not start.** If the GPUs are busy, report it and wait.
- Run everything inside `tmux` (or `screen`). The full pipeline takes many hours.
- Only change one thing at a time. The supported knobs are listed in
  [Handling failures](#handling-failures).

## 0. Requirements: check before anything else

Run these checks and stop if one fails:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv   # >= 2 GPUs, each >= 80 GB
nvidia-smi | head -4                     # driver must support CUDA 12.6 (driver >= 560)
df -h <data disk>                        # >= 900 GB free (see the budget below)
bash --version | head -1                 # >= 4.4
command -v git python3 tmux curl         # all present
command -v conda || echo "no conda"      # install Miniconda if missing (below)
```

What the checks must show:

- **GPUs.** At least 2 GPUs with 80 GB or more each (A100 80GB, H100, H200, H20). 40 GB and
  24 GB cards cannot hold the 6B student. The tested target is 4 × A100 80GB. Any GPU count
  that divides 64 / micro-batch works (1, 2, 4, 8, …).
- **Disk, about 900 GB in total:**

  | item | size |
  |---|---|
  | model weights | 41 GB |
  | teacher tars | 162 GiB |
  | prompt contexts | 70 GB |
  | Group 1: 10 checkpoints × 24.6 GB + `resume.pth` (~26 GB, held twice while it is replaced) | ~300 GB |
  | Group 2: same as Group 1 | ~300 GB |
  | Qwen-Image-Bench judge, only with `--eval` | +65 GB |

- **Network.** Access to `huggingface.co` and `github.com`. Behind a mirror, export
  `HF_ENDPOINT=https://<mirror>` before running.

If conda is missing, install Miniconda on the data disk:

```bash
curl -fsSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/mc.sh -b -p <data disk>/miniconda3 && eval "$(<data disk>/miniconda3/bin/conda shell.bash hook)"
```

### Credentials

- **Hugging Face.** All datasets and models used here are public. If a download fails with HTTP
  401 or 403 (gated weights such as DINOv3), the user must accept that model's licence on
  huggingface.co and give you a read token. Save the token as `RDM/.hf_token`, which git ignores,
  or export `HF_TOKEN`. Never print it or commit it.
- **Weights & Biases.** The script stops at the start unless `wandb login` has been done on the
  machine. If the user did not give you a wandb key, add `--no-wandb` to **every** command below.

## 1. Get the code

```bash
cd <data disk>
git clone -b zimage-exp https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
tmux new -s k512        # run everything below inside this session
ROOT=<data disk>/rdm-sets
```

## 2. Dry run (changes nothing)

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --dry-run
```

Check the `preflight` block it prints:

- `gpus`: lists the cards you expect.
- `batching`: reads `N GPUs x micro 2 x accum A = 64 rows per update`.
- `attention`: reads `native` on non-Hopper cards (A100) and FA3 on H100/H20.
- `disk`: no error.

It should then list stages 1–8 with the commands it would run. If it prints `ERROR:`, fix that
first (see [Handling failures](#handling-failures)).

## 3. Gate: prepare all data, then 3 updates per group

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --gate
```

The gate runs stages 1–7, which build everything the experiments need, then trains each group
for 3 updates in a separate directory (`$ROOT/zimage/work_dirs_gate`) and saves no checkpoint.
These 3 updates hit every memory peak of the real run.

The first time through, this command does most of the setup work and takes hours. Download speed
dominates. These times were not measured on A100:

| stage | what happens | where to watch |
|---|---|---|
| 1 env | creates conda env `rdm` (torch 2.8, diffusers 0.40) and installs aria2 | console |
| 2 download | Z-Image-Turbo, DINOv3-L, Qwen3-VL-4B → `$ROOT/hf`, `$ROOT/models/qwen3vl-4b` | console |
| 3 reference | `jiachengcui888/zimage-ref-v1` → `$ROOT/zimage/zimage-ref-v1` | console |
| 4 contexts | rebuilds the Qwen3 prompt contexts on every GPU (~70 GB) | `$ROOT/zimage/logs/contexts_rank*.log` |
| 5 qwen text | `jiachengcui888/zimage-ref-qwen-text` → `reference_store_qwen` | console |
| 6 images | 87 teacher tars (173 GB) from `jiachengcui888/zimage-turbo-reference`, SHA-verified and indexed | `$ROOT/zimage/logs/teacher_images.log` |
| 7 qwen-vl | Qwen3-VL features of all 125,808 teacher images → `reference_store_dino_qwenvl` | `$ROOT/zimage/zimage-ref-v1/reference_store_dino_qwenvl.partial/rank*.log` |
| 8 gate | 3 updates of `mmd-only`, then 3 of `mmd-gan` | `$ROOT/zimage/logs/train_<group>_gate.log` |

**The gate passed if** the console shows both of these lines:

```
GATE PASS mmd-only: 3 updates, ...
GATE PASS mmd-gan: 3 updates, ...
```

Each `peak memory per GPU (MiB)` line should also stay below the card size. Report the peak
values to the user.

If the gate fails with CUDA out of memory, follow [Handling failures](#handling-failures) and
run the gate again with the same command plus the fix. Do not start step 4 until both groups
pass.

## 4. The two experiments

Use **exactly the same flags that made the gate pass**, minus `--gate`:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT [--micro-batch 1] [--no-wandb]
```

This trains Group 1 (`mmd-only`) for 500 updates, then Group 2 (`mmd-gan`) for 500 updates.
Stages 1–7 are skipped because they are already done. If Group 1 fails, Group 2 does not start.

To run one group only, for example to run them on two machines:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT mmd-only
bash z-image/scripts/k512_two_groups.sh --root $ROOT mmd-gan
```

### Monitoring

```bash
RUN=$ROOT/zimage/work_dirs/zimage-2step-1024-dino-qwenvl-k512-b64-mmd-only-from0   # or ...-fixed-from0
tail -1 $RUN/train_log.jsonl | cut -c1-400     # one JSON row per update: "step", losses, lr, mmd2
wc -l < $RUN/train_log.jsonl                   # updates done so far (500 at the end)
ls $RUN | grep step_                           # step_0000050.pth ... step_0000500.pth
tail -20 $ROOT/zimage/logs/train_mmd-only.log
nvidia-smi
```

How to read the logs:

- **Loss columns.** Raw MMD² is logged per student step and encoder (`s1/…/mmd2`, `s2/…/mmd2`)
  separately from the force surrogate, which is not an MMD and can be negative.
- **Group 2 GAN columns.** Group 2 also logs discriminator and generator GAN losses. Its GAN
  generator term starts at update 51.
- **Learning rate.** It warms up over 100 updates, then follows a cosine decay to half its peak
  at update 500.

### Resuming after an interruption

If the machine or the session dies, continue the group that was running:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT <mmd-only|mmd-gan> --resume [same flags as before]
```

It continues from `$RUN/resume.pth`, which is written every 50 updates. If `mmd-only` was
interrupted, resume it first, then start `mmd-gan` with the normal command for that one group.

## 5. Evaluation (optional, only if the user asks)

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --eval-only [--no-wandb]
bash z-image/scripts/evaluate-baseline.sh --root $ROOT --steps 9,2    # teacher baselines, once per root
```

`--eval-only` skips setup and training. It installs the judge the first time (~65 GB), then runs
Qwen-Image-Bench (1000 prompts, Chinese prompts, the official judge) on `step_0000500.pth` of
each group. The table is written to `$ROOT/qib/runs/report.md`. If the evaluation stops, re-run
the same command: finished parts are kept.

## Handling failures

Each failure prints `ERROR:` and the log to read. The script never deletes anything, so fix the
cause and run the same command again.

| symptom | action |
|---|---|
| `CUDA out of memory` in a gate or training log | Add `--micro-batch 1`. The global batch stays 64 (the gradient accumulation doubles), so the experiment is unchanged. If it still runs out of memory at micro-batch 1, stop and report it. |
| `visible GPUs are busy` | Another job is using the cards. Wait, or choose idle cards with `CUDA_VISIBLE_DEVICES=0,1,2,3`. Never kill foreign jobs. |
| `--gpus N x --micro-batch M must divide the global batch 64` | Use a GPU count of 1, 2, 4 or 8, or set `--gpus`. |
| HTTP 401/403 during stage 2 | Gated model. Ask the user for a Hugging Face token whose account accepted the licence. Put it in `RDM/.hf_token`. |
| Stage 2, 3, 5 or 6 stops on a network error | Re-run. Downloads resume, and teacher tars are SHA-checked one by one. |
| `insufficient free disk` (stage 6) or the disk warning in the preflight | Free space or use a bigger `--root`. Do not pass `--allow-low-disk` unless the user agrees. |
| Stage 7 fails (`rank*.log`) | Re-run: finished rows are kept. If a rank log shows OOM, lower `--ref-budget-gib` (default 20). |
| `no attention backend` or a FlashAttention error on Hopper | Add `--attention native`. |
| `wandb` login error at the start | Add `--no-wandb`, or ask the user for a wandb key (`wandb login`). |
| `already holds a run` | Do not delete it. Ask the user whether to resume (`<group> --resume`) or move the old directory away. |
| Anything else | Stop and report the last 50 lines of the named log, plus the exact command you ran. |

## Done: what to report to the user

The experiments are complete when both run directories contain `step_0000500.pth`:

```bash
for e in mmd-only fixed; do d=$ROOT/zimage/work_dirs/zimage-2step-1024-dino-qwenvl-k512-b64-$e-from0
  echo "$e: $(wc -l < $d/train_log.jsonl) updates, $(ls $d | grep -c '^step_') checkpoints"; done
```

Report to the user:

1. The machine: GPU model and count.
2. The flags used, especially `--micro-batch` and the attention backend printed in the preflight.
3. The gate's peak memory per GPU.
4. Wall time per group.
5. The last `train_log.jsonl` row of each group.
6. The run directory paths.
7. The Qwen-Image-Bench table, if you ran the evaluation.

Do not compare the two groups' total generator losses. The objectives differ, so those numbers
are not comparable. Compare raw MMD², images and benchmark scores instead.

## Differences from the original H20 run

Mention these in your report:

- **Attention.** On non-Hopper GPUs the attention is PyTorch SDPA instead of FlashAttention 3.
  The mathematics is the same, but bf16 rounding differs, so losses do not match H20 bit for bit.
- **Micro-batch.** If you used `--micro-batch 1`, the gradient is accumulated over twice as many
  micro-batches. The global batch is still 64 and the objective is the same.
- **Checkpoints.** They stay in the run directory. Nothing is archived to the H20 object store.
