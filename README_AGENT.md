# Runbook: the two K512 Z-Image experiments on a fresh machine

This file is for a coding agent (Codex, Claude Code, …) asked to run two experiments on a new
Linux GPU machine, from an empty disk to two finished, evaluated training runs.

**This file may be all you have.** The code is not on the machine yet. You will fetch it in
[step 1](#1-get-the-code) from the **`zimage-exp`** branch of
<https://github.com/Jiacheng8/RMD_sw>. Do not use any other branch or fork.

Follow the steps in order. Every command is safe to re-run: finished work is detected and
skipped.

## What you are running

Two separate distillation runs of Z-Image-Turbo into a 2-step, 1024 px student, followed by a
Qwen-Image-Bench evaluation of both results against the teacher. Both runs start from the
original pretrained weights and train for **200 updates** with the same data, seed, window
(K=512, B=64) and MMD encoders (DINOv3-L + Qwen3-VL-4B).

| group | name on the command line | config (in the repo) | run directory name |
|---|---|---|---|
| 1 | `mmd-only` | `z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_mmd_only_from0.yaml` | `zimage-2step-1024-dino-qwenvl-k512-b64-mmd-only-from0` |
| 2 | `mmd-gan` | `z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_fixed_from0.yaml` | `zimage-2step-1024-dino-qwenvl-k512-b64-fixed-from0` |

Group 2 adds a fixed-weight (`5e-5`) GAN term on top of the same MMD objective. Group 2 must
**not** resume from Group 1. The script always starts each group fresh.

All the work goes through one script in the repo: `z-image/scripts/k512_two_groups.sh`.

## Rules

- **Use the `zimage-exp` branch of `https://github.com/Jiacheng8/RMD_sw`, and nothing else.**
  Do not switch branches, merge, rebase, or edit tracked files.
- **Do not edit the two YAML configs.** Their hyperparameters are the experiment. The script
  passes machine-specific values (paths, attention backend, micro-batch) and the 200-step
  length as `--set` overrides.
- **Do not use `ops/h20-run-k512-b64-fixed.sh`.** It only works on the original H20 server.
- **Never delete or overwrite a run directory, checkpoint or downloaded data** to get past an
  error. If the script refuses to start because a run directory already holds a run, stop and
  report it to the user.
- **Do not kill processes you did not start.** If the GPUs are busy, report it and wait.
- **Run everything inside `tmux`** (or `screen`). The full pipeline takes many hours.
- **Change one thing at a time.** The supported knobs are listed in
  [Handling failures](#handling-failures).

## 0. Requirements: check before anything else

Run these checks and stop if one fails:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv   # 4 GPUs, each >= 80 GB
nvidia-smi | head -4                     # driver must support CUDA 12.6 (driver >= 560)
df -h <data disk>                        # >= 650 GB free (see the budget below)
bash --version | head -1                 # >= 4.4
command -v git python3 tmux curl         # all present
command -v conda || echo "no conda"      # install Miniconda if missing (below)
```

What the checks must show:

- **GPUs.** The default and reference setup is **4 × A100 80GB**. Any GPU with 80 GB or more
  works (H100, H200, H20). 40 GB and 24 GB cards cannot hold the 6B student. The script trains
  on 4 GPUs and refuses to start with fewer. Only if the user gives you a machine with a
  different count, pass `--gpus N`, where N must divide 64 / micro-batch (2, 4, 8, …).
- **Disk, about 650 GB in total:**

  | item | size |
  |---|---|
  | model weights | 41 GB |
  | teacher tars | 162 GiB |
  | prompt contexts | 70 GB |
  | Group 1: 4 checkpoints × 24.6 GB + `resume.pth` (~26 GB, held twice while it is replaced) | ~155 GB |
  | Group 2: same as Group 1 | ~155 GB |
  | Qwen-Image-Bench judge env + Q-Judger weights | 65 GB |

- **Network.** Access to `github.com` and `huggingface.co`. Behind a Hugging Face mirror,
  export `HF_ENDPOINT=https://<mirror>` before running.

If a tool is missing:

- **git, tmux, curl, python3 or bash.** Install them with the system package manager, for
  example `sudo apt-get install -y git tmux curl python3 aria2`. If you have no sudo, ask the
  user.
- **conda.** Install Miniconda on the data disk:

  ```bash
  curl -fsSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p <data disk>/miniconda3 && eval "$(<data disk>/miniconda3/bin/conda shell.bash hook)"
  conda --version
  ```

  The conda environment itself (`rdm`, Python 3.12, torch 2.8 + CUDA 12.6) is created by the
  script in step 3. Do not create it yourself.

### Credentials

- **Hugging Face.** All datasets and models used here are public, so no token is needed
  normally. If a download fails with HTTP 401 or 403, ask the user for a read token. Save it as
  `RDM/.hf_token` after step 1 (git ignores that file), or export `HF_TOKEN`. Never print or
  commit it.
- **Weights & Biases.** The script stops at the start unless `wandb login` has been done on the
  machine. If the user did not give you a wandb key, add `--no-wandb` to **every**
  `k512_two_groups.sh` command below.

## 1. Get the code

Start a tmux session, then clone **exactly the `zimage-exp` branch**:

```bash
tmux new -s k512                                     # run everything below inside this session
cd <data disk>
git clone --branch zimage-exp --single-branch https://github.com/Jiacheng8/RMD_sw.git RDM
cd RDM
git branch --show-current                            # must print: zimage-exp
git log -1 --oneline                                 # note this commit for your report
ls z-image/scripts/k512_two_groups.sh README_AGENT.md   # both must exist
ROOT=<data disk>/rdm-sets                            # all data, models and runs go here
```

**Stop and report to the user** in any of these cases:

- `git branch --show-current` does not print `zimage-exp`.
- The clone fails.
- `z-image/scripts/k512_two_groups.sh` is missing.

Do not fall back to another branch.

If `RDM/` already exists from an earlier attempt, do not delete it. Run `cd RDM && git status`
and check:

- the branch is `zimage-exp`,
- the working tree has no local changes.

If both hold, `git pull --ff-only` and continue. Otherwise report to the user.

All commands below run from inside `RDM/`. If the session was lost, run
`tmux attach -t k512` and `cd <data disk>/RDM`, then set `ROOT` again.

## 2. Dry run (changes nothing)

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --dry-run
```

Check the `preflight` block it prints:

| line | expected |
|---|---|
| `groups` | `mmd-only mmd-gan (200 updates each)` |
| `eval` | `Qwen-Image-Bench on step 200 + teacher 9/2 steps` |
| `batching` | `4 GPUs x micro 2 x accum 8 = 64 rows per update` |
| `gpus` | the 4 cards you expect |
| `attention` | `native` on non-Hopper cards (A100), FA3 on H100/H20 |
| `disk` | no `ERROR:` |

It should then list stages 1–9 with the commands it would run. If it prints `ERROR:`, fix that
first (see [Handling failures](#handling-failures)).

## 3. Gate: set everything up, then 3 updates per group

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --gate
```

The gate does the following:

- **Setup.** It runs stages 1–7, which create the environment and build every input the
  experiments need.
- **Training check.** It trains each group for 3 updates in a separate directory
  (`$ROOT/zimage/work_dirs_gate`) and saves no checkpoint. These 3 updates hit every memory
  peak of the real run: the drift probe and, for Group 2, the GAN generator term.
- **No evaluation.**

The first time through, this command does almost all the setup and takes hours. Download speed
dominates; the teacher tars alone are 173 GB, about 2 h at 23 MB/s. These times were not
measured on A100:

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

## 4. The two experiments and their evaluation

Use **exactly the same flags that made the gate pass**, minus `--gate`:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT [--micro-batch 1] [--no-wandb]
```

This one command does the following:

1. Installs the Qwen-Image-Bench judge (about 65 GB, first time only), so that a broken judge
   shows up now and not after hours of training.
2. Trains Group 1 (`mmd-only`) for 200 updates, then Group 2 (`mmd-gan`) for 200 updates.
3. Evaluates both groups and the teacher with Qwen-Image-Bench (see
   [section 5](#5-evaluation-runs-by-default)).

Stages 1–7 are skipped because they are already done. If Group 1 fails, Group 2 does not start.
Do not pass `--steps`: 200 is the requested length. Use another value only if the user asks.

To run one group only, for example to run them on two machines (each still evaluates its own
group and the teacher):

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT mmd-only
bash z-image/scripts/k512_two_groups.sh --root $ROOT mmd-gan
```

Pass `--no-eval` only if the user asks you to skip evaluation.

### Monitoring

```bash
RUN=$ROOT/zimage/work_dirs/zimage-2step-1024-dino-qwenvl-k512-b64-mmd-only-from0   # or ...-fixed-from0
tail -1 $RUN/train_log.jsonl | cut -c1-400     # one JSON row per update: "step", losses, lr, mmd2
wc -l < $RUN/train_log.jsonl                   # updates done so far (200 at the end)
ls $RUN | grep step_                           # step_0000050.pth ... step_0000200.pth
tail -20 $ROOT/zimage/logs/train_mmd-only.log
nvidia-smi
```

How to read the logs:

- **Loss columns.** Raw MMD² is logged per student step and encoder (`s1/…/mmd2`, `s2/…/mmd2`)
  separately from the force surrogate, which is not an MMD and can be negative.
- **Group 2 GAN columns.** Group 2 also logs discriminator and generator GAN losses. Its GAN
  generator term starts at update 51.
- **Learning rate.** The generator LR warms up over 100 updates to `2.83e-5`, then follows a
  cosine decay to `1.415e-5` at update 200. The discriminator LR follows a cosine decay from
  `1e-5` to `5e-6` at update 200.

### Resuming after an interruption

If the machine or the session dies, continue the group that was running:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT <mmd-only|mmd-gan> --resume [same flags as before]
```

It continues from `$RUN/resume.pth`, which is written every 50 updates, and evaluates that group
when it finishes. If `mmd-only` was interrupted, resume it first, then start `mmd-gan` with the
normal command for that one group.

## 5. Evaluation (runs by default)

After training, the script runs Qwen-Image-Bench on every visible GPU for these runs:

- `step_0000200.pth` of each trained group, named `<exp_name>_s200`.
- The teacher Z-Image-Turbo at 9 steps (`turbo9`) and at 2 steps (`turbo2`). These are the
  baselines of the protocol. They are judged once per root and skipped on later runs.

The benchmark uses 1000 prompts, one 1024 px image each, the Chinese prompts and the official
judge. The script prints the comparison table and writes it to `$ROOT/qib/runs/report.md`.
Per-run outputs are in `$ROOT/qib/runs/<name>/`.

If evaluation stops after training finished, re-run it alone. Finished images and judge parts
are kept:

```bash
bash z-image/scripts/k512_two_groups.sh --root $ROOT --eval-only [--no-wandb]
```

`--no-baselines` judges only the two students.

## Handling failures

Each failure prints `ERROR:` and the log to read. The script never deletes anything, so fix the
cause and run the same command again.

| symptom | action |
|---|---|
| `git clone` fails, or the branch is not `zimage-exp` | Check network access to github.com and retry. Never use another branch. If it still fails, report it. |
| `CUDA out of memory` in a gate or training log | Add `--micro-batch 1`. The global batch stays 64 (the gradient accumulation doubles), so the experiment is unchanged. If it still runs out of memory at micro-batch 1, stop and report it. |
| `training needs 4 GPUs ... but only N are visible` | The machine has fewer than 4 GPUs, or `CUDA_VISIBLE_DEVICES` hides some. Report it. Use `--gpus N` only if the user agrees to a different count. |
| `visible GPUs are busy` | Another job is using the cards. Wait, or choose idle cards with `CUDA_VISIBLE_DEVICES=0,1,2,3`. Never kill foreign jobs. |
| HTTP 401/403 during a download | Ask the user for a Hugging Face read token. Put it in `RDM/.hf_token`. |
| Stage 2, 3, 5 or 6 stops on a network error | Re-run. Downloads resume, and teacher tars are SHA-checked one by one. |
| `insufficient free disk` (stage 6) or the disk error in the preflight | Free space or use a bigger `--root`. Do not pass `--allow-low-disk` unless the user agrees. |
| Stage 7 fails (`rank*.log`) | Re-run: finished rows are kept. If a rank log shows OOM, lower `--ref-budget-gib` (default 20). |
| `no attention backend` or a FlashAttention error on Hopper | Add `--attention native`. |
| judge setup fails (stage 9 setup, before training) | Re-run: downloads resume. If it keeps failing, report the error to the user. Only with their agreement, train with `--no-eval` and evaluate later with `--eval-only`. |
| `wandb` login error at the start | Add `--no-wandb`, or ask the user for a wandb key (`wandb login`). |
| `already holds a run` | Do not delete it. Ask the user whether to resume (`<group> --resume`) or move the old directory away. |
| Anything else | Stop and report the last 50 lines of the named log, plus the exact command you ran. |

## Done: what to report to the user

The experiments are complete when both conditions hold:

- both run directories contain `step_0000200.pth`;
- `$ROOT/qib/runs/report.md` lists both groups plus `turbo9` and `turbo2`.

Check the run directories with:

```bash
for e in mmd-only fixed; do d=$ROOT/zimage/work_dirs/zimage-2step-1024-dino-qwenvl-k512-b64-$e-from0
  echo "$e: $(wc -l < $d/train_log.jsonl) updates, $(ls $d | grep -c '^step_') checkpoints"; done
```

Report to the user:

1. The repo commit (`git -C RDM log -1 --oneline`, on `zimage-exp`).
2. The machine: GPU model and count.
3. The flags used, especially `--micro-batch` and the attention backend printed in the preflight.
4. The gate's peak memory per GPU.
5. Wall time per group, and for the evaluation.
6. The last `train_log.jsonl` row of each group.
7. The run directory paths.
8. The Qwen-Image-Bench table from `$ROOT/qib/runs/report.md`: both groups, `turbo9` and `turbo2`.

Do not compare the two groups' total generator losses. The objectives differ, so those numbers
are not comparable. Compare raw MMD², images and benchmark scores instead.

## Differences from the original H20 run

Mention these in your report:

- **Length.** 200 updates instead of 500. Both cosine LR schedules end at update 200. The
  100-update warmup and the GAN-G start at update 51 are unchanged.
- **Attention.** On non-Hopper GPUs the attention is PyTorch SDPA instead of FlashAttention 3.
  The mathematics is the same, but bf16 rounding differs, so losses do not match H20 bit for bit.
- **Micro-batch.** If you used `--micro-batch 1`, the gradient is accumulated over twice as many
  micro-batches. The global batch is still 64 and the objective is the same.
- **Checkpoints.** They stay in the run directory. Nothing is archived to the H20 object store.
