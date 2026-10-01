# SW-LMMD 运行手册

从零到训练、评测的完整流程。命令都在仓库根目录 `/home/jiacheng/RDM` 下执行。

## 0. 总览

| 步骤 | 做什么 | 在哪跑 | 耗时 | 产出 |
|---|---|---|---|---|
| 环境 | conda 环境 `rdm` | 每台机器一次 | 看网速（约 10–12 GB） | conda env |
| 下载 | 权重、编码器、flux2 源码、COCO | 每台机器一次 | 看网速（完整约 64 GB；新机器用 `--minimal`，约 26 GB 加参考库 8.5 GB） | `<资产目录>/env.sh` 等 |
| 预处理 01 | Qwen3 文本 context | 4090 机，1 卡 | 约 10 分钟（实测） | `qwen3_ctx_coco.npy`（61 GB） |
| 预处理 02 | 老师 4 步出图，每 prompt 24 张，PickScore 留 4 张 | 4090 机，6 卡 | 约 31 小时（实测） | `teacher_renders/`（331,132 张 PNG） |
| 预处理 03 | 10 个编码器提特征，组装参考库 | 4090 机，6 卡 | 0.5–1 小时（估计） | `reference_store/`（约 70 GB，含 context） |
| 训练 gate | 20 步，检查显存和速度（H100 上不需要单独跑，见 3.3） | 4090 或 H100 | 4090 每步 5.3 分钟（实测） | 日志、1 个 checkpoint |
| 正式训练 | 2000 步 | 4090 或 H100 | 4090 约 9 天；2×H100 估计 5–9 小时（以前几步的 s/step 为准）；可断点续训 | checkpoint、`resume.pth` |
| 评测 | GenEval + PickScore（`eval_checkpoint.sh`） | 任意 1 张空闲卡 | 每个 checkpoint 约 22 分钟（4090 实测） | `<out>/summary.json` |

**当前进度（2026-10-01）**：

- 预处理 01 → 02 → 03 已全部完成（9 月 30 日 05:00，`[store] OK`）。
- 4090 上的 20 步 gate 已跑通：约 20.7 GB/卡，每步约 5.3 分钟；算上缓存刷新，2000 步大约要 9 天。
- 新机器的整套流程（第 3.3 节）已推送到 GitHub。在这台机器上逐段实测过：全新 conda 装环境（版本锁定）、下载、模型文件核对、参考库下载/校验/解压、context 接入、GenEval 环境（含 H100 的源码编译路径）、断点续训、评测。H100 上的显存和速度只能开跑后看。
- 已发布的 s180 用 `eval_checkpoint.sh` 复现：GenEval 0.8238、PickScore 21.825（第 4 节）。
- 参考库（不含 61 GB 的 context）和 `coco_pairs.npz` 已上传到 HF 私有 dataset `jiachengcui888/sw-rdm-reference-store`（主要来源，2026-10-01 实测下载、校验和解压一共不到 3 分钟），Google Drive 的 `SW-RDM/` 里也有一份作为备用。

下一步要做的事：

1. **在新机器（H100）上训练：按第 3.3 节，一条命令从零跑到训练。**
2. 或者在这台 4090 上正式训练（第 3.2 节）。
3. 评测（第 4 节）：`bash scripts/eval_checkpoint.sh <checkpoint> --geneval-root <dir>`，一条命令出 GenEval 和 PickScore。

---

## 1. 每台机器一次：环境和下载

```bash
cd /home/jiacheng/RDM
bash scripts/setup_env.sh                  # py3.12 + torch 2.8.0+cu126 + requirements + bitsandbytes 0.50.2
                                           # --dry-run 只打印计划；--force 重建环境
bash scripts/download_all.sh --root /data/thor/jiacheng/rdm-sets    # 约 64 GB，可断点续传
```

- `ae.safetensors` 在 FLUX.2-dev 仓库里，需要授权：先在 https://huggingface.co/black-forest-labs/FLUX.2-dev 上接受 license，再执行 `conda run -n rdm hf auth login --token hf_xxx`（或者 `export HF_TOKEN=hf_xxx`），然后重跑下载脚本。
- 新机器建议直接用第 3.3 节的 `scripts/new_machine.sh`，它会把环境、下载、预处理和训练串起来。
- 开新 shell 后，先执行 `conda activate rdm && source <资产目录>/env.sh`。`scripts/train_sw_lmmd.sh` 会自己做这两步。
- 4090 这台机器上这一步**已经做完了**。

## 2. 预处理：在 4090 机器上生成参考库

```bash
tmux new -s prep
bash scripts/preprocess_all.sh             # 01 → 02 → 03 按顺序跑；中断后重跑同一条命令即可续上
```

- 也可以单独跑某一步：`bash scripts/preprocess_01_ctx.sh`、`preprocess_02_render.sh`、`preprocess_03_features.sh`。三步必须按顺序跑，03 一定要等 02 全部完成。
- 所有参数都在 `scripts/preprocess_config.sh`，也可以用环境变量覆盖。例如小规模试跑：
  `MAX_PROMPTS=64 SEEDS=1 NUM_GPUS=2 bash scripts/preprocess_all.sh`
- 各步骤的续跑行为：
  - 01 不能续跑：残缺文件会被删掉重算，大约 10 分钟。
  - 02 会跳过已完成的 chunk，只重算中断时那个 chunk，最多约 10 分钟。
  - 03 会跳过已完成的编码器。
- 查看正在跑的预处理：`tmux attach -t 16`，按 `Ctrl-b d` 退出查看（不会停止任务）。

查看进度：

```bash
W=/data/thor/jiacheng/rdm-sets/sw_lmmd
cat $W/teacher_renders/manifest_rank*.jsonl | wc -l   # 02 总进度，完成时是 331132
wc -l $W/teacher_renders/manifest_rank*.jsonl         # 各 rank 的进度，最慢的那个决定结束时间
grep -il "error\|traceback" $W/logs/*.log             # 有输出就说明出错了
ls $W/logs/                                           # 03 开始后会出现 03_extract_gpu*.log
tail -n 5 $W/logs/03_store.log                        # 出现 "[store] OK ..." 就是全部完成
nvidia-smi
```

产出（都在 `/data/thor/jiacheng/rdm-sets/sw_lmmd/` 下）：

- `qwen3_ctx_coco.npy`：(82783, 48, 7680) 的 Qwen3 context。
- `teacher_renders/{prompt:08d}_k{0..3}.png` 和 `manifest_rank*.jsonl`（记录每张图的 seed 和 PickScore）。
- `reference_store/`：训练只读这个目录。里面有：
  - `metadata.json`、`bandwidths.json`、`row_order.npy`、`prompt_ids.npy`、`text_features.npy`
  - `encoder_features/<编码器>.npy`
  - `qwen_context.npy`：**绝对路径的软链接**，指向上面的 `qwen3_ctx_coco.npy`

## 3. 训练

用 `scripts/train_sw_lmmd.sh` 启动，**不要用 `scripts/train.sh`**，那个启动的是 iRDM 训练。这个启动脚本会：

- source `$ASSETS/env.sh`（默认 `/data/thor/jiacheng/rdm-sets`），用 `conda run -n rdm` 调 torchrun；
- 启动前检查配置是否合法、B 能否整除卡数、参考库是否存在，并打印预估显存。

可以用环境变量调整：

| 变量 | 作用 |
|---|---|
| `GPUS` | 卡数（默认 2），必须整除 128 |
| `STEPS` | 覆盖训练步数 |
| `MICRO_BATCH` | 覆盖 micro-batch |
| `REFERENCE_ROOT` | 覆盖参考库路径 |
| `OUTPUT_DIR` | 覆盖输出目录（checkpoint 和日志） |
| `ASSETS` | 资产目录，决定 source 哪个 `env.sh` |
| `CONDA_ENV` | conda 环境名 |
| `MASTER_PORT` | 同一台机器上跑两个任务时改这个 |

### 3.1 选哪个配置

| 机器 | 配置 | GPUS | 设置 |
|---|---|---|---|
| 4 × RTX 4090（这台） | `configs/sw_lmmd_train_4x4090.yaml` | 4 | FSDP 4 路切分，fp32 主权重，bf16 all-gather，8-bit AdamW，micro 1 × accum 32，约 17–19 GB/卡（估计） |
| 2 × H100 80GB | `configs/sw_lmmd_train_h100_2gpu.yaml` | 2 | 不切分，fp32 主权重，8-bit AdamW，micro 8 × accum 8，约 66 GB/卡（估计） |

两个配置训练的是同一个目标：3 个编码器（dinov3_l、siglip2、aimv2_huge），K=1024，B=128，2000 步，只有显存相关的设置不同。

### 3.2 在 4090 机器上

先等预处理出现 `[store] OK`，并且 GPU 已经空出来。参考库不存在时，启动脚本的检查会直接拒绝启动。

```bash
cd /home/jiacheng/RDM
tmux new -s train

# ① 20 步 gate：确认显存够、速度多少、结束时会存一次 checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 STEPS=20 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_train_4x4090.yaml

# gate 通过后，把 gate 的输出目录挪走，否则正式训练的日志会接在 gate 日志后面
mv /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/sw-lmmd-flux-4x4090{,_gate}

# ② 正式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_train_4x4090.yaml
```

这台机器有 6 张卡，但只用 4 张：128 不能被 6 整除。

2026-09-30 的 gate 实测：每张卡约 20.7 GB，每步约 5.3 分钟，loss 正常下降，checkpoint 写出正常。

### 3.3 新机器（H100）：一条命令从零到训练

`scripts/new_machine.sh` 会按顺序做完：环境 → 下载 → 重建 context → 装 GenEval 评测环境 → 训练。默认使用 2 × H100 和 `configs/sw_lmmd_train_h100_2gpu.yaml`。训练完以后直接用 `eval_checkpoint.sh` 评测（第 7 步），不用再装别的。

**H100 清单（按顺序）：**

| 时间点 | 做什么 |
|---|---|
| 租之前 | 在这台 4090 上 push 代码（第 0 步）；HF 账号已在 FLUX.2-dev 页面接受 license，并备好 Read token |
| 选机器 | 2 × H100 **80 GB**、数据盘至少 350 GB、驱动 ≥ 525；容器的话 `--shm-size=16g` |
| 开机后 | 装 conda（系统盘小就装到数据盘）→ `git clone` → `echo hf_xxx > RDM/.hf_token && chmod 600 RDM/.hf_token`（`jiachengcui888` 的 Read token，见第 2 步）→ `tmux new -s train`。rclone 只是备用，可以不配 |
| 第一条命令 | `bash scripts/new_machine.sh --root <数据盘>/rdm-sets --dry-run`：检查都通过、没有 ERROR，再去掉 `--dry-run` 正式跑 |
| 前 30 分钟左右 | 第 2 阶段下载（参考库从 HF 下载约 2 分钟）；第 2 阶段结束时 8 个模型都应显示 `same` |
| 训练开始后 15 分钟内 | 看到 `step 1 ...`；`nvidia-smi` 显存在 80 GB 以下；记下每步秒数 |
| 第 100 步 | 日志出现 `resume state (step 100) -> ... in N s`，说明断点文件能写（N 是写盘秒数）；之后任何中断都用 `--resume` 接着跑 |
| 训练结束 | 两张卡并行评测（第 7 步）→ 拷走结果（第 9 步）→ 确认拷完再退租 |

**新机器上要事先准备好的**：

- **conda**（Miniconda 即可）。脚本不会自动安装它。如果机器的系统盘很小（租用机器常见），把 Miniconda 装在数据盘上，因为 rdm 环境本身要约 12 GB：`bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/<你>/miniconda3 && source /data/<你>/miniconda3/bin/activate`。不需要接受 Anaconda 的服务条款：两个环境都只用 conda-forge 和 NVIDIA 的频道。
- 用最新代码 `git clone` 下来的仓库（第 0 步）、HF token（第 2 步）。rclone 的 `gdrive` 远程只是备用（第 3 步）。

**租机器时确认这几项**（启动时脚本也会逐项检查）：2 × H100 **80 GB**；数据盘**至少 350 GB** 空闲；NVIDIA 驱动 ≥ 525；如果是容器，`/dev/shm` 至少 1 GB（启动容器时加 `--shm-size=16g` 或 `--ipc=host`）。

**0. 先在 4090 这台机器上把代码推上去。** 新机器是 `git clone` 下来的，没有 push 的改动它拿不到。现在的版本已经推送了（2026-10-01）。以后再改代码，在这台机器上这样推：这台机器的 HTTPS 方式没有 GitHub 凭据，`git push` 会认证失败，要走 SSH：

```bash
cd /home/jiacheng/RDM && git add -A && git commit -m "..." && git push git@github.com:Jiacheng8/RMD_sw.git main
```

**1. 推荐顺序（在新机器上）：**

```bash
git clone https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
echo hf_xxx > .hf_token && chmod 600 .hf_token   # 你的 HF Read token；.hf_token 被 git 忽略，不会被提交
# rclone config                   # 可选：只在 HF 下载不了参考库时用作备用，见第 3 步
tmux new -s train
bash scripts/new_machine.sh --root /data/<你>/rdm-sets --dry-run   # 先看计划和检查结果，什么都不改
bash scripts/new_machine.sh --root /data/<你>/rdm-sets
```

**2. Hugging Face（每台机器都要做）：**

- **用你自己的账号 `jiachengcui888` 的 token**：参考库在这个账号的私有 dataset 里，只有它的 token 能读。注意：4090 这台机器上存的 HF token 属于 `xinyuebi9`，不是你的账号，而且它现在下载 FLUX.2-dev 会被拒（403），这边能用只是因为文件早就缓存了。
- FLUX 的 VAE（black-forest-labs/FLUX.2-dev）需要授权。用 `jiachengcui888` 登录 https://huggingface.co/black-forest-labs/FLUX.2-dev 接受 license；每个 HF 账号只需要做一次。
- 启动时脚本会打印 token 属于哪个账号，并检查两件事：能不能下载 FLUX.2-dev、能不能读私有 dataset。FLUX.2-dev 下载不了就在开始前停下。
- 在 https://huggingface.co/settings/tokens 创建一个 **"Read"** 类型的 token（classic Read token 能读你自己的私有 repo）。如果用 fine-grained 类型的 token，要同时勾选 "Read access to contents of all public gated repos you can access" 和 "Read access to contents of all repos under your personal namespace"：同一个 token 还要从私有 dataset `jiachengcui888/sw-rdm-reference-store` 下载参考库。启动时脚本会检查它能不能读这个 dataset。
- 让新机器拿到 token（推荐第一种）：
  - **写进仓库根目录的 `.hf_token`**：`echo hf_xxx > ~/RDM/.hf_token && chmod 600 ~/RDM/.hf_token`。`new_machine.sh` 和 `download_all.sh` 会自动读取。这个文件在 `.gitignore` 里，`git add -A` 也不会把它提交上去。仓库是公开的，**千万不要把 token 直接写进任何脚本**。
  - `export HF_TOKEN=hf_xxx`：只对当前 shell 有效；设置了它就优先用它。
  - `conda run -n rdm hf auth login --token hf_xxx`：保存在这台机器上。
- 如果忘了登录，脚本会在第 2 阶段开始下载之前停下来，并给出这些步骤。登录后加 `--skip-env` 重跑即可。

**3. 参考库从哪里下载：**

- 参考库（`reference_store_noctx.tar` 9.1 GB，`coco_pairs.npz` 12 MB）**优先从 HF 私有 dataset `jiachengcui888/sw-rdm-reference-store` 下载**：这台机器上用你的 token 实测，下载、校验和解压一共不到 3 分钟；中断后可以续传；下载后校验 md5。
- HF 读不到时，才退回 rclone 从 Google Drive 下载（`gdrive:SW-RDM/`）。2026-10-01 实测：前一半约 30 MB/s，之后被限速到 0.2–0.5 MB/s，超过 2 小时。要用这条备用路线：安装 rclone（https://rclone.org/install/），`rclone config` 新建名为 `gdrive` 的 Google Drive 远程，用拥有 `SW-RDM/` 的那个 Google 账号授权；远程名不同就 `export SW_STORE_RCLONE=<名字>:SW-RDM`。
- `new_machine.sh` 启动时会用你的 token 检查 HF dataset 能不能读，同时检查 rclone 备用路线。两条都不通，就在开始之前停下来并说明原因。
- 如果 tar 包已经用别的方式拷过来了：`export SW_STORE_TAR=/path/reference_store_noctx.tar`。

**4. 五个阶段分别做什么：**

| 阶段 | 调用的脚本 | 做什么 | 耗时 |
|---|---|---|---|
| 1 环境 | `setup_env.sh` | conda 环境（torch 2.8.0+cu126、bitsandbytes） | 约 7 分钟（实测，网速快时） |
| 2 下载 | `download_all.sh --minimal` | 编码器、FLUX（klein-4B、VAE、Qwen3-4B）（只下训练实际用到的 3 个编码器）、评测用的 PickScore、flux2 源码（固定在 commit `50fe516`），约 26 GB；下载完会核对每个模型文件和参考结果用的是否一致；再从 HF 私有 dataset 下载参考库和 `coco_pairs.npz`（备用：rclone），校验 md5 后解压 | 这台机器上约 5–10 分钟（HF 实测 78 MB/s） |
| 3 预处理 | `preprocess_all-new-machine.sh` | 用 Qwen3 重新生成 61 GB 的 context（上传的参考库不含它），接入参考库并验证 | 约 10 分钟，1 张卡 |
| 4 GenEval | `setup_geneval.sh` | 评测环境装到 `<root>/geneval/`（H100 上 mmcv 自动源码编译），装完自检。失败只警告，不影响训练开始，之后可以单独重跑 | 5–10 分钟 |
| 5 训练 | `train_sw_lmmd.sh` | SW-LMMD 训练，2000 步；每 200 步存一个 checkpoint，每 100 步更新一次断点文件 `resume.pth` | 估计每步 8–15 秒，共 5–9 小时（以前几步的 s/step 为准） |

所有文件都放在 `--root` 下面：`env.sh`、`hf/`（权重）、`geneval/`（评测环境和检测器）、`sw_lmmd/{reference_store, qwen3_ctx_coco.npy, work_dirs/, logs/}`。pip 和 conda 的下载缓存也放在 `<root>/.cache/` 下。`rdm` 这个 conda 环境装在 conda 自己的 envs 目录（约 12 GB）。**磁盘预算约 315 GB**：权重约 26 GB，参考库 8.5 GB（解压时临时再多 8.5 GB），context 61 GB，GenEval 约 12 GB，缓存约 5 GB，checkpoint 每个 15.5 GB（H100 一共存 10 个，155 GB），`resume.pth` 23 GB（覆盖写入时临时 2 份）。

**为什么用 HF 而不是 Drive**：同一个 9.1 GB 的 tar 包，2026-10-01 在这台机器上实测，HF 不到 3 分钟（含校验和解压），Drive 超过 2 小时。训练本身只要 5–9 小时，用 Drive 等于白白多付 2 个多小时的 H100 钱。

**5. 常用选项：**

```bash
bash scripts/new_machine.sh --root <dir> --dry-run                   # 只打印计划，不做任何改动
bash scripts/new_machine.sh --root <dir> --gate                      # 只跑 20 步测试，输出到 work_dirs_gate/，不会和正式训练混在一起
bash scripts/new_machine.sh --root <dir> --skip-env --skip-download  # 从第 3 阶段继续
bash scripts/new_machine.sh --root <dir> --no-train                  # 只准备到可以开始训练为止
bash scripts/new_machine.sh --root <dir> --gpus 4 --config configs/sw_lmmd_train_4x4090.yaml   # 其他硬件
bash scripts/new_machine.sh --root <dir> --resume                    # 训练中断后接着跑（见第 8 步）
```

- 其他选项：`--steps N`、`--micro-batch N`、`--output-dir DIR`、`--env NAME`、`--hf-cache DIR`、`--full-download`（完整下载，不用 `--minimal`）、`--skip-geneval`（不装评测环境）、`--allow-low-disk`（磁盘放不下全部 checkpoint 也开跑，需要自己中途删旧的）、`--allow-hf-drift`（见下）。
- **模型文件一致性检查**：HF 上的模型仓库下载的是当天的最新版本。第 2 阶段下载完后，会逐个文件核对 8 个关键模型（klein-4B、VAE、Qwen3-4B、3 个训练编码器、PickScore 及其 processor）和这台机器上做出参考结果的版本是否一致（`assets/hf_revisions.json`，2026-10-01 记录时远端和本地完全一致）。只是仓库改了 README 不算变化；**模型文件真的变了就会在训练前停下**，确认能接受时再加 `--allow-hf-drift --skip-env` 重跑。
- 前 4 个阶段中断后重跑都会接着做。第 5 阶段：如果输出目录里已经有一次训练，脚本会拒绝从头开始（避免覆盖），提示用 `--resume` 接着跑，或者 `mv` 挪开 / 换 `--output-dir` 重新开始。
- 启动前检查（不满足就在开始前停下）：bash ≥ 4.4、conda 所在盘的空间、NVIDIA 驱动 ≥ 525、GPU 数量和显存（显存不够跑 H100 配置会提醒）、`/dev/shm` 大小（提醒）、HF token、rclone 远程。训练开始前还会检查：同一台机器上没有另一个训练在跑；磁盘放得下剩下所有 checkpoint 加 `resume.pth`。
- 依赖版本全部锁定：rdm 环境按 `requirements-lock.txt`、GenEval 环境按 `requirements-geneval-lock.txt` 安装，都是这台机器上验证过的版本。新机器不会因为某个包当天发了新版本而出问题。
- 整个运行过程的输出会记录到 `<root>/sw_lmmd/logs/new_machine_*.log`。
- **不需要单独跑 gate**：有了断点续训，正式训练的前 20 步就是 gate，而单独跑 gate 要额外花 30–40 分钟的 H100 时间。开跑后盯住这几件事：
  1. bootstrap 完成，开始打印 `step 1 | force ... | ... s`；
  2. `nvidia-smi` 显存稳定在 80 GB 以下（约 66 GB/卡是估算），前 20 步里第 10 步的漂移探针和第 20 步的缓存刷新都没有 OOM。如果 OOM，加 `--micro-batch 4` 重新开始，梯度不变；
  3. 用每步秒数乘 2000，估算总时长；
  4. 第 100 步出现 `resume state (step 100) -> .../resume.pth in N s`，说明存盘没问题。
- 仍然想先单独测一下的话：`--gate`（20 步，输出到 `work_dirs_gate/`）。

**6. 不用这个脚本、手动来做：**

1. `bash scripts/setup_env.sh`
2. `bash scripts/download_all.sh --root <dir> --minimal`
3. `ASSETS=<dir> bash scripts/preprocess_all-new-machine.sh`
4. `bash scripts/setup_geneval.sh --root <dir>/geneval --prefix <dir>/geneval/env`
5. `ASSETS=<dir> GPUS=2 REFERENCE_ROOT=<dir>/sw_lmmd/reference_store OUTPUT_DIR=<dir>/sw_lmmd/work_dirs bash scripts/train_sw_lmmd.sh configs/sw_lmmd_train_h100_2gpu.yaml`

如果机器上不止 2 张卡，在命令前加 `CUDA_VISIBLE_DEVICES=0,1`。

**7. 训练完以后评测（详见第 4 节）：**

一条命令评测这次训练的全部 checkpoint，同时测老师和已发布的 s180 作为同机基线。两张卡一直排满，最后输出一张对比表：

```bash
R=/data/<你>/rdm-sets; RUN=$R/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu
bash scripts/eval_run.sh $RUN --root $R --baselines
cat $RUN/eval_summary.md
```

- 10 个 checkpoint 加 2 个基线，按 4090 实测（22 分钟/个）推算，H100 估计每个 8–10 分钟，两张卡一共约 1 小时。80 GB 的卡可以加 `--per-gpu 2`，每张卡同时跑两个。
- 只评其中几个：`--only 2000,1800`；只评一个：`CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh $RUN/step_0002000.pth --root $R`。

- 第 4 阶段已经把 GenEval 装在 `<root>/geneval` 了，脚本会自己找到。如果第 4 阶段失败或者跳过了，先补装：`bash scripts/setup_geneval.sh --root <root>/geneval --prefix <root>/geneval/env`。
- 评测要空闲的卡，训练时两张卡都被占满，所以要等训练结束再评测。
- 结果在 checkpoint 旁边的 `eval_step_NNNNNNN/summary.json`。
- **跨机器的分数不能直接比**：bf16 计算在不同 GPU 上略有差异，第 4 节的参考分数是在 4090 上测的。想在 H100 上对比，就在 H100 上把基线也测一遍：老师用 `bash scripts/eval_checkpoint.sh base --root $R`（klein-4B 已经下载了）；已发布的 s180 先下载再测：`source $R/env.sh && conda run -n rdm hf download epfl-vita/flux2-klein-1step-rdm flux2_klein_1step_rdm_geallcoco_s180.pth`（先 source env.sh，15.5 GB 才会下到 `$R/hf` 而不是系统盘），再把它打印出的路径交给 `eval_checkpoint.sh`。另一个办法是把 checkpoint 拷回 4090 上评测。

**8. 训练中断了怎么办（断点续训）：**

- 训练每 100 步把完整状态覆盖写入 `<run dir>/resume.pth`：权重、8-bit AdamW 状态、窗口位置和缓存。写入方式是先写临时文件再改名，中途崩溃也不会损坏原来的文件。
- 不管是 SSH 断开、机器重启还是进程崩溃，先确认没有训练进程还在跑（`nvidia-smi` 里显存应该已经释放），再在 tmux 里执行：

  ```bash
  bash scripts/new_machine.sh --root /data/<你>/rdm-sets --resume
  ```

  如果原来开跑时还加了别的选项（`--gpus`、`--config`、`--output-dir`、`--steps`、`--micro-batch`），这里要带上同样的选项。`--resume` 会自动跳过第 1–4 阶段。
- 接着跑的部分和没中断时**逐位一致**（`tests/test_sw_lmmd_resume.py` 验证过，8-bit AdamW 在 GPU 上也验证过），所以最多损失 100 步，按估计的速度约 15–25 分钟。日志接着写在原来的 `train_log.jsonl` 里，checkpoint 编号也接着排。
- 不用 `new_machine.sh` 的话：`RESUME_FROM=<run dir>/resume.pth ASSETS=... bash scripts/train_sw_lmmd.sh <config>`。

**9. 还机器之前，先把结果拷走**：checkpoint、`train_log.jsonl`、各个 `eval_*/summary.json`，以及 `<root>/sw_lmmd/logs/`。一个 checkpoint 15.5 GB，一般只拷最好的一两个，加上所有的 summary 和日志。推荐经过 HF 私有 repo 中转，上传和之后在 4090 上下载都快。这一步要一个 **Write** token，用完可以在 HF 网站上删掉：

  ```bash
  read -rs -p "HF write token: " HF_TOKEN && export HF_TOKEN
  PYTHONNOUSERSITE=1 conda run -n rdm --no-capture-output hf upload jiachengcui888/sw-lmmd-h100-run $RUN . \
      --repo-type model --private --include "train_log.jsonl" --include "eval_*/summary.json" --include "step_0002000.pth"
  unset HF_TOKEN
  ```

  回到 4090 上用 `hf download jiachengcui888/sw-lmmd-h100-run --local-dir <目录>` 取回。两台机器之间能直接连通的话，用 `rsync -avP` 也可以。Drive（rclone）也能用，但往回下载会被限速。

### 3.4 gate 要看什么

- **不爆显存**：用 `nvidia-smi` 看峰值。
  - H100 如果 OOM：加 `MICRO_BATCH=4`（accum 变成 16），梯度不变，只是慢一些。
  - 4090 已经是 micro 1，没法再降，OOM 的话需要改方案。
- **速度**：日志里每一步的 `seconds`，乘以 2000 就是正式训练的大概时长。
  - 4090 如果显存还有余量，可以试 `MICRO_BATCH=2`（accum 16），PCIe 通信减半，梯度不变。
- **数值正常**：`force`、`grad_norm` 是有限值，`skipped` 是 false。
- **checkpoint**：结束时写出了 `step_0000020.pth`（15.5 GB），说明保存流程正常。gate 结束后可以删掉。
- 启动脚本打印的 "est. per-card" 没有算 autocast 的 bf16 权重缓存（H100 上大约多 7.8 GB）。以 yaml 注释里的估算为准，实际以 `nvidia-smi` 为准。

### 3.5 输出和监控

- 输出目录：`<output_dir>/<exp_name>/`
  - `output_dir` 默认是 `/data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs`
  - `exp_name` 是 `sw-lmmd-flux-4x4090` 或 `sw-lmmd-flux-h100-2gpu`
- `train_log.jsonl`（`tail -f` 查看）每步一行，字段含义：
  - `force`：训练用的 MMD force
  - `raw_mmd2`：监控用的窗口 MMD²
  - `grad_norm`：梯度范数
  - `cache_max_age`：缓存特征最老的年龄（步）
  - `seconds`：这一步用时
  - `drift_mean`、`drift_age_*`：缓存漂移，每 10 步测一次
  - `refreshed_rows`：每 20 个窗口整体刷新一次缓存时出现
- checkpoint：`step_XXXXXXX.pth`，内容是 `{"model", "step", "schedule", "world_size"}`，只含 fp32 权重，每个 15.5 GB。4090 每 250 步存一次（共 8 个），H100 每 200 步一次（共 10 个）。
- **断点续训**：H100 配置每 100 步把完整状态（权重、8-bit AdamW、窗口位置、缓存，约 23 GB）覆盖写入 `<run dir>/resume.pth`。中断后用 `bash scripts/new_machine.sh --root <root> --resume` 接着跑（见 3.3 第 8 步），结果和没中断时逐位一致。日志里每次写入会打一行 `resume state (step N) -> ... in N s`。4090 配置没有开这个功能（`save_resume: false`）。

## 4. 评测

**GenEval 是主要指标**；PickScore 只作参考，因为参考集是用同一个 PickScore 模型挑出来的，对它天然有利。

评测用到两个独立的 conda 环境：出图和 PickScore 在 `rdm` 环境里做；GenEval 打分在单独的 `geneval` 环境里做（它需要 torch 2.1 + mmdet 3，和 `rdm` 环境装不到一起）。`scripts/eval_checkpoint.sh` 把两边串起来，一条命令出结果。

### 4.1 安装 GenEval 环境（每台机器一次，约 5–10 分钟）

```bash
bash scripts/setup_geneval.sh --root /data/<你>/rdm-sets/geneval                # 推荐放在数据根目录下，4.2 的脚本会自动找到
bash scripts/setup_geneval.sh --root <dir> --prefix <dir>/env                   # 环境装到指定路径（/ 空间不够时用）
bash scripts/setup_geneval.sh --root <dir> --dry-run                            # 只打印计划
```

- 用 `new_machine.sh` 准备的机器，第 4 阶段已经装在 `<root>/geneval` 了，不用再装。
- 这台 4090 机器已经装好了：`/data/thor/jiacheng/rdm-sets/geneval_test`（环境在它下面的 `env/`），下面的命令都用它。
- 按 `docs/geneval_protocol.md` 的标准配置安装：
  - torch 2.1.2 + cu121、mmcv 2.1.0、mmdet 3.3.0；
  - Mask2Former Swin-S 检测器（mmdet 自带的配置，`c9d0c4f2` 版权重，下载后做 sha256 校验）；
  - open_clip ViT-L-14 颜色分类器；
  - 官方打分器 djghosh13/geneval，固定在 commit `af4902f`，并把它移植到 mmdet 3.x 的接口。
- **mmcv 会根据 GPU 自动选择安装方式**：
  - 4090、A100 等（计算能力 8.x 及以下）用预编译包；
  - H100（9.0）预编译包里没有它的 GPU 代码，会自动从源码编译（约 5–10 分钟，在环境内部用 conda 装 nvcc 12.1；系统 gcc 版本高于 12 时会自动装 gcc 11）。
  - 想装一个同时支持 4090 和 H100 的环境：`GENEVAL_CUDA_ARCH_LIST="8.9;9.0" bash scripts/setup_geneval.sh ... --build-mmcv`。
- 安装完会自检：先检查 mmcv 的 CUDA 算子在本机 GPU 上是否正确，再用一张已知内容的 COCO 照片完整跑一遍打分。
- 重跑会跳过已经完成的步骤；`--force` 会删掉环境重新装。
- 需要 NVIDIA GPU 和网络。空间：环境约 7 GB（源码编译时再加约 3 GB），检测器 0.3 GB，CLIP 1.7 GB。

### 4.2 一条命令评测一个 checkpoint

```bash
tmux new -s eval
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh \
    /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/sw-lmmd-flux-4x4090/step_0002000.pth \
    --geneval-root /data/thor/jiacheng/rdm-sets/geneval_test
```

脚本做三件事：

1. **出图 + PickScore**（`rdm` 环境，`reproduce.py eval-flux`）：
   - GenEval：553 条 prompt × 4 张 = 2212 张，固定种子，ctx_len 48（和训练一致），存到 `<out>/geneval/`；
   - Pick-a-Pic：499 条 prompt 各 1 张，ctx_len 232（论文里 PickScore 主结果用的设置），算平均 PickScore。
2. **GenEval 打分**（`geneval` 环境，`score_geneval.sh`）。
3. **汇总**：打印 GenEval 总分、6 项分数和 PickScore，写到 `<out>/summary.json`。

说明：

- **输出目录** `<out>`：默认是 `<checkpoint 所在目录>/eval_<checkpoint 名>/`，比如 `.../sw-lmmd-flux-4x4090/eval_step_0002000/`。评测 `base`、或者 HF 缓存里的 checkpoint 时，放到 `<root>/sw_lmmd/work_dirs/eval/` 下。用 `--out DIR` 可以改。
- **耗时**（4090 实测）：1 步的 checkpoint 22 分钟（出图 10 + 打分 11）；4 步的老师出图慢一倍多，估计约 30 分钟。在 tmux 里跑。
- **显存**：一张空闲的 24 GB 卡就够（实测峰值 21.2 GB）。开始前会检查目标卡的空闲显存，不够就直接停下，并列出每张卡的占用，**不会去挤正在训练的卡**。没设 `CUDA_VISIBLE_DEVICES` 时用 0 号卡。
- **checkpoint 加载不干净**（文件不存在、key 对不上）会直接报错退出，不会悄悄去评测未训练的 base 模型。
- **可以重跑**：已完成的阶段会跳过（比如打分中断了，重跑时只打分，不重新出图）；`--force` 全部重做。如果 `<out>` 里已经是另一个 checkpoint 或另一个步数的结果，脚本会拒绝运行，避免混在一起。
- `<out>` 里的其他文件：`eval_config.yaml`（自动生成的评测配置）、`eval_flux.log`、`flux_eval.json`、`geneval_results.jsonl`（每张图的判定和原因）、`geneval_score.log`。

其他用法：

```bash
bash scripts/eval_checkpoint.sh base --geneval-root <dir>               # klein-4B 老师（base 默认 4 步）
bash scripts/eval_checkpoint.sh base --steps 1 --geneval-root <dir>     # 未训练的 1 步 base，作为下限
bash scripts/eval_checkpoint.sh <ckpt> --geneval-root <dir> --dry-run   # 只打印计划和评测配置
bash scripts/eval_checkpoint.sh <ckpt> --root /data/<你>/rdm-sets       # 新机器：--root 和 new_machine.sh 用的一样；
                                                                        # GenEval 装在 <root>/geneval 时不用再给 --geneval-root
```

- 其他选项：`--out DIR`、`--steps N`、`--env NAME`（rdm 环境名）、`--no-geneval`（只出图和算 PickScore）。

**参考分数（2026-10-01，这台 4090 实测）：**

| 模型 | GenEval | single | two | counting | position | colors | color_attr | PickScore（ctx 232） | 文档参考值（GenEval / PickScore） |
|---|---|---|---|---|---|---|---|---|---|
| 已发布的 iRDM 学生 s180（1 步） | **0.8238** | 99.4 | 92.4 | 75.0 | 66.0 | 91.8 | 69.8 | **21.825** | 模型卡 0.8258 / 21.817；文档的 H100 复现 0.830 / 21.83 |
| klein-4B 老师（4 步） | **0.8001** | 99.4 | 87.9 | 80.6 | 58.8 | 90.2 | 63.3 | 未测 | 0.7944 / 21.848 |

- s180 这一行是 `eval_checkpoint.sh` 的完整结果（`/data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/eval/flux2_klein_1step_rdm_geallcoco_s180/summary.json`）。它出的 2212 张 GenEval 图和之前单独渲染的标准版本逐字节相同。
- 老师这一行的 GenEval 来自之前同一套代码路径的标准渲染。老师还没有用 `eval_checkpoint.sh` 完整跑过，所以 PickScore 还空着；空出一张卡后跑：`CUDA_VISIBLE_DEVICES=<i> bash scripts/eval_checkpoint.sh base --geneval-root /data/thor/jiacheng/rdm-sets/geneval_test`。
- 和文档的差距在 ±0.006 以内，和文档自己的复现差距（0.830 和 0.826）是同一个量级。原因是不同 GPU 上 bf16 计算略有不同，生成的图片会有个别差异。
- 所以**比较不同模型时，要用同一台机器、同一套流程渲染并打分**，不要直接拿文档里的数字去比。
- 已发布 s180 的 checkpoint 已经下载到 HF 缓存里：`/data/hulk/jiacheng/cache/hub/models--epfl-vita--flux2-klein-1step-rdm/`。

### 4.3 评测一次训练的所有 checkpoint：`eval_run.sh`

```bash
bash scripts/eval_run.sh <run dir> --root <root> [--baselines] [--only 2000,1800] [--gpus 0,1] [--per-gpu 2] [--dry-run]
```

- 找出 `<run dir>` 里所有 `step_*.pth`（不包括 `resume.pth`），从最新的开始排队。每个任务就是一次 `eval_checkpoint.sh`，单独占一张卡。哪张卡空了就接着跑下一个，直到全部跑完。
- `--baselines`：再加上 klein-4B 老师（4 步）和已发布的 s180（公开 repo，会自动下载 15.5 GB），用来做同一台机器上的对比，放在最后跑。
- `--gpus` 默认用所有能看到的卡（或 `CUDA_VISIBLE_DEVICES`）。`--per-gpu 2` 只允许 48 GB 以上的卡。
- 结果都放在 run dir 里，方便一起拷走：
  - 每个任务的结果：`eval_step_NNNNNNN/`、`eval_teacher_4step/`、`eval_s180_release/`；
  - 每个任务的日志：`eval_logs/<名字>.log`；
  - 汇总表：`eval_summary.md`（标出 GenEval 最高的 checkpoint）和 `eval_summary.json`。
- 某个任务失败不影响其他任务，最后会列出失败的任务和日志路径。用同一条命令重跑，已经完成的阶段会跳过。
- Ctrl-C 会停掉所有正在跑的评测，包括它们的子进程。
- `--root`、`--geneval-root`、`--env`、`--no-geneval`、`--force` 会原样传给 `eval_checkpoint.sh`。

### 4.4 不用脚本、手动分步（和 4.2 等价）

1. 写一个评测配置，比如 `configs/eval_sw_lmmd.yaml`：

   ```yaml
   extends: eval_flux.yaml
   output_dir: /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/eval_4x4090_step2000   # 结果直接写在这里，不会拼接 exp_name
   load_from: /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/sw-lmmd-flux-4x4090/step_0002000.pth
   strict_load: true         # 加载不干净就报错（默认只打警告，然后评测未训练的 base）
   pickscore_ctx_len: 232    # PickScore 用 ctx 232；GenEval 仍然是 flux_ctx_len 48
   geneval_repo: null        # 保持 null：这里只出图，打分用第 2 步
   autocast_cache: false     # 4090 必需：不缓存 bf16 权重副本，结果完全一样
   vae_decode_batch: 4       # 4090 必需：VAE 分块解码
   ```

2. 出图 + PickScore：

   ```bash
   conda activate rdm && source /data/thor/jiacheng/rdm-sets/env.sh
   CUDA_VISIBLE_DEVICES=0 python reproduce.py eval-flux --config configs/eval_sw_lmmd.yaml
   ```

3. GenEval 打分：

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/score_geneval.sh <output_dir>/geneval --root /data/thor/jiacheng/rdm-sets/geneval_test
   ```

   写出 `<output_dir>/geneval_results.jsonl` 和 `geneval_results.jsonl.summary.json`；总分 `overall` 是 6 项任务正确率的**简单平均**，这是 GenEval 的标准定义。

- **不要设置 `geneval_repo`**：那个快捷方式会在 rdm 环境里调用打分器（rdm 环境里没有 mmdet），而且用的不是标准的检测器配置，颜色类分数会有偏差。

## 5. 测试

```bash
conda activate rdm && source /data/thor/jiacheng/rdm-sets/env.sh
CUDA_VISIBLE_DEVICES="" python -m pytest tests -q     # 只用 CPU，约 1 分钟，不占 GPU
```

不 source `env.sh` 的话，`tests/test_sw_lmmd_fsdp.py` 里的 FSDP 测试会因为找不到 flux2 源码而跳过。

## 6. 注意事项

- **磁盘**：`/`（包括 `/home`）只剩约 37 GB（2026-09-29）。checkpoint 和参考库都要放在 `/data/thor`，新配置已经这么设置了。
- **卡数必须整除 128**：可以是 1/2/4/8/16/32。
- **不要用旧配置** `sw_lmmd_h100_2gpu.yaml` 和 `sw_lmmd_debug_4x4090.yaml`：它们把参数存成 bf16，在 lr 2.83e-6 下 klein-4B 有 96% 的权重一步也不会更新。`train_sw_lmmd.sh` 头部注释里的示例还是这两个旧配置，请用 `sw_lmmd_train_*.yaml`。
- **`env.sh`**：`HF_HOME` 必须是 hub 缓存的上一级目录（FLUX 权重只按 `HF_HOME` 查找）。换机器时记得设 `ASSETS`：启动脚本找不到 `env.sh` 时会静默跳过，要到加载模型时才报错。
- **参考库里的 `qwen_context.npy` 是软链接**，指向 61 GB 的 `qwen3_ctx_coco.npy`。上传到 HF / Drive 的版本不含它，新机器用 `preprocess_all-new-machine.sh` 重新生成。如果要在机器之间直接拷贝完整的参考库，要用 `rsync -aL`，否则软链接会失效。
- **新机器必须用同一个 `coco_pairs.npz`**：context 的第 i 行要对应参考库里第 i 个 prompt。`download_all.sh` 下载的就是这份文件，会用 md5 校验。
- **直接用 torchrun 时的 `--set`**：顶层键名拼错会被静默忽略，所以改路径优先直接改 yaml。
- 预处理必须按 01 → 02 → 03 的顺序。`preprocess_all.sh` 会自动保证顺序，单独跑某一步时要自己注意。
