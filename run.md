# SW-LMMD 运行手册

从零到训练、评测的完整流程。命令都在仓库根目录 `/home/jiacheng/RDM` 下执行。

## 0. 总览

| 步骤 | 做什么 | 在哪跑 | 耗时 | 产出 |
|---|---|---|---|---|
| 环境 | conda 环境 `rdm` | 每台机器一次 | 看网速（约 10–12 GB） | conda env |
| 下载 | 权重、编码器、flux2 源码、COCO | 每台机器一次 | 看网速（约 64 GB） | `<资产目录>/env.sh` 等 |
| 预处理 01 | Qwen3 文本 context | 4090 机，1 卡 | 约 10 分钟（实测） | `qwen3_ctx_coco.npy`（61 GB） |
| 预处理 02 | 老师 4 步出图，每 prompt 24 张，PickScore 留 4 张 | 4090 机，6 卡 | 约 31 小时（实测） | `teacher_renders/`（331,132 张 PNG） |
| 预处理 03 | 10 个编码器提特征，组装参考库 | 4090 机，6 卡 | 0.5–1 小时（估计） | `reference_store/`（约 70 GB，含 context） |
| 训练 gate | 20 步，检查显存和速度 | 4090 或 H100 | 实测后才知道 | 日志、1 个 checkpoint |
| 正式训练 | 2000 步 | 4090 或 H100 | gate 后估算 | checkpoint |
| 评测 | GenEval + PickScore | 任意 1 卡 | — | 分数 |

**当前进度（2026-09-29 21:00）**：01 已完成；02 完成 79%（262,656 / 331,132），预计 9 月 30 日约 04:30 结束。预处理是在 tmux 会话 `16` 里用 `preprocess_all.sh` 启动的，02 结束后 **03 会自动开始**，不需要手动操作。

下一步要做的事：

1. 等预处理跑完，看到 `[store] OK`（见第 2 节）。
2. 在 4090 上跑 20 步 gate，再跑正式训练（第 3.2 节）。
3. （可选）把参考库拷到 H100，在 H100 上训练（第 3.3 节）。
4. 评测（第 4 节）。

---

## 1. 每台机器一次：环境和下载

```bash
cd /home/jiacheng/RDM
bash scripts/setup_env.sh                  # py3.12 + torch 2.8.0+cu126 + requirements + bitsandbytes 0.50.2
                                           # --dry-run 只打印计划；--force 重建环境
bash scripts/download_all.sh --root /data/thor/jiacheng/rdm-sets    # 约 64 GB，可断点续传
```

- `ae.safetensors` 在 FLUX.2-dev 仓库里，需要授权：先在 https://huggingface.co/black-forest-labs/FLUX.2-dev 上接受 license，执行 `hf auth login`，再重跑下载脚本。
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

### 3.3 在 H100 机器上

一次性准备：

1. 环境：`bash scripts/setup_env.sh`
2. 下载：
   - 全部下载：`bash scripts/download_all.sh --root <H100资产目录>`
   - 只下训练需要的编码器、FLUX 权重和 flux2 源码：
     ```bash
     conda activate rdm
     python scripts/fetch_prerequisites.py --root <H100资产目录> --group encoders --group flux --group flux2src
     ```
   - `ae.safetensors` 同样需要授权，见第 1 节。
3. 从 4090 机器拷参考库（约 70 GB）。**必须加 `-L`**：`qwen_context.npy` 是绝对路径软链接，`-L` 会把它指向的 61 GB 实际文件一起拷过去。
   ```bash
   rsync -aL --progress /data/thor/jiacheng/rdm-sets/sw_lmmd/reference_store/  <h100主机>:<路径>/reference_store/
   ```
   训练不需要 `teacher_renders/` 里的 PNG。
4. 修改 `configs/sw_lmmd_train_h100_2gpu.yaml` 里的两行：
   ```yaml
   reference_root: <路径>/reference_store
   output_dir: <路径>/work_dirs
   ```

启动：

```bash
cd /path/to/RDM
export ASSETS=<H100资产目录>              # 让启动脚本 source 到正确的 env.sh，否则找不到 FLUX 权重
tmux new -s train
GPUS=2 STEPS=20 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_train_h100_2gpu.yaml    # ① gate
mv <路径>/work_dirs/sw-lmmd-flux-h100-2gpu{,_gate}                                      # 挪走 gate 输出
GPUS=2 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_train_h100_2gpu.yaml             # ② 正式训练
```

如果机器上不止 2 张卡，在命令前加 `CUDA_VISIBLE_DEVICES=0,1`。

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
- **目前没有断点续训**：中断后重跑会从第 0 步开始。如果只想从某个 checkpoint 的权重接着训，可以在 yaml 里加 `load_from: <ckpt路径>`，但 Adam 状态和窗口位置会从头开始。

## 4. 评测

写一个评测配置，比如 `configs/eval_sw_lmmd.yaml`：

```yaml
extends: eval_flux.yaml
# 评测结果直接写到 output_dir 下（geneval/、geneval_summary.json），不会拼接 exp_name，所以给一个单独的目录
output_dir: /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/eval_4x4090_step2000
load_from: /data/thor/jiacheng/rdm-sets/sw_lmmd/work_dirs/sw-lmmd-flux-4x4090/step_0002000.pth
```

```bash
conda activate rdm && source /data/thor/jiacheng/rdm-sets/env.sh
python reproduce.py eval-flux --config configs/eval_sw_lmmd.yaml
```

- `load_from` 路径不存在时**只会打警告**，然后评测未训练的 base 模型，所以务必确认路径正确。
- GenEval 需要外部评分器：在配置里设 `geneval_repo: <djghosh13/geneval 的本地 clone>`。不设的话只出图不打分。详见 `docs/evaluating_released_checkpoints.md` 和 `docs/geneval_protocol.md`。
- `eval_flux.yaml` 的 `flux_ctx_len: 48` 和训练一致。论文里的 PickScore 主结果是在 ctx 232 下测的（`eval_flux_pspa.yaml`）。

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
- **拷参考库要用 `rsync -aL`**，否则 `qwen_context.npy` 软链接会失效。
- **直接用 torchrun 时的 `--set`**：顶层键名拼错会被静默忽略，所以改路径优先直接改 yaml。
- 预处理必须按 01 → 02 → 03 的顺序。`preprocess_all.sh` 会自动保证顺序，单独跑某一步时要自己注意。
