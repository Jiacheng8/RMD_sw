# 跑实验指南：sliding-window cache 漂移测量（E1）

在主实验（K128 · GAN，`configs/sw_lmmd_train_h100_2gpu_gan.yaml`，K=128，B=32，2000 步，HF 仓库 `jiachengcui888/w128_s32_2000steps_w-gan`）的 step 200、1000、2000 三个 checkpoint 上，测量窗口里缓存特征有多旧（漂移），以及这种旧对训练梯度影响多大。这个实验只做测量，不训练新模型：脚本按训练原样走几步，在每步之间读取和重新生成特征。不会改任何训练代码，也不会保存 checkpoint。

脚本是 `tools/measure_cache_drift.py`，用法和每个量的定义见文件开头的说明。

| 运行 | checkpoint | 输出 |
|---|---|---|
| 1 | `step_0000200.pth`（只有权重） | `outputs/cache_drift/step_0200.json` |
| 2 | `step_0001000.pth`（只有权重） | `outputs/cache_drift/step_1000.json` |
| 3 | `resume.pth`（step 2000，完整状态） | `outputs/cache_drift/step_2000.json` |
| 4 | `resume.pth`，行龄曲线 1–31 | `outputs/cache_drift/age_curve_step2000.json` |
| 汇总 | 上面 4 个 json + HF 训练日志 | `summary.md`、`summary.csv`、`age_curve_step2000.png`、`notes.md` |

- **机器**：2 张 H100 80 GB，脚本只支持不分片（`memory.shard: none`），4090 跑不了。
- **时间**（估计，没有实测）：环境和下载 30–60 分钟（Qwen3 context 重建约 10 分钟）；运行 1–3 每个 5–15 分钟；运行 4 约 10 分钟。
- **磁盘**：约 180 GB。模型权重约 26 GB，reference store 约 8.5 GB（解压时再要 8.5 GB），Qwen3 context 61 GB，三个 checkpoint 共 55 GB。

---

## 第 1 步：下载代码（分支 `cache-drift`）

代码在 GitHub 仓库 `Jiacheng8/RMD_sw` 的 **`cache-drift` 分支**上：
```bash
git clone -b cache-drift https://github.com/Jiacheng8/RMD_sw.git /root/RDM
```
已经 clone 过这个仓库的话：`cd /root/RDM && git fetch origin && git checkout cache-drift && git pull`。

确认当前在 `cache-drift` 分支上（`git branch --show-current`），并且 `tools/measure_cache_drift.py` 和本文件都在。

## 第 2 步：安装 Miniconda3

机器上已有 conda（`conda --version` 能输出版本号）就跳过。
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda init bash
```

## 第 3 步：Hugging Face 登录（人来做，不交给 agent）

需要一个 HF token，并满足两个条件：
- 能读私有数据集 `jiachengcui888/sw-rdm-reference-store`（reference store）；
- 已在网页上同意 `black-forest-labs/FLUX.2-dev` 的许可。

建议用 classic 的 Read token。在第 4 步装好 conda 环境后，自己执行 `conda run -n rdm hf auth login`；也可以在启动 agent 的终端里先 `export HF_TOKEN=...`。token 不要写进 prompt，也不要交给 agent。

## 第 4 步：把下面的 prompt 交给 coding agent

把 `/root/RDM` 和 `/root/rdm-sets` 换成实际目录。第 1 步没做也没关系，prompt 里已经包含 clone。

```
先准备代码：如果 /root/RDM 不存在，运行
  git clone -b cache-drift https://github.com/Jiacheng8/RMD_sw.git /root/RDM
如果已存在，运行
  cd /root/RDM && git fetch origin && git checkout cache-drift && git pull
并确认当前分支是 cache-drift。
然后按 /root/RDM/tools/Cache-Drift-README.md 里“agent 执行步骤”一节，从 A 做到 G，跑完 cache 漂移测量 E1，生成 outputs/cache_drift/ 下的全部报告文件。
代码目录 REPO=/root/RDM，数据目录 ROOT=/root/rdm-sets。

规则：
- 不修改任何代码和配置，不 commit，不 push。
- 不询问、不读取、不打印任何 token 或密钥；遇到权限错误就停下，把报错告诉我。
- 不删除任何文件。
- 任何一步失败，或者自动检查（sanity）没通过，都不要改参数重跑，把日志末尾和报错原样告诉我。
- 4 次测量按顺序一个一个跑，不要同时跑。

全部跑完后：把 outputs/cache_drift/summary.md 原样贴给我；补写 notes.md 最后的“Agent remarks”一节；
告诉我每次运行用了多久，以及 outputs/cache_drift/ 下有哪些文件。
```

---

## agent 执行步骤

下面的命令都在 bash 里执行，先设置变量：
```bash
export REPO=/root/RDM ROOT=/root/rdm-sets
[[ -d $REPO/.git ]] || git clone -b cache-drift https://github.com/Jiacheng8/RMD_sw.git $REPO
cd $REPO && git fetch origin && git checkout cache-drift && git pull
test "$(git branch --show-current)" = cache-drift && test -f tools/measure_cache_drift.py && echo "code OK"
```

### A. 环境、模型、reference store、Qwen3 context

```bash
bash scripts/new_machine.sh --root $ROOT --gpus 2 --config configs/sw_lmmd_train_h100_2gpu_gan.yaml \
    --no-train --skip-geneval --allow-low-disk
```
可以重复运行，已完成的阶段会自动跳过。结束时应该有 `$ROOT/env.sh` 和 `$ROOT/sw_lmmd/reference_store/bandwidths.json`。

如果第 1 阶段装完环境后，第 2 阶段报 401 或 403，说明还没登录 HF。停下来，告诉负责人去做第 3 步。

### B. 下载三个 checkpoint 和训练日志

```bash
source $ROOT/env.sh
mkdir -p $ROOT/e1/ckpt
conda run -n rdm --no-capture-output hf download jiachengcui888/w128_s32_2000steps_w-gan \
    step_0000200.pth step_0001000.pth resume.pth train_log.jsonl --local-dir $ROOT/e1/ckpt
ls -l $ROOT/e1/ckpt   # step_*.pth 各 15502249848 字节，resume.pth 23504415310 字节
```

### C. 公共设置（之后每个新终端都要先执行）

```bash
cd $REPO && source $ROOT/env.sh && export PYTHONPATH=$REPO:${PYTHONPATH:-} CUDA_VISIBLE_DEVICES=0,1
CFG=configs/sw_lmmd_train_h100_2gpu_gan.yaml
SET="--set reference_root=$ROOT/sw_lmmd/reference_store --set output_dir=$ROOT/sw_lmmd/work_dirs"
RUN="conda run -n rdm --no-capture-output torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tools/measure_cache_drift.py measure"
OUT=$REPO/outputs/cache_drift && mkdir -p $OUT/logs
nvidia-smi --query-gpu=index,name,memory.used --format=csv   # 两张 H100 都应该空闲
```

### D. 冒烟测试（约 5 分钟）

只测 1 个窗口，输出放在 `$ROOT/e1/smoke/`，不进 `outputs/`：
```bash
$RUN $CFG --ckpt $ROOT/e1/ckpt/step_0000200.pth --out $ROOT/e1/smoke/smoke.json --windows 1 $SET \
    2>&1 | tee $ROOT/e1/smoke/smoke.log
```
通过的标准：
- 日志里有 `[E1] ... step 200 | restored {... 'model': True ...} | critic OFF` 和一行 `[E1] window 1/1`；
- `$ROOT/e1/smoke/smoke.json` 存在，其中 `sanity.after_warmup.passed` 是 `true`，`age_counts` 是 `{"1": 32, "2": 32, "3": 32}`。

如果报 `not the main-run setup`、显存不足（OOM）或 `RuntimeError: retained rows are not ...`，就停下来报告，不要改 `micro_batch` 或其他配置。`micro_batch` 决定每次前向的 batch 组成，改了会影响 floor 的测量。

### E. 四次测量（按顺序）

```bash
$RUN $CFG --ckpt $ROOT/e1/ckpt/step_0000200.pth --out $OUT/step_0200.json $SET 2>&1 | tee $OUT/logs/step_0200.log
$RUN $CFG --ckpt $ROOT/e1/ckpt/step_0001000.pth --out $OUT/step_1000.json $SET 2>&1 | tee $OUT/logs/step_1000.log
$RUN $CFG --ckpt $ROOT/e1/ckpt/resume.pth --out $OUT/step_2000.json $SET 2>&1 | tee $OUT/logs/step_2000.log
$RUN $CFG --ckpt $ROOT/e1/ckpt/resume.pth --out $OUT/age_curve_step2000.json --long-ages 31 $SET \
    2>&1 | tee $OUT/logs/age_curve_step2000.log
```
每次运行都应该：
- 开头一行写明 checkpoint、step 和 critic 状态（200 和 1000 是 `critic OFF`，2000 是 `critic on`）；
- 前三次会打印 8 行 `[E1] window k/8`，第四次不打印窗口；
- 最后一行是 `[E1] -> <json> (x.x min)`。

`resume.pth` 跑两次是有意的：第 3 次测 8 个窗口，第 4 次测行龄曲线，两次都从同一个完整状态开始。

### F. 汇总

```bash
conda run -n rdm --no-capture-output python tools/measure_cache_drift.py summarize \
    --dir $OUT --train-log $ROOT/e1/ckpt/train_log.jsonl
```
会生成 `summary.md`、`summary.csv`、`age_curve_step2000.png` 和 `notes.md`，并在终端打印汇总表。

### G. 检查并补写 notes.md

`notes.md` 前几节是自动生成的，包括和协议的差异、每个 checkpoint 的状态、自动检查、和训练日志 `drift_age_k` 的对照。agent 要做三件事：
1. **确认自动检查都通过**：
   - 每个 checkpoint 热身后都是 `{'1': 32, '2': 32, '3': 32} (OK)`；
   - `floor (a) bitwise identical` 应该是 `True`，否则说明重新生成不确定，要在 remarks 里写明；
   - 有没有被标出 `<-` 的项（比如某个编码器比别的大 2 倍以上，或 floor 和漂移差不多大）。
2. **看图**：打开 `age_curve_step2000.png`，描述漂移是否随行龄增长，以及增长的形状（线性、变平或有跳变），和 floor 线的距离。
3. **在 "Agent remarks" 里写**：上面两点；日志里的异常（warning、某个窗口数值特别大）；和训练日志对照表里 ours/log 比值的大致范围。比值偏大是预期的：step 200 和 1000 用的是新建的优化器、没开 critic，而训练日志那边每 10 步刷新一次 cache。只描述事实，不要改其他几节。

---

## 输出说明（`outputs/cache_drift/`）

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.csv` | 每个 checkpoint 一行，再加一行三者平均。列：行龄 1/2/3 的漂移（σ_e，三个编码器各取中位数后再平均），floor（σ_e），三组梯度相对变化（%：cache vs now、floor、resampled window），cos(g_cache, g_now)，cos(g_resamp, g_now)。每格是 8 个窗口的均值 ± 标准差；带 † 的行是只有权重的 checkpoint |
| `step_XXXX.json` | 见下表 |
| `age_curve_step2000.json` / `.png` | 行龄 1–31：每个编码器漂移（σ_e）的中位数，floor (a)/(b)，图上的虚线是 floor |
| `notes.md` | 和协议的差异、自动检查、与训练日志的对照、agent 的说明 |
| `logs/*.log` | 每次运行的完整日志 |

`step_XXXX.json` 的结构：

| 键 | 内容 |
|---|---|
| `meta` | checkpoint 文件名、step、lr；优化器、critic、cache 是恢复的还是新建的；热身和测量时 critic 是否开启；GPU 型号和数量；commit；`uncommitted_changes`；每个编码器的 σ_e、β；`deviations`（和协议的差异） |
| `drift` | `{编码器 或 all: {行龄: {abs_sigma, rel_image, rel_joint: {mean, median}, rows}}}`，8 个窗口的行合在一起统计。abs_sigma = ‖Δφ‖/σ_e，rel_image = ‖Δφ‖/‖φ_now‖（只用图像块），rel_joint 是日志里 `drift_age_k` 的联合特征定义 |
| `floor` | `{a, b, c: {编码器 或 all: {mean, median}}}`，单位 ‖Δφ‖/σ_e。a：同一分块重新生成两次；b：训练 pass 1 的分块 vs refresh 的分块；c：训练真实的 pass 1 vs 按同样分块重新生成 |
| `grad` | `{cache_vs_now, floor, resample_vs_now: {编码器 或 all: {rel: {mean, std}, cos: {mean, std}, per_window: {rel: [8 个值], cos: [8 个值]}}}}`；`all` 是三个编码器拼起来的梯度 |
| `grad_total` | 只有 step 2000 有：同样三组，但损失用 L_MMD + λ·L_adv（critic 的 fake 侧包含缓存行），另有 λ 的统计 |
| `table` | 汇总表里每一列的 8 个窗口均值和标准差 |
| `sanity` | 热身后 96 行的行龄分布（应该是 1/2/3 各 32 行）、每个测量窗口的行龄分布、floor (a) 是否逐位相同及最大差值 |
| `per_window` | 每个窗口的原始记录 |
| `train_logs` | 热身和测量期间每次更新的 force、grad_norm、lr、λ |

## 常见问题

- **`not the main-run setup: ...`**：配置不是主实验的设置，比如用错了配置文件。检查 `CFG`，不要改脚本里的检查。
- **OOM**：两张卡上不能有别的进程。如果卡是空的仍然 OOM，就停下来报告。
- **`hf download` 报 401/403**：没登录或 token 权限不够，回到第 3 步。
- **`retained rows are not 3 whole active blocks ...`**：窗口组成检查没通过，测量不可信。停下来报告，附上日志。
- **端口被占用**：把 `RUN` 里的 `--master_port=29500` 换成别的端口。
