# 跑实验指南：sliding-window cache 漂移测量（E1）

这个实验测量主实验（K128 · GAN，`configs/sw_lmmd_train_h100_2gpu_gan.yaml`，K=128，B=32，2000 步，HF 仓库 `jiachengcui888/w128_s32_2000steps_w-gan`）step 200、1000、2000 三个 checkpoint 上的两件事：窗口里缓存的特征有多旧（漂移），以及这种旧对训练梯度的影响有多大。

它只做测量，不训练新模型：脚本按训练原样走几步，在每步之间读取和重新生成特征。不会改训练代码，也不会保存 checkpoint。

**一条命令就能跑完**：`bash tools/run_cache_drift.sh --root <数据目录>`。它会依次配环境、下载模型和数据、下载 checkpoint、做冒烟测试、跑 4 次测量、生成报告。可以重复运行，已完成的部分会自动跳过。各个量的定义见 `tools/measure_cache_drift.py` 开头的说明。

| 运行 | checkpoint | 输出（`outputs/cache_drift/`） |
|---|---|---|
| 1 | `step_0000200.pth`（只有权重） | `step_0200.json` |
| 2 | `step_0001000.pth`（只有权重） | `step_1000.json` |
| 3 | `resume.pth`（step 2000，完整状态） | `step_2000.json` |
| 4 | `resume.pth`，行龄曲线 1–31 | `age_curve_step2000.json` |
| 汇总 | 上面 4 个 json + HF 训练日志 | `summary.md`、`summary.csv`、`age_curve_step2000.png`、`notes.md` |

- **机器**：2 张 H100 80 GB。脚本只支持不分片（`memory.shard: none`），4090 等 24 GB 的卡跑不了，脚本会在开始测量前检查并报错。
- **时间**（估计，没有实测）：环境和下载 30–60 分钟（其中 Qwen3 context 重建约 10 分钟）；冒烟测试约 5 分钟；运行 1–3 每个 5–15 分钟；运行 4 约 10 分钟。合计 1.5–2 小时。
- **磁盘**：数据目录约 180 GB。其中模型权重约 26 GB，reference store 8.5 GB（解压时再要 8.5 GB），Qwen3 context 61 GB，三个 checkpoint 共 55 GB。另外 conda 环境约 12 GB，放在 conda 自己的目录里。

---

## 第 1 步：下载代码（分支 `cache-drift`）

代码在 GitHub 仓库 `Jiacheng8/RMD_sw` 的 **`cache-drift` 分支**上：
```bash
git clone -b cache-drift https://github.com/Jiacheng8/RMD_sw.git /root/RDM
```
已经 clone 过这个仓库的话：`cd /root/RDM && git fetch origin && git checkout cache-drift && git pull`。

## 第 2 步：安装 Miniconda3

机器上已有 conda（`conda --version` 能输出版本号）就跳过。
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda init bash
```
脚本在非交互的 shell 里也会去 `~/miniconda3`、`~/anaconda3`、`/opt/conda` 找 conda。

## 第 3 步：Hugging Face token（人来做，不交给 agent）

配环境阶段要下载两样需要权限的东西：门控的 FLUX.2 VAE，以及私有数据集 `jiachengcui888/sw-rdm-reference-store`（reference store）。所以需要一个 HF token，并满足两个条件：
- 能读私有数据集 `jiachengcui888/sw-rdm-reference-store`；
- 已在网页 https://huggingface.co/black-forest-labs/FLUX.2-dev 上同意许可。

建议用 classic 的 Read token。三个 checkpoint 所在的仓库是公开的，不需要 token。

任选一种方式给机器 token，**不要写进 prompt，也不要交给 agent**：
- 写进仓库里的文件（已被 git 忽略）：`echo hf_xxx > /root/RDM/.hf_token && chmod 600 /root/RDM/.hf_token`；
- 或者在启动 agent 的那个终端里先 `export HF_TOKEN=hf_xxx`，再启动 agent。agent 执行的命令会继承这个变量。

## 第 4 步：把下面的 prompt 交给 coding agent

把 `/root/RDM` 和 `/root/rdm-sets` 换成实际目录。第 1 步没做也没关系，prompt 里包含了 clone。

```
请跑 cache 漂移测量 E1，并生成报告。

1. 准备代码：如果 /root/RDM 不存在，运行
     git clone -b cache-drift https://github.com/Jiacheng8/RMD_sw.git /root/RDM
   如果已存在，运行
     cd /root/RDM && git fetch origin && git checkout cache-drift && git pull
   确认 git branch --show-current 输出 cache-drift。
2. 按 /root/RDM/tools/Cache-Drift-README.md 里“agent 执行步骤”一节，用数据目录 /root/rdm-sets 做完 A、B、C。

规则：
- 不修改任何代码和配置，不 commit，不 push。
- 不询问、不读取、不打印任何 token 或密钥；遇到 401/403 等权限错误就停下来告诉我。
- 不删除任何文件。
- 脚本报 ERROR 时，不要改参数或代码绕过它，把日志最后 50 行原样告诉我。
  只有在 README“常见问题”里写明可以直接重跑的情况下，才可以用同一条命令重跑。
- 同一时间只能有一个 run_cache_drift.sh 在运行。

全部完成后：把 outputs/cache_drift/summary.md 原样贴给我；告诉我 notes.md 里 Agent remarks 写了什么、
每次测量用了多久（看 logs/ 里每个 log 的最后一行），以及 outputs/cache_drift/ 下有哪些文件。
```

---

## agent 执行步骤

### A. 启动一键脚本（后台运行）

整个流程要 1.5–2 小时，比大多数 agent 单条命令的时间上限长，所以要放在后台运行，再查看日志：
```bash
cd /root/RDM && mkdir -p /root/rdm-sets
nohup bash tools/run_cache_drift.sh --root /root/rdm-sets > /root/rdm-sets/e1_run.log 2>&1 &
```
想先看一下计划、不做任何改动的话，可以先运行 `bash tools/run_cache_drift.sh --root /root/rdm-sets --dry-run`。

### B. 等它跑完

每隔几分钟看一次 `tail -n 30 /root/rdm-sets/e1_run.log`：
- 跑完时最后一行是 `==> done: .../summary.md, notes.md ...`；
- 出错时有一行 `ERROR: ...`，脚本会停下。处理方法见下面的“常见问题”。

脚本按下面 6 个阶段执行，日志里每个阶段开头是 `==> k/6`：

| 阶段 | 做什么 | 正常的样子 |
|---|---|---|
| 1/6 setup | `scripts/new_machine.sh --no-train --skip-geneval`：conda 环境 `rdm`、编码器和 FLUX.2 权重、reference store、重建 Qwen3 context | 最后检查 `env.sh`、`bandwidths.json` 和 Qwen3 context 都在 |
| 2/6 ckpt | 从公开仓库下载 3 个 checkpoint 和 `train_log.jsonl` 到 `<root>/e1/ckpt`，并核对文件大小 | 已下载的文件显示 `present` |
| 3/6 gpu | 检查 2 张卡，每张至少 75 GB 空闲 | `GPU 0: NVIDIA H100 ...` |
| 4/6 smoke | step 200 测 1 个窗口，结果放在 `<root>/e1/smoke/`，不进报告 | `smoke: step 200, ages {'1': 32, '2': 32, '3': 32}, ... -> OK` |
| 5/6 measurements | 依次跑 4 次测量，每次的日志在 `outputs/cache_drift/logs/` | 每次开头有 `[E1] ... step N | restored ... | critic OFF/on`；运行 1–3 各有 8 行 `[E1] window k/8`；结尾是 `[E1] -> <json> (x.x min)` |
| 6/6 report | `summarize`：生成 summary.md/csv、行龄曲线图和 notes.md，并在日志里打印汇总表 | 列出 `outputs/cache_drift/` 下的文件 |

step 200 和 1000 只有权重，所以显示 `critic OFF`；step 2000 从 `resume.pth` 完整恢复，显示 `critic on`。`resume.pth` 跑两次是有意的：运行 3 测 8 个窗口，运行 4 测行龄曲线，两次都从同一个完整状态开始。

### C. 检查结果，补写 notes.md

`notes.md` 前几节是自动生成的，包括和协议的差异、每个 checkpoint 的状态、自动检查、和训练日志 `drift_age_k` 的对照。agent 要做三件事：
1. **确认自动检查**：
   - 每个 checkpoint 热身后都是 `{'1': 32, '2': 32, '3': 32} (OK)`；
   - `floor (a) bitwise identical` 应该是 `True`，否则说明重新生成不确定，要在 remarks 里写明；
   - 有没有被标出 `<-` 的项（比如某个编码器比别的大 2 倍以上，或 floor 和漂移差不多大）。
2. **看图**：打开 `age_curve_step2000.png`，描述漂移是否随行龄增长，增长的形状（线性、变平或有跳变），以及和 floor 线的距离。
3. **只在最后的 "Agent remarks" 一节里写**（把占位文字替换掉，不要改其他几节）：
   - 上面两点；
   - 日志里的异常，比如 warning、某个窗口数值特别大；
   - 和训练日志对照表里 ours/log 比值的大致范围。step 200 和 1000 的比值偏大是预期的：这两个用的是新建的优化器、没开 critic，而训练日志那边每 10 步刷新一次 cache。

---

## 输出说明（`outputs/cache_drift/`）

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.csv` | 每个 checkpoint 一行，再加一行三者平均。列：行龄 1/2/3 的漂移（σ_e，三个编码器各取中位数后再平均），floor（σ_e），三组梯度相对变化（%：cache vs now、floor、resampled window），cos(g_cache, g_now)，cos(g_resamp, g_now)。每格是 8 个窗口的均值 ± 标准差；带 † 的行是只有权重的 checkpoint |
| `step_XXXX.json` | 见下表 |
| `age_curve_step2000.json` / `.png` | 行龄 1–31：每个编码器漂移（σ_e）的中位数，floor (a)/(b)，图上的虚线是 floor |
| `notes.md` | 和协议的差异、自动检查、与训练日志的对照、agent 的说明 |
| `logs/*.log` | 每次测量的完整日志 |

`step_XXXX.json` 的结构：

| 键 | 内容 |
|---|---|
| `meta` | checkpoint 文件名、step、lr；优化器、critic、cache 是恢复的还是新建的；热身和测量时 critic 是否开启；GPU 型号和数量；commit；`uncommitted_changes`；每个编码器的 σ_e 和 β；`deviations`（和协议的差异） |
| `drift` | `{编码器 或 all: {行龄: {abs_sigma, rel_image, rel_joint: {mean, median}, rows}}}`，8 个窗口的行合在一起统计。abs_sigma = ‖Δφ‖/σ_e；rel_image = ‖Δφ‖/‖φ_now‖（只用图像块）；rel_joint 是日志里 `drift_age_k` 的联合特征定义 |
| `floor` | `{a, b, c: {编码器 或 all: {mean, median}}}`，单位 ‖Δφ‖/σ_e。a：同一分块重新生成两次；b：训练 pass 1 的分块 vs refresh 的分块；c：训练真实的 pass 1 vs 按同样分块重新生成 |
| `grad` | `{cache_vs_now, floor, resample_vs_now: {编码器 或 all: {rel: {mean, std}, cos: {mean, std}, per_window: {rel: [8 个值], cos: [8 个值]}}}}`；`all` 是三个编码器拼起来的梯度 |
| `grad_total` | 只有 step 2000 有：同样三组，但损失用 L_MMD + λ·L_adv（critic 的 fake 侧包含缓存行），另有 λ 的统计 |
| `table` | 汇总表里每一列的 8 个窗口均值和标准差 |
| `sanity` | 热身后 96 行的行龄分布（应该是 1/2/3 各 32 行）、每个测量窗口的行龄分布、floor (a) 是否逐位相同及最大差值 |
| `per_window` | 每个窗口的原始记录 |
| `train_logs` | 热身和测量期间每次更新的 force、grad_norm、lr、λ |

## 常见问题

| 报错 | 原因 | 怎么办 |
|---|---|---|
| 阶段 1 报 401/403，或提示 `no Hugging Face token` / `cannot download the gated FLUX.2-dev VAE` | 没有 token，或 token 权限不够 | 停下来，请人做第 3 步；做完后用同一条命令重跑 |
| 阶段 1 网络或下载中断 | 网络问题 | 可以直接用同一条命令重跑，已完成的部分会跳过 |
| `GPU k has only ... MiB free` | 卡被别的进程占用，或者不是 80 GB 的卡 | 停下来报告，附上 `nvidia-smi` 的输出 |
| `... is N bytes, expected M` | checkpoint 没下载完整 | 可以直接重跑一次；再次出错就停下来报告 |
| `smoke test failed its checks`、`retained rows are not ...` | 窗口组成检查没通过，测量不可信 | 停下来报告，附上 `<root>/e1/smoke/smoke.log` 或对应的 log |
| `not the main-run setup: ...` | 配置不是主实验的设置 | 停下来报告；不要改脚本里的检查 |
| CUDA out of memory | 显存不足 | 停下来报告；不要改 `micro_batch`（它决定每次前向的 batch 组成，改了会影响 floor 的测量） |
| `Address already in use` | torchrun 的端口被占用 | 加 `--port 29511` 重跑 |
| 中途被打断（机器重启、进程被杀） | — | 用同一条命令重跑；已写出的 json 会跳过，正在跑的那次从头开始 |
