# Z-Image 2 步蒸馏：上手指南（给用户和 coding agent）

把这个文件交给 coding agent（Codex、Claude Code 等），或者自己照着做，就能在一台新租的 GPU 机器上，从零跑完本项目的任何一个任务。所有命令都是一条命令完成配环境、下载、训练、评测。

> **给 agent 的话**：先通读第 0 节的规则，再按用户要做的任务走对应的小节。不确定的事情问用户，不要猜。

---

## 0. 给 agent 的规则（必读）

1. **不要改代码。** 训练、评测的实现都已经定稿；只运行 `z-image/scripts/` 下的脚本。脚本报错时，把报错和日志最后 30 行给用户看，不要自己改 `.py` / `.sh`。
2. **长任务一律放进 tmux 后台。** 这些任务要跑几小时到几天，远超 agent 单条命令的超时时间。用这个格式启动，然后定期看日志：
   ```bash
   tmux new -d -s <名字> "bash <命令> 2>&1 | tee /root/<名字>.out"
   tail -n 30 /root/<名字>.out          # 看进度
   tmux ls                              # 还在跑吗
   ```
   不要在前台直接跑；也不要因为“好像卡住了”就杀进程，先看日志（见第 7 节）。
3. **需要用户提供的东西，开始前先问齐**（见第 2 节）：用哪个任务、`--root` 放哪块盘、wandb key。agent 的 shell 一般不是交互终端，脚本不会弹出输入框，所以 key 要用环境变量传进去。
4. **不要删除** `--root` 下的任何东西（权重、contexts、checkpoint、评测结果），除非用户明确要求。
5. **一个 `--root` 同一时间只跑一个任务**（它们共用 GPU）。
6. 失败后 **先重跑同一条命令**：所有阶段都会跳过已完成的部分，训练用 `--resume` 接着跑。
7. 每个任务结束时，把脚本最后打印的**结果表**原样告诉用户。

---

## 1. 这个项目在做什么（一段话）

把 8 步的文生图模型 Z-Image-Turbo 蒸馏成 2 步，1024×1024。我们的方法（**ours**，SW-LMMD：每个 student step 一个 MMD + 一个 GAN critic）和两个基线比较：**RDM**（原始 RDM，local-to-global 匹配）和 **DMD2**。三者都用 **Qwen-Image-Bench**（1000 个 prompt，Qwen3.6-27B 当裁判，0–100 分）评测，并在 **相同 GPU 时间** 下比较。代码在仓库的 `z-image/` 目录，分支 `zimage`。

---

## 2. 开始前要确定的事（agent：先问用户）

| 问题 | 默认 / 说明 |
|---|---|
| 做哪个任务？ | 见第 4 节的任务表 |
| 机器是什么卡、几张？ | 需要 **80 GB** 的卡（A100 80GB / H100 / H200）。DMD2 至少 4 张 |
| `--root` 放在哪？ | 必须在**空间最大的那块盘**上（见 3.1）。下文用 `/root/rdm-sets` 举例 |
| 要不要 wandb？ | 要的话需要一个 **sw-lmmd team 成员**的 API key（https://wandb.ai/authorize）。不要也能跑 |
| HF token？ | **不需要**，用到的模型和数据都是公开的 |

---

## 3. 新机器准备（每台机器做一次）

### 3.1 看盘
```bash
nvidia-smi                      # 卡的型号、数量、是否空闲
df -h / /root /workspace 2>/dev/null
```
选可用空间最大的盘放 `--root`，也把代码 clone 在那块盘上。如果大盘是 `/workspace`，下文所有 `/root/...` 都换成 `/workspace/...`。

需要的空间：

| 任务 | 磁盘 |
|---|---|
| 冒烟测试 | ~300 GB 峰值 |
| ours 一次正式训练 | ~330 GB |
| ours + RDM + DMD2 都在同一个 root | ~650–700 GB |
| 只评测 teacher | ~100 GB |

机器需要有 `conda` 和 NVIDIA 驱动（常见的 GPU 云镜像都有）。环境由脚本自己装。

### 3.2 拉代码
```bash
cd /root                        # 或大盘所在目录
git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM
cd RDM && git log --oneline -1  # 记下 commit，报告问题时带上
```

### 3.3 wandb（可选）
推荐用环境变量，写进 tmux 启动命令里（tmux 会话不一定继承你当前 shell 的变量）：
```bash
tmux new -d -s smoke "export WANDB_API_KEY=<key>; bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets 2>&1 | tee /root/smoke.out"
```
或者在环境装好之后（第一个任务跑过准备阶段）登录一次，之后一直有效：`conda run -n rdm wandb login <key>`。
- 不给 key：脚本照常跑，只是不记 wandb（日志里会写 `wandb off`）。
- 结果在 `wandb.ai/sw-lmmd/zimage-distill`（冒烟测试在 `zimage-distill-smoke`）。
- key 错误或账号不在 team 里：打印 `[wandb] not logging: ...`，训练不受影响。

---

## 4. 任务表

| 任务 | 推荐机器 | 大概时间 | 小节 |
|---|---|---|---|
| A. 冒烟测试（三个方法各跑 10 步 + 评 10 个 prompt） | 4×A100 80GB | ~半天 | 5.A |
| B. 评测 teacher（Z-Image-Turbo 9 步和 2 步） | 4×A100 | ~2–2.5 h（含准备） | 5.B |
| C. 正式训练 ours | 8×A100/H100 | ~15–20 h + 评测 [估计] | 5.C |
| D. 正式训练 RDM 基线 | 同 C 的卡数 | 和 ours 相同的 GPU 时间 + 评测 | 5.D |
| E. 正式训练 DMD2 基线 | 8 卡（至少 4 卡） | 和 ours 相同的 GPU 时间 + 评测 | 5.E |

**顺序很重要：C 必须在 D、E 之前**，D 和 E 会读取 ours 的训练日志，花同样多的 GPU 时间。D、E 最好在同一个 `--root`（同一台机器或同一块盘）上跑。

每个启动脚本都支持：

| 参数 | 作用 |
|---|---|
| `--dry-run` | 只检查并打印计划，什么都不改 |
| `--resume` | 中断后从最近的 `resume.pth` 接着训练 |
| `--eval-only` | 只评测已有的 checkpoint |
| `--no-eval` | 只训练，不评测 |
| `--root DIR` | 数据根目录（默认 `/root/rdm-sets`） |
| `-h` | 完整说明 |

---

## 5. 各任务的命令

下面所有命令都在仓库根目录 `RDM/` 下执行。第一次跑的任务会先花 **1–2 小时**准备（装环境、下载约 37 GB 权重、55 GB 裁判模型、reference pack、重建 70 GB 的 Qwen3 contexts），之后同一个 `--root` 上的任务会跳过这些。

### 5.A 冒烟测试（4 卡）
在花大钱跑正式实验前，确认三个方法都能走通“训练 → 存模型 → 出图 → 打分”。
```bash
tmux new -d -s smoke "bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets 2>&1 | tee /root/smoke.out"
```
依次跑四个阶段：`ours` → `rdm`（每步 128 行）→ `rdm-oom`（测每步 10240 行会不会爆显存，几分钟）→ `dmd2`。
- 进度：`tail -f /root/rdm-sets/zimage/smoke/logs/<阶段>.log`
- 结束时打印汇总表：每个阶段 PASS/FAIL、耗时、s/step、峰值显存、10 个 prompt 的总分。**10 个 prompt 的分数只用来确认流程通了，不能拿来比较模型。**
- 只重跑某一阶段：加 `--only dmd2`（可选 `ours,rdm,rdm-oom,dmd2`）。

### 5.B 评测 teacher
```bash
tmux new -d -s teacher "bash z-image/scripts/evaluate-baseline.sh --root /root/rdm-sets --steps 9,2 2>&1 | tee /root/teacher.out"
```
- 9 = 官方设置；2 = 不蒸馏直接跑 2 步。结果在 `/root/rdm-sets/qib/runs/turbo9`、`turbo2`，表在 `qib/runs/report.md`。
- 4×A100 上：出图 turbo9 约 4.4 s/张（实测）；整体两个模型约 1–1.5 h（不含首次准备）。
- 跑过任务 C 的 root 不用再做：任务 C 结束时已经评过 teacher。

### 5.C 正式训练 ours
按卡数选脚本：`ours_two-cards.sh` / `ours_four-cards.sh` / `ours_eight-cards.sh`。
```bash
bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --dry-run      # 先检查
tmux new -d -s ours "bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/ours.out"
```
- 2000 步，每 500 步存一个 checkpoint（每个 24.6 GB）。2×A100 实测 93 s/步（约 52 h）；8 卡估计 25–30 s/步（约 15–20 h）。
- 想先测速度和显存：`--gate`（准备 + 20 步，不评测）。
- 结束后自动评测最后一个 checkpoint，加上 teacher 9 步、2 步，表在 `<root>/zimage/work_dirs/zimage-2step-1024/qwen_image_bench.md`。
- 中断了：同一条命令加 `--resume`。

### 5.D 正式训练 RDM 基线
```bash
bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --dry-run
tmux new -d -s rdm "bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/rdm.out"
```
- 每步 10240 行（iRDM 的设置）。自动读 ours 的 run（`<root>/zimage/work_dirs/zimage-2step-1024`）算出 GPU 预算，先实测几步的速度，再定总步数（写进 `plan.json`）。一步要好几个小时，所以总共只有几步，这是设计如此。
- ours 不在这个 root 上时，用 `--match-gpu-hours <ours 的 GPU 小时数>` 代替。
- 想先确认 10240 行不爆显存（几分钟）：`--oom-probe`。
- 输出：`<root>/zimage/work_dirs/zimage-2step-1024-rdm-global-N10240/`。

### 5.E 正式训练 DMD2 基线（至少 4 卡）
```bash
bash z-image/scripts/dmd2_eight-cards.sh --root /root/rdm-sets --latents-hf jiachengcui888/zimage-dmd2-latents --dry-run
tmux new -d -s dmd2 "bash z-image/scripts/dmd2_eight-cards.sh --root /root/rdm-sets --latents-hf jiachengcui888/zimage-dmd2-latents 2>&1 | tee /root/dmd2.out"
```
- `--latents-hf`：直接下载已经编码好的真实数据（约 18 GB），省掉 20–30 分钟编码。
- 会额外下载未蒸馏的 Z-Image（约 25 GB）。同样自动对齐 ours 的 GPU 时间（或 `--match-gpu-hours H`）。
- 输出：`<root>/zimage/work_dirs/zimage-2step-1024-dmd2/`。

---

## 6. 结果在哪

| 内容 | 位置 |
|---|---|
| 训练日志（每步一行 JSON） | `<run 目录>/train_log.jsonl` |
| checkpoint | `<run 目录>/step_*.pth`（`resume.pth` 用于续跑） |
| 评测对比表 | `<run 目录>/qwen_image_bench.md` |
| 每次评测的图和分数 | `<root>/qib/runs/<名字>/`（`images/`、`input_bench_scores.json`） |
| 各阶段日志 | `<root>/zimage/logs/`、`<root>/qib/runs/<名字>/logs/` |
| wandb | `wandb.ai/sw-lmmd/zimage-distill`（group：ours / rdm-global / dmd2） |

run 目录默认在 `<root>/zimage/work_dirs/`：`zimage-2step-1024`（ours）、`…-rdm-global-N10240`、`…-dmd2`。

看懂日志里的数：
- ours 的 `force`、RDM 的 `loss` 是自归一化后的值，**数值是常数（RDM 恒为 2.0）**，不代表训练好坏。要看 ours 的 `raw_mmd2`、RDM 的 `mmd2 s1/s2`、DMD2 的 `dmd`/`gan_g`。
- Qwen-Image-Bench 的 **Total** 越高越好；论文里的参考：GPT Image 2 64.69，FLUX 2 Pro 54.57。

---

## 7. 出问题时

| 现象 | 怎么办 |
|---|---|
| 网络断了 / 进程被杀 | 重跑同一条命令；训练加 `--resume`；评测加 `--eval-only` |
| `already holds a run` | 这个 run 目录已有训练。续跑加 `--resume`；重新开始先 `mv <run 目录>{,_old}`（问过用户再动） |
| `CUDA out of memory` | 记下是哪个阶段，把日志最后 30 行给用户；不要自己改 batch |
| 磁盘满了 | `df -h`；告诉用户。可以删的只有**已经评测过**的 `step_*.pth`（问过用户再删） |
| 好几分钟没有新输出 | 先 `nvidia-smi`：GPU-Util 接近 100% 说明在算。准备阶段（下载、contexts）和 RDM 的单步（几小时）本来就很久 |
| 评测报 `judge` 失败 | 加 `--eval-only` 重跑；已完成的部分会保留 |
| wandb 没有数据 | `cat <run 目录>/logs/wandb_watch.log` 看原因（key 错、不在 team） |

报告问题时给用户这些：执行的命令、`git log --oneline -1`、出错阶段的日志最后 30 行、`nvidia-smi`、`df -h`。
