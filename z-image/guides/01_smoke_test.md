# 实验 1：冒烟测试（4×A100 80GB）

**目的**：正式实验前，确认 ours / RDM / DMD2 三条流程都能走通“训练 → 存模型 → 出图 → 打分”。每个方法只训练 10 步、评 10 个 prompt。**不出可用的分数，只看 PASS/FAIL。**

**机器**：4×A100 80GB（或 4×H100）。**磁盘**：~300 GB。**时间**：约半天（其中准备 1–2 h）。

> 给 coding agent（Codex / Claude Code）：这份文件是自包含的，按顺序做即可。

## 给 agent 的规则（必读）

1. **只做本文件这一个实验。** 不要顺手跑别的脚本。
2. **不要改代码。** 只运行本文件里的命令。脚本报错时，把命令、报错和日志最后 30 行交给用户，不要自己改 `.py` / `.sh`。
3. **长任务放进 tmux 后台**（会跑几小时，超过 agent 单条命令的超时），然后定期看日志：
   ```bash
   tmux new -d -s <名字> "<命令> 2>&1 | tee /root/<名字>.out"
   tail -n 30 /root/<名字>.out     # 进度
   tmux ls                         # 还在不在跑
   ```
   不要因为“好像没动”就杀进程：先 `nvidia-smi`，GPU-Util 接近 100% 就是在算。
4. **开始前先问齐用户**：下面「开始前要问的」那张表。agent 的 shell 不是交互终端，脚本不会弹输入框，wandb key 只能用环境变量传。
5. **不要删除** `--root` 下的任何东西，除非用户明确同意。
6. **失败了先重跑同一条命令**：已完成的阶段会自动跳过；训练中断加 `--resume`。
7. **结束时**把脚本最后打印的结果表原样交给用户。

## 开始前要问的

| 问题 | 默认 / 说明 |
|---|---|
| 机器是 4 张 80 GB 卡吗？ | 不是就停下问用户 |
| wandb key | 可选，sw-lmmd team 成员的 key |
| `--root` 放哪 | 空间最大的盘，默认 `/root/rdm-sets` |

## 机器准备（每台新机器一次）

```bash
nvidia-smi                                   # 卡型号、数量、是否空闲（要 80 GB 的卡）
df -h / /root /workspace 2>/dev/null         # 选空间最大的盘放 --root
cd /root                                     # 或大盘所在目录
git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM   # 已经有就 cd RDM && git pull
cd RDM && git log --oneline -1               # 记下 commit
```
- 下文用 `/root/rdm-sets` 作 `--root`、`/root/RDM` 作仓库；大盘是 `/workspace` 就全部换掉。
- 需要 `conda` 和 NVIDIA 驱动（GPU 云镜像一般都有），环境由脚本自己装。**不需要 HF token**。
- **wandb（可选）**：把 key 写进 tmux 命令（`export WANDB_API_KEY=<key>; ...`），或在环境装好后登录一次 `conda run -n rdm wandb login <key>`。key 要属于 **sw-lmmd** team 成员（https://wandb.ai/authorize）。不给 key 也照常跑，只是不记 wandb。

## 运行

```bash
cd /root/RDM
tmux new -d -s smoke "export WANDB_API_KEY=<key>; bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets 2>&1 | tee /root/smoke.out"
```
（不用 wandb 就去掉 `export WANDB_API_KEY=<key>;`。）

依次四个阶段：

| 阶段 | 做什么 | 日志 |
|---|---|---|
| ours | 准备（第一次 1–2 h：环境、权重、judge、contexts）；训练 10 步；评 teacher 9 步/2 步和 ours，各 10 个 prompt | `/root/rdm-sets/zimage/smoke/logs/ours.log` |
| rdm | RDM 每步 128 行，训练 10 步，评 10 个 prompt | `.../logs/rdm.log` |
| rdm-oom | 只测 RDM 每步 10240 行会不会爆显存（几分钟，不训练不评测） | `.../logs/rdm-oom.log` |
| dmd2 | 下载 undistilled Z-Image + 准备好的 latents（~18 GB），训练 10 iteration，评 10 个 prompt | `.../logs/dmd2.log` |

看进度：`tail -n 30 /root/rdm-sets/zimage/smoke/logs/<阶段>.log`。wandb：`wandb.ai/sw-lmmd/zimage-distill-smoke`。

## 结束时

`/root/smoke.out` 最后有一张汇总表：每个阶段 PASS/FAIL、耗时、s/step、峰值显存、10 个 prompt 的总分（rdm-oom 那行最后一列是峰值显存）。**把这张表原样交给用户。** 有 FAIL 的话，加上对应 `logs/<阶段>.log` 的最后 30 行。

只重跑某个阶段：
```bash
tmux new -d -s smoke2 "bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets --only dmd2 2>&1 | tee /root/smoke2.out"
```
（`--only` 可选 `ours,rdm,rdm-oom,dmd2`，逗号分隔。）

## 出问题时

| 现象 | 怎么办 |
|---|---|
| 网络断了 / 进程被杀 | 重跑同一条命令（训练加 `--resume`，只差评测加 `--eval-only`） |
| `already holds a run` | 已有同名 run。续跑加 `--resume`；要重来先问用户，再 `mv <run 目录>{,_old}` |
| `CUDA out of memory` | 不要改参数；把出错阶段和日志最后 30 行交给用户 |
| 磁盘满 | `df -h` 报告用户；只有**已评测过**的 `step_*.pth` 可以删，且要先问 |
| 很久没新输出 | 准备阶段（下载、重建 contexts）和 RDM 的单步本来就要很久；看 `nvidia-smi` |
| wandb 没数据 | `cat <run 目录>/logs/wandb_watch.log` 看原因 |

报告问题时附上：命令、`git log --oneline -1`、日志最后 30 行、`nvidia-smi`、`df -h`。
