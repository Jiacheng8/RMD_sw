# 实验 2：正式训练 RDM 基线（原始 RDM，local-to-global）

**目的**：对照基线。同一个 2 步 student，用原始 RDM 目标：每步 10240 个新生成的样本，和整个 reference set 做 MMD（Nyström），没有 cache、没有 GAN。**花的 GPU 时间和 ours 一样**：脚本读 ours 的训练日志得到预算，先实测几步的速度，再定总步数（写进 `plan.json`）。一步要好几个小时，所以总共只有几步——这是设计如此。

**前提**：实验 1（ours，`01_ours.md`）已经在**同一个 `--root`** 上跑完。没有的话要用户给 ours 的 GPU 小时数（`--match-gpu-hours`）。

**机器**：和 ours 相同的卡数（2/4/8 卡版本）。**时间**：≈ ours 的训练时间 + 评测 ~1 h。

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
| 几张卡？ | 8 → `rdm-global_eight-cards.sh`；4 → `_four-cards`；2 → `_two-cards` |
| ours 是否在这个 root 上跑完？ | 是：什么都不用加。否：问用户 ours 的 GPU 小时数，加 `--match-gpu-hours H` |
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

## 运行（以 8 卡为例）

```bash
cd /root/RDM
bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --dry-run         # 检查 + 预算
tmux new -d -s rdmoom "bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --oom-probe 2>&1 | tee /root/rdmoom.out"   # 可选，几分钟：确认 10240 行不爆显存
tmux new -d -s rdm "export WANDB_API_KEY=<key>; bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/rdm.out"
```
- `--oom-probe` 结束时每张卡打印 `oom-probe: PASS ... peak ... GiB` 或 `FAIL`；FAIL 就停下告诉用户。
- 进度：`/root/rdm.out` 里的 `step N | loss 2.0000 | mmd2 s1=... s2=...`。**`loss` 恒为 2.0（自归一化），看 `mmd2`。** 计划定下后打印 `plan: ... -> N steps`。
- 每步耗时超过 10 分钟时每步都存 `resume.pth`；中断后加 `--resume`。
- wandb：group `rdm-global`。

## 结束时

- 输出目录：`/root/rdm-sets/zimage/work_dirs/zimage-2step-1024-rdm-global-N10240/`。
- 对比表：该目录下 `qwen_image_bench.md`（会同时列出已经评过的 ours，以及评过的 teacher），**交给用户**。

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
