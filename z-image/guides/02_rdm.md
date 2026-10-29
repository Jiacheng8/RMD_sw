# 实验 2：正式训练 RDM 基线（原始 RDM，local-to-global）

**目的**：对照基线。同一个 2 步 student，用原始 RDM 目标：每步 10240 个新生成的样本，和整个 reference set 做 MMD（Nyström），没有 cache、没有 GAN。**花的 GPU 时间和 ours 一样**：默认预算 100 GPU 小时（ours 2000 步的训练时间），先实测第一步的速度，再定总步数（写进 `plan.json`，预计 ~6 步）。一步要好几个小时，所以总共只有几步——这是设计如此。

**不需要等 ours**：预算是固定的 100 GPU 小时，可以和 ours 同时在另一台机器上跑。

**机器**：和 ours 相同的卡数（2/4/8 卡版本）。**时间**：≈ ours 的训练时间 + 评测几小时（judge 先逐项推理再打分）。

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
4. **用 prompt 给的设置**：用户的 prompt 里给了仓库目录、大数据目录（`--root`）和 GPU 数。下文命令里的 `/root/RDM`、`/root/rdm-sets` 一律换成 prompt 里的值，脚本按 GPU 数选；prompt 没给的才问用户。wandb 不用你管：用户已经登录过，**不要向用户要 key，也不要处理 key**。
5. **不要删除** `--root` 下的任何东西，除非用户明确同意。
6. **失败了先重跑同一条命令**：已完成的阶段会自动跳过；训练中断加 `--resume`。
7. **结束时**把脚本最后打印的结果表原样交给用户。

## 开始前确认

| 什么 | 怎么定 |
|---|---|
| 用哪个脚本 | 按 prompt 的 GPU 数： 8 → `rdm-global_eight-cards.sh`；4 → `_four-cards`；2 → `_two-cards` |
| GPU 时间预算 | 默认 **100 GPU 小时**（= ours 2000 步的训练时间），什么都不用加；dry-run 里 `budget      --match-gpu-hours 100` 就对了。用户给了别的数才加 `--match-gpu-hours H` |

## 检查机器（开始前）

代码由用户 clone 好了，不要再 clone。在仓库目录（prompt 给的，下面以 `/root/RDM` 为例）里：
```bash
cd /root/RDM && git pull && git log --oneline -1   # 记下 commit
conda --version                                    # 用户装好的 Miniconda3
nvidia-smi                                         # 卡数和 prompt 一致、都是 80 GB、都空闲
df -h /root/rdm-sets 2>/dev/null || df -h "$(dirname /root/rdm-sets)"   # --root 所在的盘
```
- `--root` 所在的盘剩余空间不到 **~700 GB**，或卡数 / 显存和 prompt 对不上：**先停下告诉用户**，不要开始。
- 用户已经装好 Miniconda3（`conda --version` 能用；不能用就停下告诉用户，不要自己装别的 conda）。项目环境由脚本自己装进去。**不需要 HF token**。
- **wandb**：由用户在交给 agent 之前自己登录（见 `README.md` 第 1 步）。脚本**要求** wandb：`--dry-run` 输出里应有 `wandb       login ok: account ...`；如果是 `MISSING` 或 `FAILS`，正式运行会直接停下——这时**不要开始**，把那一行发给用户，等用户登录好再继续。不要自己处理 key，也不要自己加 `--no-wandb`（只有用户明确要求不用 wandb 时才加）。

## 运行（以 8 卡为例）

```bash
cd /root/RDM
bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --dry-run         # 检查 + 预算
tmux new -d -s rdmoom "bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --oom-probe 2>&1 | tee /root/rdmoom.out"   # 可选，几分钟：确认 10240 行不爆显存
tmux new -d -s rdm "bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/rdm.out"
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
