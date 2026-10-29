# 实验 2：评测 teacher（Z-Image-Turbo 在 Qwen-Image-Bench 上）

**目的**：Z-Image-Turbo 本身的分数——官方 9 步（上限参考）和直接跑 2 步（不蒸馏的 2 步基线）。1000 个中文 prompt，Q-Judger（Qwen3.6-27B）打分，0–100。

**机器**：4×A100 80GB 推荐（任何 80 GB 卡都行，卡越多越快）。**磁盘**：~100 GB。**时间**：新机器约 2–2.5 h（准备 30–40 min + turbo9 出图 ~20 min（实测 4.4 s/张）+ turbo2 出图 ~7 min + 打分每个模型 ~20–35 min [估计]）。

如果这个 `--root` 上已经跑完了实验 3（ours 正式训练），teacher 已经评过了，不用再跑。

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
| 评哪些步数？ | 默认 `9,2` |
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
tmux new -d -s teacher "bash z-image/scripts/evaluate-baseline.sh --root /root/rdm-sets --steps 9,2 2>&1 | tee /root/teacher.out"
```
- 自动挑空闲的卡（显存占用 < 2 GB 的）；指定卡用 `--gpus 0,1,2,3`。
- 先试跑 20 个 prompt 确认流程：加 `--limit 20`（结果叫 `turbo9_limit20`，不进正式表）。

## 结束时

- 分数表：`/root/rdm-sets/qib/runs/report.md`（也打印在 `/root/teacher.out` 最后），**交给用户**。
- 每个模型：`/root/rdm-sets/qib/runs/turbo9/`、`turbo2/`（`images/` 1000 张图，`input_bench_scores.json` 分数）。
- 论文参考：GPT Image 2 64.69，FLUX 2 Pro 54.57，Qwen Image 49.23。

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
