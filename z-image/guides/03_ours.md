# 实验 3：正式训练 ours（SW-LMMD，2 步 student）

**目的**：主方法。Z-Image-Turbo（8 步）→ 2 步 student，1024 px，每个 student step 一个 MMD + 一个 GAN critic，2000 步；结束后自动在 Qwen-Image-Bench（1000 prompt）上评最后一个 checkpoint，以及 teacher 9 步 / 2 步。

**必须在实验 4（RDM）和 5（DMD2）之前跑**：它们读这个 run 的日志来花同样多的 GPU 时间。三个实验最好在同一台机器、同一个 `--root` 上做。

**机器**：8×A100/H100 80GB（也有 2 卡、4 卡版本）。**磁盘**：本实验 ~330 GB；三个实验都放这个 root 共 ~650–700 GB。**时间**：2×A100 实测 93 s/步（~52 h）；8 卡估计 25–30 s/步（~15–20 h）；再加评测 ~2–3 h、首次准备 1–2 h。

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
| 几张卡？ | 8 → `ours_eight-cards.sh`；4 → `ours_four-cards.sh`；2 → `ours_two-cards.sh` |
| 先 `--gate` 测速度吗？ | 推荐：准备 + 20 步，看显存和 s/step，不评测 |
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
bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --dry-run          # 只检查，什么都不改
tmux new -d -s gate "bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --gate 2>&1 | tee /root/gate.out"   # 可选
tmux new -d -s ours "export WANDB_API_KEY=<key>; bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/ours.out"
```
- 每 500 步存一个 checkpoint（24.6 GB），另有 `resume.pth` 用于续跑。
- 中断后：同一条命令加 `--resume`。只差评测：加 `--eval-only`。
- 进度：`tail -n 5 /root/rdm-sets/zimage/work_dirs/zimage-2step-1024/train_log.jsonl`，或 `/root/ours.out` 里的 `step N | force ... | raw_mmd2 ...`。wandb：`wandb.ai/sw-lmmd/zimage-distill`（group `ours`）。
- 看训练好坏用 `raw_mmd2`（应下降）；`force` 是自归一化后的值，不反映好坏。

## 结束时

- 对比表：`/root/rdm-sets/zimage/work_dirs/zimage-2step-1024/qwen_image_bench.md`（ours 最后一步、turbo9、turbo2），**交给用户**。
- 告诉用户：可以接着在同一个 `--root` 上做实验 4、5。

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
