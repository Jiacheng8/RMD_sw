# 实验 1：正式训练 ours（SW-LMMD，2 步 student）

**目的**：主方法。Z-Image-Turbo（8 步）→ 2 步 student，1024 px，每个 student step 一个 MMD + 一个 GAN critic，2000 步；结束后自动在 Qwen-Image-Bench（1000 prompt）上评最后一个 checkpoint（不评 teacher，teacher 由用户自己评）。

**必须在实验 2（RDM，`02_rdm.md`）和 3（DMD2，`03_dmd2.md`）之前跑**：它们读这个 run 的日志来花同样多的 GPU 时间。三个实验最好在同一台机器、同一个 `--root` 上做。

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
4. **用 prompt 给的设置**：用户的 prompt 里给了仓库目录、大数据目录（`--root`）和 GPU 数。下文命令里的 `/root/RDM`、`/root/rdm-sets` 一律换成 prompt 里的值，脚本按 GPU 数选；prompt 没给的才问用户。wandb 不用你管：用户已经登录过，**不要向用户要 key，也不要处理 key**。
5. **不要删除** `--root` 下的任何东西，除非用户明确同意。
6. **失败了先重跑同一条命令**：已完成的阶段会自动跳过；训练中断加 `--resume`。
7. **结束时**把脚本最后打印的结果表原样交给用户。

## 开始前确认

| 什么 | 怎么定 |
|---|---|
| 用哪个脚本 | 按 prompt 的 GPU 数： 8 → `ours_eight-cards.sh`；4 → `ours_four-cards.sh`；2 → `ours_two-cards.sh` |
| 先 `--gate` 吗 | 只有用户要求时：准备 + 20 步，看显存和 s/step，不评测 |

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
bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --dry-run          # 只检查，什么都不改
tmux new -d -s gate "bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --gate 2>&1 | tee /root/gate.out"   # 可选
tmux new -d -s ours "bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets 2>&1 | tee /root/ours.out"
```
- 每 500 步存一个 checkpoint（24.6 GB），另有 `resume.pth` 用于续跑。
- 中断后：同一条命令加 `--resume`。只差评测：加 `--eval-only`。
- 进度：`tail -n 5 /root/rdm-sets/zimage/work_dirs/zimage-2step-1024/train_log.jsonl`，或 `/root/ours.out` 里的 `step N | force ... | raw_mmd2 ...`。wandb：https://forge.coreweave.com/wandb/sw-lmmd ，项目 `zimage-distill`（group `ours`）。
- 看训练好坏用 `raw_mmd2`（应下降）；`force` 是自归一化后的值，不反映好坏。

## 结束时

- 对比表：`/root/rdm-sets/zimage/work_dirs/zimage-2step-1024/qwen_image_bench.md`（ours 最后一步；已评过的 teacher 也会列进去），**交给用户**。
- 告诉用户：可以接着在同一个 `--root` 上做实验 2、3。

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
