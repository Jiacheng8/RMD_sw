# 机制消融：把 own-reference 拉力从窗口里分离出来（FLUX.2 [klein] 4B）

Lemma 1 说 `E[L_W] = (1 - 1/K) J + (2/K) L_self`。审稿人想知道两件事：一是 local-to-global 加一个显式的 `beta * L_self` 能不能追上窗口；二是窗口去掉对角交叉项 `k(X_i, Y_i)` 之后会怎样。当前范围是两个训练臂，各训 3,400 次更新，放进 Table 2，和 Global（0.746 / 21.81）、Local（0.854 / 21.90）并列：

| 臂 | 机器脚本 | 损失 | 对照 |
|---|---|---|---|
| `window-nodiag` | `run_machine1_window_nodiag.sh` | 主实验的窗口损失，吸引项去掉 `j = i`，其余项按 `K/(K-1)` 缩放（`attraction.skip_diagonal: true`） | Local（主实验） |
| `global-plus-self-b1` | `run_machine2_global_plus_self.sh` | Global 臂 + `beta* (2/(\|E\| B)) Σ_e Σ_i [1 - κ_e(u_i, v_i)]`（`own_term.beta = beta*`） | Global（`global-ablation-nogroup`） |

## 两个配置开关（默认都关）

```yaml
attraction:
  skip_diagonal: false   # run (b)
own_term:
  beta: 0.0              # run (a)；critic 的自适应权重 λ 对“global 损失 + beta L_self”的梯度计算（作者的决定）
```

由 `experiments.mechanism_ablation.train` 读取。它就是 Table 2 Global 臂用的训练入口（`experiments/ref_ablation`），只多了这两个开关。开关全关时与原来逐位一致：`ref_mode: local` 就是主实验的 force，`ref_mode: global` 就是 Global 臂（测试里逐步比较了日志和权重）。两个开关各自在 K=16 / B=8 的小窗口上、开着 critic 跑过 20 步的测试。

## 学习率：沿用各自对照的两段式

Table 2 的两个 step-3400 checkpoint 都是续训出来的：先在 0→2000 上 warm-up 100 步，然后余弦降到 0.1×；之后从 step 2000 的 `resume.pth` 续训，把余弦衰减拉长重新开始（`--lr restart`）。两者第二段的终点不同：

- 主实验续训到 4000，所以 step 3400 还在衰减中途。`window-nodiag` 照此执行：第二段 `steps 3400`，`lr_schedule.total_steps 4000`。
- Global 臂续训的最终状态就是 3400。`global-plus-self-b1` 照此执行：第二段的衰减终点是 3400。Global 续训的日志不在 HF 上；这里按 `run_new-machine-resume.sh` 默认的 restart 模式推断，衰减终点就是最后一步。

两台机器都是 2×H100，与两个对照相同（Global 的 step-3400 checkpoint 里记录了 `world_size 2`）。

## beta* 的标定（结果出来之前就定好）

1. Step 0：在主实验的 step 200、1000、2000（`step_0002000.pth`）和 3400 上各测 8 个连续窗口的 `|g_own|/|g_total|`。都是只有权重的 checkpoint，critic 关，按 tools/measure_cache_drift.py 的协议。
2. 第 1 轮：Global 臂加上 `beta_0 = 1/(K-1)` 的 own 项（即 J 权重的 `(2/K)/(1-1/K)`），从 base 开始跑 50 步，学习率就是正式训练前 50 步的学习率，取这 50 步 share 的平均。
3. 与主实验 step 200 的 share 相差不超过 20% 就接受。否则把每一步记录下来的梯度当成固定值，解出新的 beta，再跑 50 步验证。最多 4 轮；还不行就停下来，不开始训练。

`beta*` 写进 `WORK_DIR/beta_star.json`，两段训练都从这里读。标定成功之后，重跑脚本也不会再标定。

## 运行：两台租来的 2×H100，各跑一个脚本，互不依赖

```bash
curl -fsSLO https://raw.githubusercontent.com/Jiacheng8/RMD_sw/mechanism-ablation/experiments/mechanism_ablation/run_machine1_window_nodiag.sh
tmux new -s mech
HF_TOKEN=hf_... bash run_machine1_window_nodiag.sh        # 机器 1，约 5 h；可选 WORK_DIR=...（默认 ./work）
HF_TOKEN=hf_... bash run_machine2_global_plus_self.sh     # 机器 2，约 6 h
```

`HF_TOKEN` 要用 jiachengcui888 的 **write** token。它还要能读私有数据集 `jiachengcui888/sw-rdm-reference-store`，并且已在网页上接受 gated 的 `black-forest-labs/FLUX.2-dev` 的许可。

- **预检**（任何超过 1 GB 的下载之前）：恰好 2 张 H100；打印驱动和 CUDA 版本；磁盘空间（机器 1 要约 330 GB，机器 2 约 350 GB，会打印明细）；`HF_TOKEN` 能否写 jiachengcui888、读私有参考库、读 FLUX.2-dev。
- **环境**：Miniconda（没有就装到 `~/miniconda3`）；仓库 clone 到固定 commit；`scripts/new_machine.sh --no-train --skip-geneval` 装环境和下载数据。环境是 torch 2.8.0+cu126、bitsandbytes 0.50.2 和 `requirements-lock.txt`，模型 snapshot 按 `assets/hf_revisions.json`，这几个文件自主实验的 commit 71e96e9 以来没有变过。参考库 tar 的 md5 是 `2165876e…`。
- **H100 上不评测**：脚本不装 GenEval，只负责训练和上传。评测在 6×4090 这台论文评测机上做，见下文。
- **可续跑**：每个阶段完成后写 `WORK_DIR/done/<stage>`。重跑会跳过已完成的阶段，训练从最近的 `resume.pth`（每 400 步一个）接着跑。
- **上传**：训练侧文件传到 `datasets/jiachengcui888/evaluation-result/mechanism-ablation/<run>/train_side/`，包括 `report.md`、`run_info.json`、`train_log.jsonl` 和 `run.log`；机器 2 另有 Step 0 表和标定日志。最终的 step 3400 checkpoint 传到公开 model repo `jiachengcui888/<run>`。
- 日志只有一个：`WORK_DIR/logs/run.log`，同时输出到 stdout。各子步骤也有自己的日志，放在同一目录下。

## 评测：在 6×4090 论文评测机上

```bash
cd /home/jiacheng/RDM-mechanism && git pull
tmux new -s mecheval
bash experiments/mechanism_ablation/eval_4090.sh --dry-run
bash experiments/mechanism_ablation/eval_4090.sh            # HF 上已有的臂都会评；加 --anchor 会顺便重新评测主实验 step 3400
```

脚本从 HF 下载每个臂的 step 3400 和 `train_side/`，用论文的流程评测：eval root 是 `/data/thor/jiacheng/rdm-sets-eval`，评测脚本用 `/home/jiacheng/RDM` 的 `eval_checkpoint.sh --no-diversity`；GenEval 553×4、shared noise、官方检测器；PickScore 499 个 prompt，每个 1 张图。Table 2 里 Global 3400 的 0.7462 就是这套流程在这里评出来的。之后与 Table 2 并列生成 `report.md`、`scores.json`、`run_info.json`，上传到 `mechanism-ablation/<run>/`。每张空闲 GPU 跑一个 checkpoint，每个约 22 分钟。

两个臂都评完后，合并成顶层 `report.md`（每臂一行）这一步还没有写成脚本。

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 两个开关的解析，未知键报错 |
| `loss.py` | `MechanismLoss(RefAblationLoss)`：`main`（只测量对角项）、`nodiag`、`global_self` |
| `trainer.py` | `MechanismTrainer(RefAblationTrainer)`：只用 `step()` 的四个钩子，不复制 `step()`；每步记录 own 项的值，以及 `g_own` / `g_rest` 的拆分 |
| `train.py` | 入口：Global 臂的训练循环，运行目录 = `exp_name`；写 `run_info.json` |
| `probe.py` | Step 0（`--table` 打印 markdown 表） |
| `calibrate.py` | beta* 标定 |
| `run_arm.sh` | 单次训练：`run_arm.sh <window_nodiag\|global_plus_self\|main_probe> [--set K=V]` |
| `machine_lib.sh` | 两个机器脚本共用的阶段 |
| `run_machine{1,2}_*.sh` | 两个一键脚本 |
| `report.py` | 两种报告：`train`（H100 上，只有训练信息）和 `scored`（4090 上，与 Table 2 并列） |
| `eval_4090.sh` | 在 4090 上评测、生成报告并上传 |
| `tests/` | `python -m pytest experiments/mechanism_ablation/tests`（CPU，玩具数据） |

## 日志字段（`train_log.jsonl`，每一步）

ref-ablation 原有的字段都保留，另外新增：`mech/<enc>/own_kernel`（`k(x_i, y_i)` 的均值）、`mech/<enc>/own_term`（own 项的值，force 单位；nodiag 下是被去掉的那一项）、`mech/<s>/grad_own|grad_rest|grad_total|share|cos_own_total|cos_rest_total`（`s` 是各编码器或 `all`），以及 `mech/all/sq_own|sq_total|dot_own_total`（标定用的原始平方和）。
