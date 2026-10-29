# 实验指南

每个文件负责一个实验，自包含：交给 coding agent（Codex / Claude Code）或自己照着做都行。

| 文件 | 实验 | 推荐机器 | 时间 |
|---|---|---|---|
| [01_smoke_test.md](01_smoke_test.md) | 冒烟测试：三个方法各 10 步 + 10 个 prompt，只看能否跑通 | 4×A100 80GB | ~半天 |
| [02_teacher_eval.md](02_teacher_eval.md) | 评测 teacher（Turbo 9 步、2 步） | 4×A100 80GB | ~2–2.5 h |
| [03_ours.md](03_ours.md) | 正式训练 ours | 8×A100/H100 | ~15–20 h + 评测 [估计] |
| [04_rdm.md](04_rdm.md) | 正式训练 RDM 基线 | 同 ours 的卡数 | ≈ ours |
| [05_dmd2.md](05_dmd2.md) | 正式训练 DMD2 基线 | 8 卡（≥4） | ≈ ours |

顺序：01 →（02 可选）→ 03 → 04、05。04 和 05 要在 03 之后、最好在同一个 `--root` 上（它们按 ours 的 GPU 时间定训练长度）。三个正式实验放同一个 root 需要 ~650–700 GB 磁盘。

用法示例：把 `03_ours.md` 交给 agent，并告诉它“按这个文件跑，8 张卡，root 放 /workspace/rdm-sets，wandb key 是 …”。
