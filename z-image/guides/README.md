# 实验指南

每个文件负责一个正式实验，自包含：交给 coding agent（Codex / Claude Code）或自己照着做都行。

| 文件 | 实验 | 推荐机器 | 时间 |
|---|---|---|---|
| [01_ours.md](01_ours.md) | 正式训练 ours | 8×A100/H100 | ~15–20 h + 评测 [估计] |
| [02_rdm.md](02_rdm.md) | 正式训练 RDM 基线 | 同 ours 的卡数 | ≈ ours |
| [03_dmd2.md](03_dmd2.md) | 正式训练 DMD2 基线 | 8 卡（≥4） | ≈ ours |

顺序：01 → 02、03。02 和 03 要在 01 之后、最好在同一个 `--root` 上（它们按 ours 的 GPU 时间定训练长度）。三个实验放同一个 root 需要 ~650–700 GB 磁盘。

用法示例：把 `01_ours.md` 交给 agent，并告诉它“按这个文件跑，8 张卡，root 放 /workspace/rdm-sets，wandb key 是 …”。
