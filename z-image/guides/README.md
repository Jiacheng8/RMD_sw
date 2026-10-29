# 实验指南

每个文件负责一个正式实验，自包含：交给 coding agent（Codex / Claude Code）或自己照着做都行。

| 文件 | 实验 | 推荐机器 | 时间 |
|---|---|---|---|
| [01_ours.md](01_ours.md) | 正式训练 ours | 8×A100/H100 | ~15–20 h + 评测 [估计] |
| [02_rdm.md](02_rdm.md) | 正式训练 RDM 基线 | 同 ours 的卡数 | ≈ ours |
| [03_dmd2.md](03_dmd2.md) | 正式训练 DMD2 基线 | 8 卡（≥4） | ≈ ours |

顺序：01 → 02、03。02 和 03 要在 01 之后、最好在同一个 `--root` 上（它们按 ours 的 GPU 时间定训练长度）。三个实验放同一个 root 需要 ~650–700 GB 磁盘。

## 怎么交给 coding agent

**第 1 步（人来做，每台机器一次）：登录 wandb。** key 只经过你自己的终端，不进 agent 的对话。
```bash
pip install -q wandb          # 在 base 环境里装：此时项目环境 rdm 可能还没装
wandb login                   # 打开 https://wandb.ai/authorize ，复制 key，粘贴（不显示），回车
```
- 账号必须是 wandb team **sw-lmmd** 的成员（Member 角色）。
- key 存进 `~/.netrc`，之后项目环境和脚本都自动读取。不登录也能跑，只是不记 wandb。

**第 2 步：把下面的 prompt 发给 agent**（只改带 `←` 的行）：

```
请按照仓库里的实验指南，在这台机器上跑一个实验。

- 仓库：https://github.com/Jiacheng8/RMD_sw.git ，分支 zimage（机器上没有就 clone 到空间最大的盘上；已有就 git pull）
- 实验指南：z-image/guides/01_ours.md          ← 换成 02_rdm.md / 03_dmd2.md
- GPU 数量：8                                   ← 实际卡数
- --root：/workspace/rdm-sets                   ← 放在空间最大的盘上
- wandb：我已经在这台机器上登录过，你不用处理

要求：
1. 先完整读一遍指南，严格按里面的规则做。不要修改任何代码，也不要删除 --root 下的数据。
2. 先运行指南里的 --dry-run，把输出发给我。我确认后再正式开始。
3. 正式运行放在 tmux 后台。定期检查，并简短汇报：当前阶段、步数、s/step、有没有报错；
   启动后告诉我输出里是 `wandb on` 还是 `wandb off`。
4. 出错时不要自己改代码或重开实验。把执行的命令、报错信息和日志最后 30 行发给我。
5. 结束后把脚本打印的结果表原样发给我。
```

第 2 条的 dry-run 是有意的：正式实验要跑十几个小时，开跑前人看一眼计划（GPU 数、预算、root 在哪）。
