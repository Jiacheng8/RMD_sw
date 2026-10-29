# 跑实验指南

一共三个实验，**按顺序跑，放在同一台机器上**：

| 顺序 | 实验 | 指南文件 | 大概时间 |
|---|---|---|---|
| 1 | ours（我们的方法） | `01_ours.md` | ~15–20 小时 |
| 2 | RDM 基线 | `02_rdm.md` | 和 ours 差不多 |
| 3 | DMD2 基线 | `03_dmd2.md` | 和 ours 差不多 |

机器：**8 张 A100 或 H100（80 GB）**，磁盘 **~700 GB**。实验 2、3 必须等实验 1 跑完再开始，而且一次只跑一个。

---

## 第 1 步：登录 wandb（每台机器做一次）

```bash
pip install -q wandb
wandb login
```
打开 https://wandb.ai/authorize ，复制 key，粘贴进去（屏幕上不显示），回车。
账号要在 wandb 的 **sw-lmmd** team 里。

## 第 2 步：把下面这段发给 coding agent（Codex / Claude Code）

复制下面这段（代码块右上角有复制按钮），把表里列的 4 处改成你的情况，其余不动：

```text
请按照仓库里的实验指南，在这台机器上跑一个实验。

- 仓库：https://github.com/Jiacheng8/RMD_sw.git ，分支 zimage
- 工作目录：/workspace —— 仓库放在 /workspace/RDM（没有就 clone 到这里；已有就 cd 进去 git pull），所有命令在 /workspace/RDM 下执行
- 大数据目录（即 --root）：/workspace/rdm-sets —— 模型权重、训练数据、checkpoint、评测结果全部放这里（约 700 GB），不要放到别的盘
- 实验指南：z-image/guides/01_ours.md
- GPU 数量：8
- wandb：我已经在这台机器上登录过，你不用处理

要求：
1. 先完整读一遍指南，严格按里面的规则做。不要修改任何代码，也不要删除 --root 下的数据。
2. 先运行指南里的 --dry-run，把输出发给我。我确认后再正式开始。
3. 正式运行放在 tmux 后台。定期汇报：当前阶段、步数、每步耗时、有没有报错；
   启动后告诉我输出里是 wandb on 还是 wandb off。
4. 出错时不要自己改代码或重开实验，把命令、报错和日志最后 30 行发给我。
5. 结束后把结果表原样发给我。
```

| 要改的地方 | 填什么 |
|---|---|
| <span style="color:#d73a49">**工作目录** `/workspace`</span> | 仓库 clone 到哪（会放在它下面的 `RDM/`）。先 `df -h` 看哪块盘最大 |
| <span style="color:#d73a49">**大数据目录** `/workspace/rdm-sets`</span> | 所有大文件放哪（~700 GB）。必须在最大的那块盘上；**三个实验都用同一个**。（conda 环境装在 conda 自己的目录，另需 ~20 GB） |
| <span style="color:#d73a49">**实验指南** `01_ours.md`</span> | 第几个实验：`01_ours.md` → `02_rdm.md` → `03_dmd2.md` |
| <span style="color:#d73a49">**GPU 数量** `8`</span> | 机器上的 GPU 数量 |

<details>
<summary><b>English version of the prompt</b>（点开；同样改这 4 处）</summary>

```text
Please run one experiment on this machine by following the experiment guide in the repository.

- Repository: https://github.com/Jiacheng8/RMD_sw.git , branch zimage
- Working directory: /workspace -- the repository lives in /workspace/RDM (clone it there if missing; otherwise cd into it and git pull); run every command from /workspace/RDM
- Big-data directory (i.e. --root): /workspace/rdm-sets -- model weights, training data, checkpoints and evaluation results all go here (~700 GB); do not put them on any other disk
- Experiment guide: z-image/guides/01_ours.md
- Number of GPUs: 8
- wandb: I have already logged in on this machine; you do not need to handle it

Requirements:
1. Read the whole guide first and follow its rules strictly. Do not modify any code, and do not delete anything under --root.
2. Run the guide's --dry-run first and send me its output. Start the real run only after I confirm.
3. Run it in the background in tmux. Report regularly: current stage, step, seconds per step, any errors;
   after launch, tell me whether the output says wandb on or wandb off.
4. If something fails, do not change the code or restart the experiment yourself. Send me the command, the error and the last 30 lines of the log.
5. When it finishes, send me the result table exactly as the script printed it.
```

The guides themselves are in Chinese; Codex and Claude Code read them fine.
</details>

## 第 3 步：等结果

- agent 会先给你看计划（dry-run），你回复“开始”后它才正式跑。
- 训练曲线：https://wandb.ai/sw-lmmd/zimage-distill
- 每个实验跑完，agent 会发回一张分数表（Qwen-Image-Bench，越高越好）。
- 一个实验跑完，再用同样的 prompt 开下一个（只改指南文件名）。
