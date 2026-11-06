# 跑实验指南：Qwen3 文本版 A、B

现在要跑两个 ours 的实验，每个 500 步，学习率在这 500 步内完整衰减，训练完自动在 Qwen-Image-Bench 上评测第 500 步。两个实验的 joint MMD 和判别器都用 Qwen3-4B 文本特征，不再用 SigLIP2。

| Setting | 设置 | 配置文件（`z-image/configs/`） | 实验名 |
|---|---|---|---|
| A | Qwen3 文本 + 新的判别器学习率：`gan.lr` 2e-4 → 1e-4 | `zimage_2step_1024_dlr1e-4_qwen.yaml` | `zimage-2step-1024-dlr1e-4-qwen` |
| B | A + 两步的 loss 权重不同：`student.step_weights` [1, 1] → [0.25, 1] | `zimage_2step_1024_dlr1e-4_stepw0.25_qwen.yaml` | `zimage-2step-1024-dlr1e-4-stepw0.25-qwen` |

B 和 A 只差 loss 权重，所以两者可以直接比较。上一轮 ours 的 s500（Qwen-Image-Bench 47.92）用的是 SigLIP2 文本，是 2000 步计划的中途 checkpoint，当时学习率还有峰值的约 90%，没有退火；A 和它比时，差距同时来自文本特征、判别器学习率和退火。

Qwen3 文本特征：每个 prompt 的 Qwen3-4B context 按 token 取平均再做 L2 归一化（2560 维）；`beta` 约放大 4.5 倍，让文本核的中位距离和 SigLIP2 一致。启动脚本读到配置里的 `qwen_text_repo` 后，会自动从 Hugging Face 数据集 `jiachengcui888/zimage-ref-qwen-text` 下载（约 1.3 GB），在 reference pack 里组装成 `reference_store_qwen` 并用它训练。图像特征和 Z-Image 自己生成图像用的条件都没变。

- **机器**：8 张 A100 或 H100（80 GB），4 张或 2 张也行。
- **时间**：每个实验训练约 8 卡 4 小时、4 卡 7 小时、2 卡 13 小时（按上一轮 2×A100 的 93 s/步估算），再加评测约 1 小时。
- **磁盘**：两个实验共约 400 GB，都放在同一个大数据目录。
- **顺序**：一台机器上先跑 A，跑完再跑 B，下面的 prompt 就是这样写的。已完成的准备阶段会自动跳过。

---

## 第 1 步：下载代码

```bash
git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git /root/RDM
```
`/root/RDM` 可以换成别的目录，第 4 步的 prompt 里写同一个。已经 clone 过的话，进去更新一下：`cd /root/RDM && git pull`。

## 第 2 步：安装 Miniconda3

机器上已经有 conda（`conda --version` 能输出版本号）就跳过。
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
安装时一路回车 / 输入 `yes`，装完关掉终端重新打开。

## 第 3 步：登录 wandb

```bash
pip install -q wandb
wandb login
```
打开 https://wandb.ai/authorize ，复制 key，粘贴进去（屏幕上不显示），回车。
账号要在 wandb 的 **sw-lmmd** team 里。**没登录、key 不对、或不在 team 里，脚本会直接拒绝开始。**

## 第 4 步：把下面这段发给 coding agent（Codex / Claude Code）

<span style="color:#d73a49">**红色的 3 处**</span>改成你的情况，其余不动（在预览里选中复制，不要从源码复制）：

<pre>
请在这台机器上依次跑两个 ours 的 500 步实验：先 A，再 B。

- 仓库目录：<span style="color:#d73a49"><b>/root/RDM</b></span> —— 代码已经 clone 好（分支 zimage），所有命令都在这个目录下执行
- 大数据目录（即 --root）：<span style="color:#d73a49"><b>/root/rdm-sets</b></span> —— 所有大文件都放这里（约 400 GB），不要放到别的盘
- GPU 数量：<span style="color:#d73a49"><b>8</b></span>
- 实验 A 的配置：z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml
- 实验 B 的配置：z-image/configs/zimage_2step_1024_dlr1e-4_stepw0.25_qwen.yaml
- 环境：Miniconda3 已经装好（conda 能用）；wandb 我已经登录过，你不用处理

要求：
1. 先完整读一遍 z-image/guides/01_ours.md，严格按里面的规则和步骤做，但有下面几处不同：
   - 所有 ours_*-cards.sh 命令都要加上 --config 和对应的配置文件。不加的话跑的是原来的 2000 步配置，那就错了。
   - 这里要跑 A 和 B 两个实验，步数是 500，不是指南里写的 2000。
   - 结果目录是 work_dirs/ 下配置里 exp_name 对应的目录，不是 zimage-2step-1024。
   - 磁盘要求是约 400 GB，不是指南里写的 700 GB。
   不要修改任何代码，也不要删除 --root 下的数据。
2. 先对两个配置各运行一次 --dry-run，把输出发给我。training 那一行必须是对应的配置文件名，必须有 Qwen3 text 那一行，wandb 那一行必须是 login ok；不对就先停下告诉我。我确认后再正式开始。
3. 正式运行放在 tmux 后台。A 跑完（包括评测）并把结果表发给我之后，直接开始 B。定期汇报：当前实验、阶段、步数、每步耗时、有没有报错。
4. 出错时不要自己改代码或重开实验，把命令、报错和日志最后 30 行发给我。
5. 每个实验结束后，把结果表原样发给我。
</pre>

| 要改的地方 | 填什么 |
|---|---|
| <span style="color:#d73a49">**仓库目录** `/root/RDM`</span> | 第 1 步 clone 到的目录 |
| <span style="color:#d73a49">**大数据目录** `/root/rdm-sets`</span> | 所有大文件放哪（~400 GB）。默认 `/root/rdm-sets`；先 `df -h` 确认 `/root` 所在的盘有这么大，不够就换成最大的那块盘（比如 `/workspace/rdm-sets`）。（conda 环境装在 conda 自己的目录，另需 ~20 GB） |
| <span style="color:#d73a49">**GPU 数量** `8`</span> | 机器上的 GPU 数量（8、4 或 2） |

<details>
<summary><b>English version of the prompt</b>（点开；同样改这 3 处）</summary>

<pre>
Please run two 500-step ours experiments on this machine, one after the other: A, then B.

- Repository directory: <span style="color:#d73a49"><b>/root/RDM</b></span> -- the code is already cloned there (branch zimage); run every command from this directory
- Big-data directory (i.e. --root): <span style="color:#d73a49"><b>/root/rdm-sets</b></span> -- all large files go here (~400 GB); do not put them on any other disk
- Number of GPUs: <span style="color:#d73a49"><b>8</b></span>
- Config of experiment A: z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml
- Config of experiment B: z-image/configs/zimage_2step_1024_dlr1e-4_stepw0.25_qwen.yaml
- Environment: Miniconda3 is installed (conda works); I have already logged in to wandb, you do not need to handle it

Requirements:
1. Read the whole of z-image/guides/01_ours.md first and follow its rules and steps strictly, with these differences:
   - Add --config with the matching config file to every ours_*-cards.sh command. Without it the old 2000-step config runs, which is wrong.
   - Run the two experiments A and B, 500 steps each, not the 2000 steps the guide mentions.
   - The result directory is the one named by the config's exp_name under work_dirs/, not zimage-2step-1024.
   - The disk requirement is ~400 GB, not the 700 GB the guide mentions.
   Do not modify any code, and do not delete anything under --root.
2. Run --dry-run once for each config first and send me the output. The training line must show the matching config file name, the Qwen3 text line must be there, and the wandb line must say login ok; if not, stop and tell me. Start the real runs only after I confirm.
3. Run them in the background in tmux. Once A has finished (evaluation included) and you have sent me its result table, start B without waiting. Report regularly: current experiment, stage, step, seconds per step, any errors.
4. If something fails, do not change the code or restart the experiment yourself. Send me the command, the error and the last 30 lines of the log.
5. After each experiment, send me the result table exactly as the script printed it.
</pre>

The guides themselves are in Chinese; Codex and Claude Code read them fine.
</details>

命令长这样（4 卡用 `ours_four-cards.sh`，2 卡用 `ours_two-cards.sh`）：

```bash
bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --config z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml --dry-run
bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets --config z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml
```

## 第 5 步：等结果

- agent 会先给你看计划（dry-run），你回复“开始”后它才正式跑。
- 训练曲线：https://forge.coreweave.com/wandb/sw-lmmd （项目 zimage-distill）。训练 loss 只有这些：每一步（`s1`、`s2`）的 `gan/{dinov3_l,siglip2}/g_loss`、`gan/{dinov3_l,siglip2}/d_loss`、`{dinov3_l,siglip2}/mmd2`，以及整体的 `grad_norm`。`g_loss` 从第 50 步才开始有。其余字段（包括 `d_acc`）在 run 目录的 `train_log.jsonl` 里。
- 每个实验跑完，agent 会发回一张分数表（Qwen-Image-Bench，越高越好）。
