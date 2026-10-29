# 跑实验指南

一共三个实验，**按顺序跑，放在同一台机器上**：

| 顺序 | 实验 | 指南文件 | 大概时间 |
|---|---|---|---|
| 1 | ours（我们的方法） | `01_ours.md` | 8 卡 ~15–20 小时（4 卡约翻倍） |
| 2 | RDM 基线 | `02_rdm.md` | 和 ours 差不多 |
| 3 | DMD2 基线 | `03_dmd2.md` | 和 ours 差不多 |

机器：**8 张 A100 或 H100（80 GB）**（4 张也行，时间约翻倍），磁盘 **~700 GB**。实验 2、3 必须等实验 1 跑完再开始，而且一次只跑一个。

---

## 第 1 步：登录 wandb（每台机器做一次）

```bash
pip install -q wandb
wandb login
```
打开 https://wandb.ai/authorize ，复制 key，粘贴进去（屏幕上不显示），回车。
账号要在 wandb 的 **sw-lmmd** team 里。**没登录、key 不对、或不在 team 里，脚本会直接拒绝开始。**

## 第 2 步：下载代码（每台机器做一次）

```bash
git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git /root/RDM
```
`/root/RDM` 可以换成别的目录，第 3 步的 prompt 里写同一个。已经 clone 过的话，进去更新一下：`cd /root/RDM && git pull`。

## 第 3 步：把下面这段发给 coding agent（Codex / Claude Code）

<span style="color:#d73a49">**红色的 4 处**</span>改成你的情况，其余不动（在预览里选中复制，不要从源码复制）：

<pre>
请按照仓库里的实验指南，在这台机器上跑一个实验。

- 仓库目录：<span style="color:#d73a49"><b>/root/RDM</b></span> —— 代码已经 clone 好（分支 zimage），所有命令都在这个目录下执行
- 大数据目录（即 --root）：<span style="color:#d73a49"><b>/root/rdm-sets</b></span> —— 模型权重、训练数据、checkpoint、评测结果全部放这里（约 700 GB），不要放到别的盘
- 实验指南：z-image/guides/<span style="color:#d73a49"><b>01_ours.md</b></span>
- GPU 数量：<span style="color:#d73a49"><b>8</b></span>
- wandb：我已经在这台机器上登录过，你不用处理

要求：
1. 先完整读一遍指南，严格按里面的规则做。不要修改任何代码，也不要删除 --root 下的数据。
2. 先运行指南里的 --dry-run，把输出发给我。我确认后再正式开始。
3. 正式运行放在 tmux 后台。定期汇报：当前阶段、步数、每步耗时、有没有报错；
   dry-run 里 wandb 那一行必须是 login ok；不是的话先停下告诉我。
4. 出错时不要自己改代码或重开实验，把命令、报错和日志最后 30 行发给我。
5. 结束后把结果表原样发给我。
</pre>

| 要改的地方 | 填什么 |
|---|---|
| <span style="color:#d73a49">**仓库目录** `/root/RDM`</span> | 第 2 步 clone 到的目录 |
| <span style="color:#d73a49">**大数据目录** `/root/rdm-sets`</span> | 所有大文件放哪（~700 GB）。默认 `/root/rdm-sets`；先 `df -h` 确认 `/root` 所在的盘有这么大，不够就换成最大的那块盘（比如 `/workspace/rdm-sets`）；**三个实验都用同一个**。（conda 环境装在 conda 自己的目录，另需 ~20 GB） |
| <span style="color:#d73a49">**实验指南** `01_ours.md`</span> | 第几个实验：`01_ours.md` → `02_rdm.md` → `03_dmd2.md` |
| <span style="color:#d73a49">**GPU 数量** `8`</span> | 机器上的 GPU 数量 |

<details>
<summary><b>English version of the prompt</b>（点开；同样改这 4 处）</summary>

<pre>
Please run one experiment on this machine by following the experiment guide in the repository.

- Repository directory: <span style="color:#d73a49"><b>/root/RDM</b></span> -- the code is already cloned there (branch zimage); run every command from this directory
- Big-data directory (i.e. --root): <span style="color:#d73a49"><b>/root/rdm-sets</b></span> -- model weights, training data, checkpoints and evaluation results all go here (~700 GB); do not put them on any other disk
- Experiment guide: z-image/guides/<span style="color:#d73a49"><b>01_ours.md</b></span>
- Number of GPUs: <span style="color:#d73a49"><b>8</b></span>
- wandb: I have already logged in on this machine; you do not need to handle it

Requirements:
1. Read the whole guide first and follow its rules strictly. Do not modify any code, and do not delete anything under --root.
2. Run the guide's --dry-run first and send me its output. Start the real run only after I confirm.
3. Run it in the background in tmux. Report regularly: current stage, step, seconds per step, any errors;
   The wandb line of the dry-run must say login ok; if it does not, stop and tell me.
4. If something fails, do not change the code or restart the experiment yourself. Send me the command, the error and the last 30 lines of the log.
5. When it finishes, send me the result table exactly as the script printed it.
</pre>

The guides themselves are in Chinese; Codex and Claude Code read them fine.
</details>

## 第 4 步：等结果

- agent 会先给你看计划（dry-run），你回复“开始”后它才正式跑。
- 训练曲线：https://forge.coreweave.com/wandb/sw-lmmd （项目 zimage-distill）
- 每个实验跑完，agent 会发回一张分数表（Qwen-Image-Bench，越高越好）。
- 一个实验跑完，再用同样的 prompt 开下一个（只改指南文件名）。
