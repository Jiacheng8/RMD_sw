# Qwen-Image-Bench：两个公开的 Z-Image-Turbo 蒸馏模型（TwinFlow 2 步、DIDR 1 步）

和我们的两步学生（`zimage-2step-1024_s500`）、teacher（`turbo9`）放在同一张表里比较。协议完全一样：
`prompt_cn`、每个 prompt 一张、1024×1024、bf16、seed 0（第 ID 个 prompt 的噪声 = `manual_seed(ID)`）。

| 模型 | 数据集里的目录 | 状态 |
|---|---|---|
| TwinFlow-Z-Image-Turbo-exp，2 NFE | `qwen-image-bench/twinflow-zimage-turbo-exp_2nfe_1024` | 图已上传（2026-11-10），**未打分** |
| DIDR，1 NFE | `qwen-image-bench/didr-zimage_1nfe_1024` | 图已上传（2026-11-10），**未打分** |

数据集：<https://huggingface.co/datasets/jiachengcui888/evaluation-result>。每个目录里有 1000 张图、
`input.jsonl`、`generation.json`、`run_info.json`（权重 revision、代码 commit、采样参数、版本、耗时）和
`contact_sheet_16.png`。

---

## 打分（图已经在 Hub 上，不用重新渲染）

机器：**4 张 A100 / H100（80 GB）**，磁盘 ~70 GB。一份 judge 占一张 80 GB 卡，4 张卡同时跑。
每个模型约 1–1.5 小时（按 s500、s2000 实测的每 1000 张 3.6–6 GPU 小时估算），两个合计 2–3 小时，
再加首次下载 judge 环境和权重（~65 GB）的时间。40 GB 的卡一份 judge 要 2 张，时间约翻倍。

需要 NVIDIA 驱动和 conda（没有就先装 Miniconda3，见 `Baseline-README.md` 第 2 步）。

```bash
git clone -b zimage-exp https://github.com/Jiacheng8/RMD_sw.git /root/RDM && cd /root/RDM
bash z-image/scripts/judge_hub_runs.sh --root /root/rdm-sets --dry-run   # 先看检查和计划
bash z-image/scripts/judge_hub_runs.sh --root /root/rdm-sets             # 在 tmux 里跑
```

脚本会：建 rdm 环境 → 装 judge 环境 + Q-Judger → 从 Hub 下载两组图 → 前 20 张冒烟测试 →
逐个打分，每打完一个就把分数传回同一个目录（图不重复上传）→ 和 s500、turbo9、turbo2 出一张表
（`<root>/qib/runs/report_hub_runs.md`）。开始时会要一次 HF **Write** token（不落盘）。中断后重跑同一条命令，
已完成的部分都会保留。

**先测 judge 自己的噪声**（建议）：同一组图再打一次分，看分数会漂多少，之后才知道差多少分算"真的更好"：

```bash
bash z-image/scripts/judge_hub_runs.sh --root /root/rdm-sets zimage-2step-1024_s500 --suffix _rejudge
```

已经有分数的目录不会被覆盖：再打一次必须加 `--suffix`，结果另存为 `<run><suffix>`。

## 重新渲染（一般不需要）

`z-image/scripts/qib_render_released.sh` 是单文件脚本，自己装环境（不需要 rdm 环境）、下载权重
（~47 GB）、渲染、检查，可选上传。所有版本和 revision 都固定在 2026-11-10 那次。每张卡需要 ≥ 24 GB。

```bash
bash z-image/scripts/qib_render_released.sh --root /data/qib_released --setup-only   # 只装环境+下载，不用 GPU
bash z-image/scripts/qib_render_released.sh --root /data/qib_released --limit 8      # 冒烟测试
bash z-image/scripts/qib_render_released.sh --root /data/qib_released                # 两个模型各 1000 张
```

原来的两个目录名在 Hub 上已存在，所以重新上传要加 `--upload --suffix <后缀>`。

## 和我们协议不一样的地方

- TwinFlow 用 diffusers **0.36.0**（官方 wrapper 按位置传构造参数，0.40 会错位），DIDR 用 0.40.0。
- TwinFlow 的 `max_sequence_length` 用协议的 512（官方默认 1024）；1000 个 prompt 都不超过 512 token，没有影响。
- TwinFlow `aux_time_embed=True`：权重里有 4 个 `t_embedder_2` key 且和 `t_embedder` 不同，载入 525 个 key 无
  missing / unexpected。bf16 取整后两步的 t 实际是 1.0 和 0.6016（官方代码本来的行为）。
