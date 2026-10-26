# Z-Image-Turbo → 2 步学生（SW-LMMD + 每步一个 GAN）

老师 Z-Image-Turbo（官方设置 `num_inference_steps=9`），学生 2 步，1024 px。
方法沿用 K128 · GAN（K=128，B=32，ungrouped，row critic，自适应权重 1.5），并对**每一步**都施加：

```
L = Σ_k w_k · ( force_k + λ_k · adv_k ),   k = 1, 2
```

- 第 k 步的 x̂₀ 有自己的 128 行生成窗口和缓存，和**同一批**参考行做 local-to-local MMD；
- 第 k 步有自己的 critic，λ_k 只和 force_k 平衡；
- 两步之间：`x_2 = (1 - 0.75)·sg(x̂₀⁽¹⁾) + 0.75·ε`，即重新加噪到 σ=0.75（Turbo 自己的 2 步 schedule），
  第 2 步的 loss 只回传到第 2 次前向。

`rdm/` 里没有任何改动：trainer、缓存、refresh、GradCache、训练循环、checkpoint/resume 都是原来的，
这里只是把每个 (step, encoder) 当成一个特征流 `s{k}/{enc}` 交给原 trainer。

**参考集**由 `process_z-image/`（原 `/data/ironman/jiacheng/RDM/z-image/process`）构建（`launch.py all`）；`prompt_build/` 的图和 caption 仍在数据盘上，不进 git。

## 目录

| 文件 | 作用 |
|---|---|
| `zimage_distill/model.py` | `ZImageStudent`（多步采样器，返回每一步的 x̂₀）、`ZImageVAE`、`reference_pixels`、`ZImageGenerator` |
| `zimage_distill/store.py` | 读 process/ 的 store（预存 Qwen3 context、prompt 表）、按步的特征流 |
| `zimage_distill/trainer.py` | `MultiStepTrainer`：每步一套 MMD 窗口、缓存和 critic |
| `zimage_distill/launch.py` | 从 config 组装并训练（torchrun 入口） |
| `zimage_distill/check_teacher.py` | M0：采样器和官方 pipeline 对齐、NFE、老师 9/4/2 步 baseline |
| `zimage_distill/sample.py` | 用 checkpoint 出图（评测用） |
| `zimage_distill/contexts.py` | 在训练机上重建每条 prompt 的 Qwen3 context（按长度分桶、多卡），存成大文件 + 索引 |
| `configs/zimage_2step_1024_h100.yaml` | 默认配置：H100，FSDP，reference_root 指向 process/ 的 reference_v1 |
| `process_z-image/` | 参考集构建代码（prompt 冻结 → 老师渲染 → 提特征 → finalize），从 `/data/ironman/jiacheng/RDM/z-image/process` 复制，只改了路径 |
| `scripts/upload_reference_v1.sh` | 本机：reference_v1 做完后一键打包并公开上传到 `jiachengcui888/zimage-ref-v1` |
| `scripts/pack_reference.sh` | 本机：把做好的参考集打成训练包（不含 PNG 和 context，约 1.6 GB），可上传 HF dataset |
| `eval_qib/` | Qwen-Image-Bench 本地评测：出图 → 官方 `judge.py` + Q-Judger 打分 → 和论文榜单对照的表 |
| `scripts/run_machine-zimage.sh` | 新机器一键：环境 → 下载 → 训练包 → 重建 context → 训练 |
| `scripts/train.sh` | 训练入口（含 preflight） |
| `tests/` | CPU 小模型测试（`pytest z-image/tests`） |

## 新机器一键运行

```bash
# 本机（reference_v1 做完后）：打包并公开上传到 jiachengcui888/zimage-ref-v1，会提示输入 HF Write token
bash z-image/scripts/upload_reference_v1.sh

# 新机器（2×H100）
git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
tmux new -s train
bash z-image/scripts/run_machine-zimage.sh --ref-hf jiachengcui888/zimage-ref-v1 --dry-run   # 只检查
bash z-image/scripts/run_machine-zimage.sh --ref-hf jiachengcui888/zimage-ref-v1 --gate      # 20 步测显存和速度
bash z-image/scripts/run_machine-zimage.sh --ref-hf jiachengcui888/zimage-ref-v1             # 训练 + 评测
bash z-image/scripts/run_machine-zimage.sh --resume                                          # 中断后续跑
bash z-image/scripts/run_machine-zimage.sh --eval-only [--eval-ckpts all]                    # 只评测
```

| 阶段 | 内容 |
|---|---|
| 1 env | conda 环境 rdm |
| 2 download | Z-Image-Turbo + 三个 encoder（约 37 GB） |
| 3 eval env | Qwen-Image-Bench 评审环境（ms-swift）+ Q-Judger 权重（约 65 GB）；不装 GenEval、PickScore |
| 4 reference | 从 HF 下载训练包（约 1.6 GB） |
| 5 contexts | 每张卡并行重建 Qwen3 context（约 70 GB，2×H100 约 8–10 分钟） |
| 6 train | FSDP 训练 2 步学生 |
| 7 eval | Qwen-Image-Bench：最终 checkpoint + 老师 9 步、2 步（老师每个 root 只评一次），表格写到 `<run>/qwen_image_bench.md` |

每个阶段做完的会自动跳过；`--gate` 不存 checkpoint，所以不跑第 7 步；`--no-eval` 跳过 3 和 7。
磁盘约 330 GB。

训练包不带 context：新机器上用 Qwen3 按实际长度分桶重建（和 pipeline 补齐到 512 的结果在数学上相同，
只差 bf16 舍入）。实测 4090 约 51 条/秒，12.6 万条在 2×H100 上约 8–10 分钟，约 70 GB。
token 长度和 pipeline 逐条一致（reference_v1 抽 1000 条：1000/1000），数值和渲染时的 context 相差约 0.6%。

## 流程

```bash
# 0. M0：采样器 vs 官方 pipeline，老师几步的 baseline（单卡，24 GB 够用）
PYTHONPATH=z-image python -m zimage_distill.check_teacher --prompts <prompts> --out <dir> --n 8

# 1. 参考集：在 /data/ironman/jiacheng/RDM/z-image 下按 process/README.md 跑完 launch.py all

# 2. 训练（4×H100）。换机器时把整个 render root 拷过去（context 和图在 reference_store 的上一级）
GPUS=4 bash z-image/scripts/train.sh                          # 默认 config
GPUS=4 bash z-image/scripts/train.sh <config> --set steps=5    # smoke
REFERENCE_ROOT=<root>/reference_store GPUS=4 bash z-image/scripts/train.sh
RESUME_FROM=<run>/resume.pth GPUS=4 bash z-image/scripts/train.sh

# 3. 出图评测
PYTHONPATH=z-image python -m zimage_distill.sample --ckpt <run>/step_0002000.pth \
    --prompts <bench prompts> --out <dir> [--all-steps]
```

## 评测：Qwen-Image-Bench（`eval_qib/`）

[Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench)（EMNLP 2026，arXiv 2605.28091）：1000 条 prompt
（每条有中英文两版），评审模型 Q-Judger（Qwen3.6-27B 微调，54.7 GB），每张图按它涉及的一级维度各评一次
（3–5 次，共 4105 次），每个三级细项打 Fail/Pass/Excel/N/A = 0/60/100/不计，逐级平均：
56 个细项 → 23 个二级 → 5 个一级（Quality、Aesthetics、Alignment、Real-world Fidelity、Creative Generation）
→ 每张图的总分 → 1000 张取平均。

```bash
bash z-image/eval_qib/run_eval.sh --setup                                   # 一次：评审环境 + 约 55 GB 权重
bash z-image/eval_qib/run_eval.sh --name s2000 --ckpt <run>/step_0002000.pth
bash z-image/eval_qib/run_eval.sh --name turbo9 --teacher-steps 9           # 老师（官方 9 步）
bash z-image/eval_qib/run_eval.sh --name turbo2 --teacher-steps 2           # 老师直接跑 2 步
bash z-image/eval_qib/run_eval.sh --report s2000 turbo9 turbo2              # 汇总表 runs/report.md
```

- 每条 prompt 生成 1 张 1024 图，所有模型用同一组初始噪声（按 ID 定）。
- **用中文 prompt**：论文榜单的评审全部用的是 `prompt_cn`（从发布的评审原文核对了全部 1000 条），
  `--lang en` 只用于看语言鲁棒性，不能和榜单比。
- 打分直接调用官方 `judge.py`（固定版本，贪心解码、seed 42、thinking 开），没有改动。
  用官方发布的评审原文喂进 `judge.py` 的聚合，复现了论文 Table 2（GPT Image 2 64.69、Qwen Image 49.23、FLUX 2 Pro 54.57）。
- 评审模型一份约 55 GB：一张 80 GB 卡放一份，24 GB 卡要 3 张放一份。脚本按显存自动决定，
  卡上能放几份就把图分成几份**同时**评（`eval_qib/judge_parallel.py`，每份调用一次官方 `judge.py`，
  最后用官方的聚合函数合并）。所以 2×H100 = 2 份并行，约是 1 张的 2 倍快；一份切在多张卡上是逐层轮流算，
  并不比一张卡快。切开再合并已验证：用官方评审原文分成 2 份再合并，仍是 64.69。
- 估计耗时（未实测）：2×H100 约 40 分钟/模型；4×4090（一份切在 3 张上）约 4.5–6.5 小时/模型。
  先用 `--limit 20` 冒烟测出真实速度。

## 和 process/ 参考集的对接

- **条件**：直接用 store 里每条 prompt 的 `context.pt`（就是渲染参考图时 `pipe.encode_prompt` 的输出），
  训练时不加载 Qwen3（每卡省 8 GB）。在线重新编码会和它差约 0.9%（bf16、batch 大小不同），所以优先用预存的。
- **特征预处理逐像素一致**：参考特征来自 1024 PNG → ADM 裁剪到 512（`--img-size 512`）→ encoder。
  训练端的 `reference_pixels` 在前向复现同一条路径：pipeline 的 bf16 后处理、`round(255x)` 存 PNG、
  PIL 的 2× BOX 缩小（两遍，横向再纵向，每遍两两平均、四舍五入）。梯度按 2×2 平均池化直通。
  CPU 测试里和 PIL 路径逐像素相同。不做这一步的话，1024 直接进 encoder 和参考特征差 1.9–3.2%，
  只做平均池化差 1.3–2.0%（同一路径重放的噪声底是 0.3–1.6%），critic 可能学会靠预处理区分真假。
  下采样倍数从 `encoder_features/*.done.json` 里的 `img_size` 自动读出（`zimage.feature_pool: auto`）。
- VAE 解码按 pipeline 的顺序：先把 latent 转成 bf16，再在 bf16 里做 `/scaling + shift`。
- 文本 τ、带宽、row order、prompt_ids 都直接用 store 里的。

## 设计要点

- **噪声按 row id 生成**，形状 `(S, 16, 128, 128)`：第 0 槽是初始噪声，第 k 槽是第 k 步的重新加噪。
  GradCache 两遍、refresh、drift probe 都能精确重放同一条 rollout。
- **checkpoint**：fp32 权重每个 24.6 GB，所以 `save_freq: 500`。`resume.pth` 带两份缓存和两个 critic；
  FSDP 下 optimizer 是分片的、不保存，续训时 Adam 动量重新开始（和原来的 FLUX 流程一样）。

## 已验证的（2026-10-26，本机 4090）

- `pytest z-image/tests`：14 个 CPU 测试通过，覆盖采样器数学、两步之间截断梯度、每步独立的窗口/缓存/critic、
  梯度可按步相加、resume、原训练循环、process/ 布局的 store（context、下采样倍数、渲染配置）、
  `reference_pixels` 和 PIL 路径逐像素一致、分桶重建的 context 和逐条编码完全相同、训练包布局。
- `check_teacher`：同一个 latent 下，我们的 9 步 ODE 和 `ZImagePipeline` 的 latent **逐位相同**（差 0.0）。
  当前 diffusers 版本下 `num_inference_steps=9` 实际是 **9 次** DiT 前向，不是模型卡说的 8 次。
  bf16 的 DiT 对数值很敏感：t 差一个 fp32 ulp，单次前向就差约 4%；所以开 autocast 后得到的样本
  看起来不同，但同样有效。
- 真实权重的反向传播（512 px）：第 1 步的 loss 只回传到初始噪声，第 2 步的 loss 只回传到它的重新加噪，
  截断生效；1024 px 在 24 GB 卡上会 OOM（预期内，要用 H100）。
- process/ 的 smoke_fast store：读取、context、下采样倍数（2×）、渲染配置检查都正常。

## 还没定 / 没验证的

- **reference_v1 每条 prompt 只有 1 张老师图**（candidates=1，keep=1，不筛选）。FLUX 的 store 是每条 4 张：
  每条 prompt 的参考分布只有一个点，配对的吸引会把这条 prompt 的所有 seed 都拉向同一张图，
  多样性可能下降。训练启动时会打 warning。要不要改成多张，需要和世通确认。
- 完整训练还没在 H100 上跑过，显存是估算；放不下的话用 8 卡。
- 真实 pipeline 上 `reference_pixels` 和 PNG→ADM 的逐像素比对还没跑（GPU 被占用）；CPU 测试已逐像素一致。
- lr 沿用 klein-4B 的 2.83e-6，没有针对 6B 调过。
- SigLIP2 文本 τ 只看前 64 个 token，长 prompt 的后半段不进联合核（process/ 也是同样的截断）。
- Qwen-Image-Bench 的评审还没在本机真正跑过一次（GPU 被占用）；每个模型的评审耗时还是估算。
