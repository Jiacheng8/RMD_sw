> **Copied** on 2026-10-26 from `/data/ironman/jiacheng/RDM/z-image/process` (the code that built
> `reference_v1`), without `artifacts/`. Only paths changed: the RDM checkout defaults to this one
> (`--rdm-repo`), `prompt_build/` stays on the data volume (`PROMPT_ROOT` / `PROMPT_BUILD`, default
> `/data/ironman/jiacheng/RDM/z-image`), and `run_reference.sh` runs from that folder. Where this
> README says `process/`, read `z-image/process_z-image/`. Training reads its output through
> `z-image/zimage_distill/store.py`; see `z-image/README.md`.

# Z-Image-Turbo reference 构建

参考了 `/home/jiacheng/RDM/scripts/preprocess_config.sh`、
`_preprocess_render.py`、`_preprocess_extract.py`、`_preprocess_finalize_store.py`
和 `rdm/sw_lmmd/reference_store.py`。

沿用旧项目的 **24 candidates → top-4 → frozen image features → SW-LMMD store**。

已配置环境后，等 prompt 全部完成，直接从项目根目录运行：

```bash
GPU_IDS=0,1 bash process/run_reference.sh
```

单卡使用 `GPU_IDS=0`。这个入口自动完成下文的冻结输入和全部构建步骤，默认三个 encoder。
`bash process/run_reference.sh --dry-run` 只检查 prompt 是否全部完成，不写文件、不占 GPU。
可以用 `ROOT=/path/to/new-run` 指定输出目录，或在命令末尾添加 `--cpu-offload` 等生图参数。

默认 1024×1024、bf16、CFG=0、9 个 pipeline steps、每卡一个进程，batch=1。
9 steps 沿用官方模型卡的示例；不同 Diffusers 版本的时间步实现可能变化，
这不是对实际 DiT forward 次数的保证。版本和解析后的模型 snapshot 路径写入运行配置。
参数来源：[官方模型卡](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)、
[Diffusers pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py)。

## 1. 环境和模型

在可用的本地 NVIDIA GPU 环境执行；当前开发环境 NVIDIA 驱动不可用，未实测吞吐或显存。
建议单独建 Python 环境，安装匹配驱动的 CUDA PyTorch/torchvision，再执行：

```bash
cd /data/ironman/jiacheng/RDM/z-image
python -m pip install -r process/requirements.txt
python -c 'from diffusers import ZImagePipeline; import torch; print(torch.cuda.is_available())'
```

特征阶段直接导入 `/home/jiacheng/RDM`，不修改该项目；换机器用 `--rdm-repo` 指定其副本。
全量 10 encoder panel 还需要旧 RDM 环境的相应依赖和权重；默认先用
`dinov3_l,siglip2,aimv2_huge`。DINO 等 gated 权重须已有访问权限。

模型可提前缓存（使用 `hf`，compute 节点上可添加 `--offline`）：

```bash
hf download Tongyi-MAI/Z-Image-Turbo
hf download google/siglip2-so400m-patch16-256
```

生产任务建议 `hf download ... --revision <commit>` 并在 render 参数传相同 `--revision` /
`--scorer-revision`。也支持 `--model /absolute/local/model`。本地目录内容须保持不变。
首次真实小样本成功后保存 `python -m pip freeze`，在全量任务复用该环境。
代码记录并检查 torch/diffusers/transformers/numpy/Pillow 版本，防止续跑混用。
`--offline` 只控制 render 的 Hub 加载；特征阶段如需离线，需预缓存所有 encoder 权重并设置
`HF_HUB_OFFLINE=1`。默认 tau 的 open_clip checkpoint 是 `ViT-SO400M-16-SigLIP2-256 / webli`。

## 2. 等 prompt 全部完成后冻结输入

从项目根目录执行。**显式列出正式产出文件**，不通配 `outputs/*.jsonl`，避免混入
tokens、rejects、重试实验。下面列的是当前生产脚本使用的主要输出；如之后增加正式文件，
同步加入 `--inputs`。若部分文件只是实验或重复版本，应去掉。

```bash
python process/prepare.py --out process/artifacts/reference_v1 --shards 0-259 \
  --inputs prompt_build/pilot_claude.jsonl prompt_build/claude_extra.jsonl \
           prompt_build/claude_s0001.jsonl prompt_build/outputs/luna_all.jsonl \
           prompt_build/outputs/sonnet_b1.jsonl prompt_build/outputs/codex_luna_b1.jsonl \
  --dry-run
```

检查通过后去掉 `--dry-run` 才会冻结。`manifest.csv` 和所选 shard 的每个 image_id
必须有合法 prompt 或明确 skipped，**missing 必须为 0**；明确 skipped 不参与生图。
所有 prompt 保留原来的中英文，原始图片只记来源路径，无需原图存在。
同一个 image_id 的同内容记录会去重，不同 prompt/语言/skip 会报冲突，要求显式选择输入，
不自动挑最后一版。损坏 JSON 和仍在写入的半行也会报错。
冻结后的顺序跟随原 `manifest.csv`，后续不会因完成时间或 GPU 数变化而重排。
已有 root 不接受另一套 prompt；更换 prompt、采样参数、模型或版本时用新的 root。

现在 prompt 尚未全部生成，可用**独立目录**做 pilot：

```bash
python process/prepare.py --out process/artifacts/smoke --shards 0-259 \
  --inputs prompt_build/outputs/sonnet_b1.jsonl --allow-incomplete --limit 4
python process/launch.py all --root process/artifacts/smoke --gpu-ids 0 -- \
  --candidates 8 --keep 4 --batch 1
```

`--allow-incomplete` 只用于 pilot，不会等待或启动 caption 任务。
全量流程必须在完成检查通过后运行；这里不自动提交长时间 GPU 作业。

## 3. 多卡生图、筛选、特征

```bash
# 先检查命令，不下载模型、不占 GPU、不写输出。
python process/launch.py all --root process/artifacts/reference_v1 --gpu-ids 0,1 --dry-run

# 执行全部阶段（建议 tmux 中运行）。
python process/launch.py all --root process/artifacts/reference_v1 --gpu-ids 0,1

# 也可以分开执行，以便检查图片、释放推理环境后再提取特征。
python process/launch.py render --root process/artifacts/reference_v1 --gpu-ids 0,1
python process/launch.py features --root process/artifacts/reference_v1 --gpu-ids 0,1

# 最后完整校验图片可解码、context 有限且非零、store 可读取。
python process/features.py check --root process/artifacts/reference_v1
```

显式 `--gpu-ids` 指物理 GPU ID 或 UUID；未指定时沿用 GPU_IDS / CUDA_VISIBLE_DEVICES，
两者都没有时用 0。每个 worker 只看自己的单卡。prompt 轮转分配，encoder 按名称分配。
允许续跑时改变 GPU 数；参数里的 batch/CPU offload/score device 属于运行配置，改变它们需新 root。
日志位于 `<root>/logs/`，某卡失败会停止其他卡；中断会清理 worker 进程组。

显存不足时在**首次运行**加 `--cpu-offload`；可用 `--height 512 --width 512` 降低成本。
默认 scorer 放 CPU，避免和生成模型争显存；CPU 筛选可能成为瓶颈。
显存充足时可用 `--score-device cuda`。不要直接套用旧 FLUX 4090 的吞吐估算。

默认筛选用支持多语言的 SigLIP2 图文 cosine，相同 prompt 内排名；这是对旧 PickScore
筛选的主动适配，**不等价于人类偏好评分**。其文本窗口 64 tokens，长中英文 prompt 会截断，
`selection.json` 记录 `scorer_text_truncated`；生成文本仍用完整 prompt（上限 512 tokens）。
正式全量前应人工检查中英文 pilot 的选图效果，必要时换 scorer 或关闭筛选。

```bash
# 完全复现旧评分器选择；CLIP 文本窗口 77 tokens，中文与长文本效果需单独评估。
python process/launch.py render --root <new-root> --gpu-ids 0 -- --curate pickscore
# 保留全部生成结果，避免筛选偏置；keep 必须等于 candidates。
python process/launch.py render --root <new-root> --gpu-ids 0 -- --curate none --candidates 4 --keep 4
```

与任意自然语言 prompt 对应的图片没有 GenEval 所需的结构化对象/关系标注，
所以本流程不套用旧 GenEval scorer，也不把相似度筛选说成语义正确性验证。

## 4. 文件和断点恢复

```text
<root>/
  prompts.jsonl, prepare.json             冻结 prompt + 校验哈希
  render_config.json                     模型 snapshot、依赖版本、参数
  renders/0000/00000000/
    context.pt                           原生变长 bf16 Z-Image conditioning (L,D)
    text.npy                             筛选器的 normalized text feature（如启用）
    k0.png ... k3.png                    只落盘 top-k
    selection.json                      最后提交：candidate、真实 RNG seed、score、token 长度
  reference_store/
    metadata.json                       最后写入的 store 完成标记
    encoder_features/<encoder>.npy       (N*keep, d_img), fp16, raw features
    text_features.npy                   (N,1152), normalized SigLIP2 tau，重新编码
    prompt_ids.npy, group_ids.npy        reference row → prompt row
    row_order.npy, bandwidths.json
    rows.jsonl, z_context.json           路径和 Z-Image context 来源
```

行号始终是 `r = prompt_id * keep + kept_rank`，k0 分数最高。seed 由
`image_id + candidate + base_seed` 的哈希生成，与 GPU 分片和文件完成顺序无关。
跨硬件/批次不保证像素完全一致。逐 prompt 只留 keep+batch 张候选在内存；
所有文件先写 partial 再替换，最后提交 selection。只有图片可解码且 commit 匹配才跳过。
没有完整 commit 的 prompt 会重做。rank 锁、prompt 锁、启动器锁阻止相同目标并发写入。

特征以 mmap 流式写入，每 batch 刷新后更新进度；中断仅补未提交行。
默认采用旧代码的 ADM 512 裁剪和相同 backbone 预处理，生成图片保留 1024 原尺寸。
sigma 是固定 seed 选取最多 2000 个已存 fp16 feature 的距离中位数 ×0.25，beta=sigma。
相较旧脚本在 fp32 feature 上估计 sigma，存在小量量化差异，带宽中记录了该估计方法。
tau 复用旧 RDM 的 open_clip TextEncoder API 但为新 prompt **全部重算**；不能切旧 COCO 表。
tau 和筛选器属于各自用途，不混作一张表。tau 完整重建失败不会标记为完成。
旧 RDM 的 SigLIP2 tau 编码也使用其原生短文本窗口，长 prompt 的尾部不会进入 tau；
后续训练必须采用相同文本编码器和截断规则。

## 5. 后续训练读取

这是完整的图片+冻结特征 reference。原 `ReferenceFeatureStore(..., require_context=False)`
可读取图文特征。Z-Image 的 conditioning 是变长 Qwen3 单层特征，不能当成 FLUX 的
`(N,48,7680)`。这里提供 `ZImageReferenceStore` 返回实际 token 长度的 tensor 列表，
直接用于 `ZImagePipeline(prompt_embeds=...)`：

```python
# PYTHONPATH=/home/jiacheng/RDM:/data/ironman/jiacheng/RDM/z-image/process
from reference_store import ZImageReferenceStore
store = ZImageReferenceStore("process/artifacts/reference_v1/reference_store",
                            ["dinov3_l", "siglip2", "aimv2_huge"])
embeds = store.generator_context([0, 4], device="cuda")
joint = store.reference_joint_features("siglip2", [0, 4], device="cuda")
# pipe(prompt_embeds=embeds, guidance_scale=0.0, ...)
```

旧 FLUX 训练配置/生成器本身没有因此适配 Z-Image，后续训练需接 Z-Image generator adapter。
移动 reference 库时需一起移动整个 root（context 和图片也在其中），不要仅复制 reference_store。
这是 SW-LMMD store，未额外构建旧 Nyström bundles。

## 6. 规模和离线验证

最终数量由 prepare 报告决定。以 130k 可用 prompt 为例，默认 3.12M 次候选生成、520k 张落盘图；
PNG 空间按 pilot 实测平均大小 ×520k 估算。3 个默认 encoder 共 3712 维，fp16 图像特征约 3.86GB。
context 空间是 `sum(token_length)*hidden_dim*2` bytes，取决于实际文本长度；还需模型缓存空间。
不要用旧流程的 FLUX context 尺寸或速度估算本流程。

```bash
python -m unittest discover -s process -p 'test_*.py' -v
python -m compileall -q process
```

这些验证不下载模型、不执行真实生图；生产前须在可用 GPU 上完成上述小样本 smoke。
