# 参考侧消融：SW-LMMD 吸引项用哪一侧的参考

比较滑窗 MMD 吸引项的三种参考侧。三种模式下其余一切都与现有方法相同：生成侧窗口、缓存、只对最后 B 行回传、排斥项、K、B、学习率、步数、编码器、σ、β、特征归一化、种子，以及 GAN critic。critic 在三种模式下都用窗口自己的参考行训练。

每个编码器的吸引项是 `L_att = -(2/B) Σ_{i∈A} [括号]`，其中 A 是窗口最后 B 行（新样本）。三种模式只差括号里的量：

| `ref_mode` | 括号 | 参考行 |
|---|---|---|
| `local`（我们的方法） | `(1/K) Σ_{j∈W} k(x_i, y_j)` | 窗口内同一批 K 行的参考特征 |
| `random` | `(1/K) Σ_{j∈R_t} k(x_i, y_j)` | 每步从全部 N 行里无放回均匀抽 K 行，与窗口无关；每行用它**自己的 caption** 拼文本特征 |
| `global` | `ψ(x_i)ᵀ μ̄` | Nyström：m=4096 个 k-means landmark，`ψ(x)=K_mm^{-1/2} k(L,x)`，`μ̄=(1/N)Σ_n ψ(y_n)` 在全部 N 行上算一次后固定 |

基础配置是 `configs/sw_lmmd_train_h100_2gpu_gan.yaml`（K128 · GAN，**不分组**），只用 COCO 参考，原样沿用：K=128、B=32（不分组：每步 32 张新图来自 32 个不同的 prompt）、row critic（自适应权重 1.5）、fp32 battery、micro 4、lr 2.83e-6（预热 100 步后余弦降到 0.1×）、2000 步、编码器 dinov3_l / siglip2 / aimv2_huge。

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | `ref_ablation:` 配置块，未知键直接报错 |
| `ref_loss.py` | **复制自** `ExactLocalMMD.active_force`（rdm/sw_lmmd/local_mmd.py，d5b2e9a），只改吸引项的参考侧，原运算顺序不变 |
| `random_reference.py` | random 模式的行抽样器：私有 `numpy.random.Generator`，不碰任何全局随机数；可存档、可续训 |
| `nystrom_global.py` | global 模式：读取 landmark 文件，加载时核对参考库指纹（σ、β、维度、行数、抽样行哈希）；提供 `ψ(x)ᵀμ̄` 和显式的 `ψ(x)` |
| `trainer.py` | `RefAblationTrainer(SWLMMDTrainer)`：`step()` **复制自** `SWLMMDTrainer.step`，只改 `# [ref-ablation]` 标出的几行；另有训练前的三模式吸引项诊断 |
| `train.py` | 训练入口，`train/main` **复制自** `rdm/sw_lmmd/launch.py`。trainer 由原版 `build_trainer` 构造后接管 |
| `build_landmarks.py` | 离线构建 Nyström 文件，用我们自己的参考库、σ、β 和 `[φ\|β·τ]` 拼接；附带真实数据上的近似误差报告 |
| `configs/ref_ablation_base.yaml` | 三个配置共用的部分 |
| `configs/ref_{local,random,global}.yaml` | 三种模式，`diff` 只有 `ref_mode` 一行 |
| `run.sh` | 单次运行：`run.sh <mode> <seed>` |
| `run_ablation-exp1.sh` | **exp 1 = random**，新机器一条命令跑完，仿照 `scripts/run_machine-new11.sh` |
| `run_ablation-exp2.sh` | **exp 2 = global（all）**，同上 |
| `_run_ablation.sh` | 两个 exp 共用的主体（类似 `scripts/_run_experiment.sh`）。依次：用 `new_machine.sh --no-train` 准备环境、下载、Qwen context、GenEval → 从 HF 下载 landmark 并校验 → 10 步检查 → 训练 → 评测全部 checkpoint |
| `smoke.sh` | 现有 pipeline 和三种模式各跑几步，比较每步耗时和显存 |
| `tests/` | 测试 A–F（`python -m pytest experiments/ref_ablation/tests`） |

## 运行

新机器（2×H100），一条命令完成环境、下载、训练和评测：

```bash
git clone -b ref-ablation https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
echo hf_xxx > .hf_token && chmod 600 .hf_token     # 需要能读私有 dataset jiachengcui888/sw-rdm-reference-store
bash experiments/ref_ablation/run_ablation-exp1.sh   # random，seed 0
bash experiments/ref_ablation/run_ablation-exp2.sh   # global，seed 0
#    --dry-run 只看计划；--resume 中断后续跑；--seed N 换种子；--no-eval 只训练
```

landmark 文件已上传到私有 dataset 的 `ref_ablation/nystrom_m4096/`，附 SHA256SUMS，脚本会自动下载并校验。

已经准备好的机器上，也可以分步运行：

```bash
# 1. global 模式需要先构建 landmark（每个编码器一份，单卡几分钟）
CUDA_VISIBLE_DEVICES=0 python -m experiments.ref_ablation.build_landmarks \
    experiments/ref_ablation/configs/ref_global.yaml       # -> ref_ablation.nystrom_dir

# 2. 训练：一个模式、一个种子；2 x H100
bash experiments/ref_ablation/run.sh local 0
bash experiments/ref_ablation/run.sh random 0
bash experiments/ref_ablation/run.sh global 0
#    检查用：只打印配置，不启动
bash experiments/ref_ablation/run.sh global 1 --dry-run
#    只做训练前诊断（bootstrap 后打印三种吸引项，然后退出）
bash experiments/ref_ablation/run.sh global 0 --diagnose-only

# 换机器时：
REFERENCE_ROOT=... OUTPUT_DIR=... NYSTROM_DIR=... bash experiments/ref_ablation/run.sh random 2
```

- 种子同时设置 `seed`（每行的生成噪声、torch 全局种子）和 `ref_ablation.ref_seed`（random 抽样器）。窗口顺序 `window.order_seed` 固定为 3407，所以同一种子下三种模式看到的窗口和噪声完全相同。想让窗口顺序也随种子变，设 `ORDER_SEED`。
- 运行目录：`<output_dir>/<mode>-ungrouped/`，例如 `global-ungrouped`；seed 0 以外的种子会加 `-s<seed>`，比如 `global-ungrouped-s1`。
- landmark 文件约 0.5 GB，可以复制到 H100 机器（用 `NYSTROM_DIR` 指过去），也可以在那边重新构建。加载时会校验参考库指纹：参考库不同就报错，不会用错。

## 日志

`train_log.jsonl` 每步都有现有方法的全部字段，含义不变；`<enc>/attraction` 现在是**本模式实际训练的**吸引项。另外新增下列字段（都是跨 rank 的平均值，与卡数无关）：

| 字段 | 含义 |
|---|---|
| `ref/<enc>/attraction`、`repulsion`、`force` | 吸引项、排斥项、`force = 2(rep − att)` |
| `ref/<enc>/force_weighted` | `λ_e · force_e`（λ_e = 1/3），即求和后回传的量。SW-LMMD 没有 iRDM 的 per-encoder self-normalization，编码器权重固定为 1/E |
| `ref/<enc>/grad_attraction`、`grad_repulsion`、`grad_force` | 各项对新样本图像特征的梯度范数（force 单位），量级与数值偏移无关 |
| `ref/<enc>/grad_cos_att_rep` | 吸引梯度与排斥梯度的余弦 |
| `ref/<enc>/attraction_local` | 窗口内 local 吸引项（各模式都记，作为共同标尺） |
| `ref/<enc>/attraction_global` | Nyström 全局吸引项（有 landmark 文件时记） |
| `ref/rows_sha1` | 本步吸引项所用行的哈希；random 模式下每步都不同 |
| `mem_peak_gb`、`mem_reserved_gb` | 本步 GPU 峰值显存 |

`<enc>/mmd2`、`raw_mmd2`、`group_k_*` 仍然对窗口自己的参考计算，三种模式之间可以直接比较。

训练前（`pre_diagnostic: true`）：bootstrap 之后，用第一步要用的同一批 B 个样本（同一 checkpoint、同一窗口）打印三种吸引项，结果存到 `pre_diagnostic_step*.json`：
- local，并拆成同 prompt 部分和其余部分；
- random，8 次抽样的均值 ± 标准差；
- global 的 Nyström 值，以及对全部 N 行精确计算的值、两者的相对误差和梯度余弦；
- 排斥项，用来对比量级。
