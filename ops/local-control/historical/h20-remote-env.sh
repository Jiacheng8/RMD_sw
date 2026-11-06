export RDM_ROOT=/share-evpfs/shaoshitong/rdm-sets
export PATH=/share-evpfs/shaoshitong/rdm-tools/bin:/share-evpfs/yixinshen/miniconda3/bin:"$PATH"
export CUDA_VISIBLE_DEVICES=4,5
export CONDA_ENVS_PATH=/share-evpfs/shaoshitong/conda-envs
export CONDA_REGISTER_ENVS=false
export CONDATOS="$RDM_ROOT/.cache/conda-tos"
export CONDA_PKGS_DIRS="$RDM_ROOT/.cache/conda-pkgs"
export PIP_CACHE_DIR="$RDM_ROOT/.cache/pip"
export TMPDIR="$RDM_ROOT/.cache/tmp"
export XDG_CACHE_HOME="$RDM_ROOT/.cache"
export TRITON_CACHE_DIR="$RDM_ROOT/.cache/triton"
export TRITON_DUMP_DIR="$RDM_ROOT/.cache/triton/dump"
export TRITON_OVERRIDE_DIR="$RDM_ROOT/.cache/triton/override"
export CUDA_CACHE_PATH="$RDM_ROOT/.cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$RDM_ROOT/.cache/torchinductor"
export TORCH_HOME="$RDM_ROOT/.cache/torch"
export HF_HOME="$RDM_ROOT/.cache/huggingface"
export HTTP_PROXY=http://127.0.0.1:18791
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export all_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export WANDB_CACHE_DIR="$RDM_ROOT/.cache/wandb"
export WANDB_CONFIG_DIR="$RDM_ROOT/.cache/wandb-config"
export WANDB_DATA_DIR="$RDM_ROOT/.cache/wandb-data"
export WANDB_ENTITY=sw-lmmd
export WANDB_PROJECT=zimage-distill
IFS= read -r WANDB_API_KEY < "$RDM_ROOT/.secrets/wandb-api-key"
export WANDB_API_KEY
if test -s "$RDM_ROOT/control/wandb-mode"; then
  IFS= read -r WANDB_MODE < "$RDM_ROOT/control/wandb-mode"
  export WANDB_MODE
fi
if test -s "$RDM_ROOT/control/hf-endpoint"; then
  IFS= read -r HF_ENDPOINT < "$RDM_ROOT/control/hf-endpoint"
  export HF_ENDPOINT
fi
