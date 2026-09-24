# kvsave dev env — full 9-var per-id block (plan §3); only TORCH_EXTENSIONS_DIR differs from prod
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI/venv-kvsave/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=$HOME/.cache/exl3_sm75c_kvsave_ext
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EXL3_MOE_CPU_THREADS=14
export EXL3_MOE_PINNED_ARENA=1
# engine fork revision pin: venv-kvsave is a cp -a of prod; P0.0(a) verified byte-identity with
# ~/exllamav3-sm75 @ sm75-v150 / 51dee159bb39acbbb176b69eeb9eaf0e0ea2d273 on 2026-09-22
export EXL3_KVSAVE_REV=51dee159bb39acbbb176b69eeb9eaf0e0ea2d273
