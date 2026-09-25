# kvsave dev env — full 9-var per-id block (plan §3); only TORCH_EXTENSIONS_DIR differs from prod
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI-v150/venv-kvsave/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=$HOME/.cache/exl3_sm75c_kvsave_ext
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EXL3_MOE_CPU_THREADS=14
export EXL3_MOE_PINNED_ARENA=1
# engine fork revision pin: venv-kvsave (recreated in-repo 2026-09-24, restructure B4b)
# carries ~/exllamav3-sm75 @ sm75-v150-kvsave installed directly into site-packages
# (no PYTHONPATH supersede needed since the recreate; re-stamped per commit;
# 51dee15 prod base 2026-09-22 -> 8f9fdc0 P1 save/restore 2026-09-24 ->
# e968676 rq_new_tokens requeue-accumulation fix 2026-09-24 ->
# 500ce04 P2 save surface (delegates + save pass + zero-stash skip) 2026-09-24 ->
# c913ac7 P2 review-fix round 2026-09-25)
export EXL3_KVSAVE_REV=c913ac7a82537395895cd57b22312df9f1da6caa
