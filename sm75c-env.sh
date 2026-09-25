# Source before ANY venv-sm75-v150 python invocation (JIT build env for exllamav3-sm75 @ sm75-v150, v1.5.0 base)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI-v150/venv-sm75-v150/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=/home/pageai/.cache/exl3_sm75c_ext
# engine fork revision pin: venv-sm75-v150 is a site-packages copy (save-path
# rev-proof, kvstore.engine_fork_revision) — a venv copy must carry this pin.
# 548dde0 = sm75-v150 merge of the kvsave workstream (P4 flip 2026-09-25).
export EXL3_KVSAVE_REV=548dde0dd88cddf6e644d2073fc70e9488784527
# per-venv rev stamp (venv-sm75-v150), written by the venv-install procedure:
#   printf 'EXL3_KVSAVE_REV=%s\n' "$REV" > "$(dirname "${BASH_SOURCE[0]}")/.kvsave-rev-venv-sm75-v150.local"
# SSOT = models-serve registry .env.EXL3_KVSAVE_REV (operationally binding).
# Sourced AFTER the literal export above so a present stamp wins; absent stamp
# = the committed literal pin (keep both in sync until the next venv install).
_stamp="$(dirname "${BASH_SOURCE[0]}")/.kvsave-rev-venv-sm75-v150.local"
if [ -f "$_stamp" ]; then . "$_stamp"; export EXL3_KVSAVE_REV; fi
unset _stamp
