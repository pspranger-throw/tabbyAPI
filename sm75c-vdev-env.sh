# Source before ANY venv-sm75-vdev python invocation (JIT build env for exllamav3 sm75-vdev @ upstream/dev f4db698 + 19 fork commits)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI/venv-sm75-vdev/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=/home/pageai/.cache/exl3_sm75c_vdev_ext
# engine fork revision pin: venv-sm75-vdev is a site-packages copy (save-path
# rev-proof, kvstore.engine_fork_revision) — a venv copy must carry this pin.
# 516400f = sm75-vdev tip (fla chunk_o tiering restored; lineage tag 1.5.1+vdev2189a00).
export EXL3_KVSAVE_REV=516400fc91564424d2c1558bcaf076ba1b54c05e
# per-venv rev stamp (venv-sm75-vdev), written by the venv-install procedure.
# SSOT = models-serve registry .env.EXL3_KVSAVE_REV (operationally binding).
_stamp="$(dirname "${BASH_SOURCE[0]:-$0}")/.kvsave-rev-venv-sm75-vdev.local"
if [ -f "$_stamp" ]; then . "$_stamp"; export EXL3_KVSAVE_REV; fi
unset _stamp
