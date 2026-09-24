# Source before ANY venv-sm75 python invocation (JIT build env for exllamav3-sm75)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI/venv-sm75/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=/home/pageai/.cache/exl3_sm75_ext
