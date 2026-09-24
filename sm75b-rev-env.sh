# Source before ANY venv-sm75 python invocation (reversed GPU order: 3090=cuda:0, 2060S=cuda:1)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI/venv-sm75/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=/home/pageai/.cache/exl3_sm75b_ext
export CUDA_VISIBLE_DEVICES=1,0
