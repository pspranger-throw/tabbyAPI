# Source before ANY venv-sm75-v154 python invocation.
# Stock exllamav3 v1.5.4 (official cp314/cu128 wheel, prebuilt sm_75+sm_86 ext)
# — no fork, no JIT at runtime. Vars kept so a source-build fallback reuses the
# repo build recipe; TORCH_CUDA_ARCH_LIST/TORCH_EXTENSIONS_DIR must stay fresh
# per base (runbook gotcha 3/14 — never reuse an ext cache across bases).
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/tabbyAPI/venv-sm75-v154/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/gcc-14
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export TORCH_EXTENSIONS_DIR=/home/pageai/.cache/exl3_sm75c_v154_ext

# Explicit content-logging guard: DEBUG would dump full request params (model.py:1717) into console+logs sink — pin INFO.
export TABBY_LOG_LEVEL=INFO
