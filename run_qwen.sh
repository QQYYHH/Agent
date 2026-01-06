#!/bin/bash
# export VLLM_LOGGING_LEVEL=DEBUG
# --uvicorn-log-level info \
vllm serve /home/qyh/.cache/modelscope/hub/models/Qwen/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --reasoning-parser qwen3 --tensor-parallel-size 2 \
    --max-model-len 20480 --max-num-seqs 2 \
    --enable-log-requests \
    --enable-log-outputs \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
