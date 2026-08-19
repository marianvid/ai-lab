#!/bin/bash
# Serve an NVFP4 model entirely in VRAM with vLLM (Blackwell FP4 tensor cores).
#   $1 model dir   $2 context length   $3 gpu memory fraction   $4+ extra flags
MODEL=$1; CTX=${2:-32768}; UTIL=${3:-0.90}; shift 3 2>/dev/null
export HF_HOME=/models/cache/hf CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
pkill -f "vllm.entrypoints.openai" 2>/dev/null; pkill -f llama-server 2>/dev/null; sleep 6
cd /opt/ai/vllm
nohup ./.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$(basename $MODEL)" \
  --host 127.0.0.1 --port 8098 \
  --max-model-len "$CTX" --gpu-memory-utilization "$UTIL" "$@" \
  > /opt/ai/tools/vllm-run.log 2>&1 &
PID=$!; echo "pid=$PID model=$MODEL ctx=$CTX util=$UTIL extra=$*"
for i in $(seq 1 1800); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/health 2>/dev/null)" = "200" ] \
    && { echo "READY after ${i}s"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; exit 0; }
  if ! kill -0 $PID 2>/dev/null; then
    echo "PROCESS DIED after ${i}s"
    grep -iE "OutOfMemory|ValueError|RuntimeError|not supported|unrecognized" /opt/ai/tools/vllm-run.log | tail -6 | cut -c1-240
    exit 1
  fi
  sleep 1
done
