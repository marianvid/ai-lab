#!/bin/bash
# Serve any GGUF on llama.cpp.  $1 model file  $2 ctx  $3 extra args (quoted)
MODEL=$1; CTX=${2:-32768}; EXTRA=${3:-}
pkill -f llama-server 2>/dev/null; pkill -f "vllm.entrypoints.openai" 2>/dev/null; sleep 5
T0=$(date +%s.%N)
nohup /opt/ai/llama.cpp/build/bin/llama-server --model "$MODEL" \
  --host 127.0.0.1 --port 8099 --n-gpu-layers 999 --ctx-size "$CTX" \
  --flash-attn on --jinja $EXTRA > /opt/ai/tools/gguf-run.log 2>&1 &
PID=$!
for i in $(seq 1 2400); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health 2>/dev/null)" = "200" ] && {
     T1=$(date +%s.%N); echo "READY load_s=$(echo "$T1-$T0"|bc)"; break; }
  kill -0 $PID 2>/dev/null || { echo "DIED"; grep -iE "error|out of memory|failed" /opt/ai/tools/gguf-run.log | tail -5 | cut -c1-200; exit 1; }
  sleep 0.5
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
