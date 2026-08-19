#!/bin/bash
# Measure how long a model takes to become usable, and how long the card takes
# to be free again afterwards.
#   $1 = llamacpp | vllm
#   $2 = model path
#   $3 = extra args (quoted)
ENGINE=$1; MODEL=$2; EXTRA=${3:-}
PORT=8099; [ "$ENGINE" = vllm ] && PORT=8098

vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

pkill -f llama-server 2>/dev/null; pkill -f "vllm.entrypoints.openai" 2>/dev/null
for i in $(seq 1 60); do [ "$(vram)" -lt 500 ] && break; sleep 1; done

T0=$(date +%s.%N)
if [ "$ENGINE" = llamacpp ]; then
  nohup /opt/ai/llama.cpp/build/bin/llama-server --model "$MODEL" \
    --host 127.0.0.1 --port $PORT --n-gpu-layers 999 --ctx-size 32768 \
    --flash-attn on --jinja $EXTRA > /opt/ai/tools/timeload-engine.log 2>&1 &
else
  export HF_HOME=/models/cache/hf CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
  cd /opt/ai/vllm
  nohup ./.venv/bin/python -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --served-model-name m --host 127.0.0.1 --port $PORT --max-model-len 32768 \
    $EXTRA > /opt/ai/tools/timeload-engine.log 2>&1 &
fi
PID=$!

for i in $(seq 1 3000); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
  kill -0 $PID 2>/dev/null || { echo "LOAD=FAILED"; tail -4 /opt/ai/tools/timeload-engine.log | cut -c1-200; exit 1; }
  sleep 0.25
done
T1=$(date +%s.%N)
USED=$(vram)
echo "LOAD_S=$(echo "$T1 - $T0" | bc) VRAM_MIB=$USED"

# unload: from the kill signal until the card reports the memory back
T2=$(date +%s.%N)
kill $PID 2>/dev/null
for i in $(seq 1 600); do [ "$(vram)" -lt 500 ] && break; sleep 0.2; done
T3=$(date +%s.%N)
echo "UNLOAD_S=$(echo "$T3 - $T2" | bc)"
