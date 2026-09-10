#!/usr/bin/env bash
# Sequential worker: runs every job in the slice file on the given GPU.
SLICE="$1"
GPU="$2"
export CUDA_VISIBLE_DEVICES="$GPU"
export TORCHINDUCTOR_CACHE_DIR=/tmp/opencode/inductor
cd "$(dirname "$0")/.."
while IFS=';' read -r tag cmd; do
  [ -z "$tag" ] && continue
  echo "[$(date +%H:%M:%S)] START $tag"
  if $cmd > ".v8_logs/${tag}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE $tag"
  else
    echo "[$(date +%H:%M:%S)] FAIL $tag"
  fi
done < "$SLICE"
echo "SLICE COMPLETE"
