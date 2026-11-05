# vLLM's own image, with nothing added to it. The platform finds the vLLM inside
# and attaches its own serving layer, so there is no integration line here.
FROM vllm/vllm-openai:v0.28.0

# HF_HOME points at the volume the deploy mounts. Without it the weights land on
# the sandbox filesystem and are fetched again on every cold start.
#
# PORT is the one container.yaml declares. Not 8000, which the platform binds
# inside the pod along with 8888, 8889 and 9090.
ENV MODEL=Qwen/Qwen3-4B-Instruct-2507 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.3 \
    HF_HOME=/weights \
    PORT=8080

EXPOSE 8080

# A plain `vllm serve`. `sh -c` so the variables above are read at container
# start, and `exec` so vllm receives the signals the platform sends.
ENTRYPOINT ["/bin/sh", "-c", \
  "exec vllm serve \"$MODEL\" --port \"$PORT\" --max-model-len \"$MAX_MODEL_LEN\" --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\""]
