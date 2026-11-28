# SGLang's own image, with nothing added to it. The platform finds the SGLang inside
# and attaches its own serving layer, so there is no integration line here.
FROM lmsysorg/sglang:v0.5.21

# HF_HOME points at the volume the deploy mounts. Without it the weights land on
# the sandbox filesystem and are fetched again on every cold start.
#
# PORT is the one container.yaml declares. 30000 is SGLang's own default and the
# platform does not hold it, so there is no port to move here.
ENV MODEL=Qwen/Qwen3-4B-Instruct-2507 \
    CONTEXT_LENGTH=8192 \
    MAX_RUNNING_REQUESTS=64 \
    MEM_FRACTION_STATIC=0.3 \
    HF_HOME=/weights \
    PORT=30000

EXPOSE 30000

# A plain `sglang serve`. ENTRYPOINT replaces the base image's own, so the argv the
# platform reads is the one written here. `sh -c` so the variables above are read at
# container start, and `exec` so sglang receives the signals the platform sends.
#
# --host 0.0.0.0 because SGLang binds the loopback by default, and the platform
# reaches the port from outside the sandbox's loopback.
ENTRYPOINT ["/bin/sh", "-c", \
  "exec sglang serve --model-path \"$MODEL\" --host 0.0.0.0 --port \"$PORT\" --context-length \"$CONTEXT_LENGTH\" --max-running-requests \"$MAX_RUNNING_REQUESTS\" --mem-fraction-static \"$MEM_FRACTION_STATIC\""]
