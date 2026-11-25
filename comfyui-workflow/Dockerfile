# The -runtime image, not the -devel one. ComfyUI runs its models as they are
# and compiles nothing, so there is no toolchain to carry.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ARG COMFYUI=0.46.3

# The release tarball rather than a clone, so the image needs no git and the
# version is the one in the ARG.
ADD https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v${COMFYUI}.tar.gz /tmp/comfyui.tar.gz
RUN mkdir -p /opt/ComfyUI \
    && tar -xzf /tmp/comfyui.tar.gz -C /opt/ComfyUI --strip-components=1 \
    && rm /tmp/comfyui.tar.gz

# ComfyUI lists torch with no version, so pip keeps the one the base image
# ships. --break-system-packages because this image's python is system managed
# and pip refuses otherwise.
RUN pip install --no-cache-dir --break-system-packages -r /opt/ComfyUI/requirements.txt

# A custom node goes here, cloned at a pinned commit with its own requirements
# installed, before the check below so that a node that fails to import fails
# the build.

# Fail the build, not the first request. --quick-test-for-ci imports every node
# and exits, and --cpu lets it do that with no GPU in the build.
RUN cd /opt/ComfyUI && python main.py --cpu --quick-test-for-ci

COPY extra_model_paths.yaml /opt/ComfyUI/
WORKDIR /app
COPY server.py workflow.json /app/

ENV PORT=8080

EXPOSE 8080
CMD ["python", "/app/server.py"]
