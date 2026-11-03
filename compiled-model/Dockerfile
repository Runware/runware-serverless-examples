# A -devel image, not a -runtime one. torch.compile hands its kernels to Triton,
# and Triton shells out to a C compiler at runtime, so the toolchain has to be in
# the image. This is the reason this example is a container at all.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel

# --break-system-packages because this image's python is system managed and pip
# refuses otherwise (PEP 668). Inside a container there is no other environment
# to break.
RUN pip install --no-cache-dir --break-system-packages timm==1.0.30 pillow==13.0.0

# Set here rather than in code: torch reads all three as it loads, and the first
# import anywhere in the process has already fixed them.
ENV HF_HOME=/weights/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/weights/inductor \
    TRITON_CACHE_DIR=/weights/triton \
    PORT=8080

# Fail the build, not the first request. There is no GPU here, so this checks the
# imports rather than the model: enough to catch a wheel that does not load.
RUN python -c "import torch, timm; from PIL import Image; print(torch.__version__, timm.__version__)"

WORKDIR /app
COPY server.py /app/

EXPOSE 8080
CMD ["python", "/app/server.py"]
