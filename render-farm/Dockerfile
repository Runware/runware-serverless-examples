FROM python:3.12-slim

ARG BLENDER=4.5.9
ARG SERIES=4.5

# Blender ships as a tarball with its own Python, so there is nothing to pip
# install. What apt is for here is the libraries it links against even when it
# never opens a window: X, GL, Wayland, fontconfig and freetype are all needed
# by a build that only ever renders to a file.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl xz-utils \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
      libxrandr2 libxinerama1 libxcursor1 \
      libwayland-client0 libwayland-cursor0 libwayland-egl1 libdecor-0-0 \
      libgl1 libegl1 libglu1-mesa libsm6 libice6 libglib2.0-0 \
      libfreetype6 libfontconfig1 libtbb12 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://download.blender.org/release/Blender${SERIES}/blender-${BLENDER}-linux-x64.tar.xz" \
      -o /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/blender /usr/local/bin/blender

# Fail the build, not the first request. A missing shared library is invisible
# until Blender is executed, and a build that succeeds and then answers 500 is
# the worst version of this.
RUN blender --version

WORKDIR /app
COPY scene.py server.py /app/

ENV PORT=8080
EXPOSE 8080
CMD ["python", "/app/server.py"]
