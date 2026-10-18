# The image worker program (Qwen Image on LibTorch/ATen). Build from services/hikari:
#   docker build -f inference_worker/workers/image/Dockerfile .
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-toolkit

FROM python:3.12-slim-bookworm AS build
COPY --from=cuda-toolkit /usr/local/cuda/ /usr/local/cuda/
ENV PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libicu-dev libpcre2-dev nlohmann-json3-dev libcurl4-openssl-dev \
      libssl-dev libarchive-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cmake==3.31.6
# Python only acquires the supported C++ library during the build. The worker
# calls ATen directly and never imports torch or launches Python.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
ENV CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/site-packages/torch
COPY inference_worker/ /src/inference_worker/
RUN set -eu; cmake -S /src/inference_worker -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DKADAN_ENABLE_IMAGE_GENERATION=ON \
    && cmake --build /build --target kadan-worker-image --parallel 4 \
    && export LD_LIBRARY_PATH="$(python -c 'import pathlib; print(":".join(str(p) for p in pathlib.Path("/usr/local/lib/python3.12/site-packages/nvidia").glob("*/lib")))'):$LD_LIBRARY_PATH" \
    && cmake --install /build --prefix /out --component kadan-worker-image \
    && cp -a /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn*.so* /out/lib/ \
    && cp -a /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc*.so* /out/lib/ \
    && mkdir -p /out/share/kadan/licenses \
    && cp -r /usr/local/lib/python3.12/site-packages/torch-*.dist-info/licenses /out/share/kadan/licenses/torch \
    && for p in /usr/local/lib/python3.12/site-packages/nvidia*.dist-info; do \
         if [ -d "$p/licenses" ]; then cp -r "$p/licenses" "/out/share/kadan/licenses/$(basename "$p")"; fi; \
       done

FROM build AS development
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r /app/bin/requirements.txt && cp -a /out /opt/kadan
COPY bin/*.py /app/bin/
WORKDIR /app
CMD ["python", "bin/hikari.py", "dev", "worker", "image"]

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/kadan/
CMD ["/opt/kadan/bin/kadan-worker-image"]
