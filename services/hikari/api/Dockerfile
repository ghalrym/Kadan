# Build from the repository root; .dockerignore admits API/inference worker source only.
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-toolkit

FROM python:3.12-slim-bookworm AS worker-build
COPY --from=cuda-toolkit /usr/local/cuda/ /usr/local/cuda/
ENV PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libopenblas-dev libicu-dev libpcre2-dev nlohmann-json3-dev libcurl4-openssl-dev libssl-dev libarchive-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
# Bookworm CMake predates the required CUDA C++20 feature detection.
RUN pip install --no-cache-dir cmake==3.31.6
# Python is used to acquire the supported C++ library during the build only.
# The worker calls ATen directly and never imports torch or launches Python.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
ENV CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/site-packages/torch
COPY inference_worker/ /src/inference_worker/
RUN set -eu; cmake -S /src/inference_worker -B /build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DKADAN_ENABLE_DECISIONS=ON -DKADAN_ENABLE_H3_GENERATION=ON -DKADAN_ENABLE_WHISPER_BLAS=ON -DKADAN_ENABLE_TTS_WORKER=ON -DKADAN_ENABLE_IMAGE_GENERATION=ON \
    && cmake --build /build --target kadan-inference-worker --parallel 2 \
    && export LD_LIBRARY_PATH="$(python -c 'import pathlib; print(":".join(str(p) for p in pathlib.Path("/usr/local/lib/python3.12/site-packages/nvidia").glob("*/lib")))'):$LD_LIBRARY_PATH" \
    && cmake --install /build --prefix /out --component inference-runtime \
    && cp -a /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn*.so* /out/lib/ \
    && cp -a /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc*.so* /out/lib/ \
    && mkdir -p /out/share/kadan/licenses \
    && cp -r /usr/local/lib/python3.12/site-packages/torch-*.dist-info/licenses /out/share/kadan/licenses/torch \
    && for p in /usr/local/lib/python3.12/site-packages/nvidia*.dist-info; do \
         if [ -d "$p/licenses" ]; then cp -r "$p/licenses" "/out/share/kadan/licenses/$(basename "$p")"; fi; \
       done

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY api/requirements*.txt /app/
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r bin/requirements.txt && pip check
COPY api/ /app/api
COPY bin/*.py /app/bin/
COPY --from=worker-build /out/ /opt/kadan/
RUN test -x /opt/kadan/bin/kadan-inference-worker \
    && LD_LIBRARY_PATH=/opt/kadan/lib ldd /opt/kadan/libexec/kadan-inference-worker > /tmp/inference-worker-linkage \
    && ! grep "not found" /tmp/inference-worker-linkage
RUN ffmpeg -version && ffprobe -version && python bin/hikari.py check install
EXPOSE 8000
CMD ["python", "-m", "api"]
