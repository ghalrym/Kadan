# The language worker program. Build from services/hikari:
#   docker build -f inference_worker/workers/llm/Dockerfile .
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-toolkit

FROM python:3.12-slim-bookworm AS build
COPY --from=cuda-toolkit /usr/local/cuda/ /usr/local/cuda/
ENV PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libicu-dev libpcre2-dev nlohmann-json3-dev libcurl4-openssl-dev \
      libssl-dev libarchive-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
# Bookworm CMake predates the required CUDA C++20 feature detection.
RUN pip install --no-cache-dir cmake==3.31.6
COPY inference_worker/ /src/inference_worker/
RUN cmake -S /src/inference_worker -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    && cmake --build /build --target kadan-worker-llm --parallel 4 \
    && cmake --install /build --prefix /out --component kadan-worker-llm

# Development keeps the toolchain and build tree so the worker rebuilds
# incrementally from mounted source.
FROM build AS development
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r /app/bin/requirements.txt && cp -a /out /opt/kadan
COPY bin/*.py /app/bin/
WORKDIR /app
CMD ["python", "bin/hikari.py", "dev", "worker", "llm"]

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/kadan/
CMD ["/opt/kadan/bin/kadan-worker-llm"]
