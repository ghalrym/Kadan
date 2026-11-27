# Unified 100% C++20/CUDA multi-workload worker image (no Python runtime).
# Supports lfm3_llm, lfm3_vision, lfm3_audio, formatting, llm,
# speech, and tts via `kadan-worker --workload=<kind>`.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG KADAN_CUDA_ARCHS="86"
ARG KADAN_ALLOW_MULTI_ARCH="OFF"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libcurl4-openssl-dev \
    libssl-dev \
    libarchive-dev \
    libpng-dev \
    libicu-dev \
    libpcre2-dev \
    libopenblas-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY inference_worker /src/inference_worker

RUN cmake -S /src/inference_worker -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DKADAN_ENABLE_CUDA=ON \
    -DKADAN_ALLOW_MULTI_ARCH=${KADAN_ALLOW_MULTI_ARCH} \
    -DCMAKE_CUDA_ARCHITECTURES="${KADAN_CUDA_ARCHS}" \
    -DKADAN_ENABLE_TTS_WORKER=ON \
    -DKADAN_ENABLE_WHISPER_BLAS=ON \
    && cmake --build /build --target kadan-worker -j"$(nproc)" \
    && cmake --install /build --component kadan-worker --prefix /opt/kadan

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libarchive13t64 \
    libcurl4t64 \
    libicu74 \
    libopenblas0 \
    libpcre2-8-0 \
    libpng16-16t64 \
    libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/kadan /opt/kadan
ENV PATH="/opt/kadan/bin:/opt/kadan/libexec:${PATH}"
ENV KADAN_WORKER_DATA="/opt/kadan/share/kadan"
ENTRYPOINT ["/opt/kadan/bin/kadan-worker"]
