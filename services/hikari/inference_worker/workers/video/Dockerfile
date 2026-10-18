# The video worker program (H3 video with audio). Build from services/hikari:
#   docker build -f inference_worker/workers/video/Dockerfile .
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-toolkit

FROM python:3.12-slim-bookworm AS build
COPY --from=cuda-toolkit /usr/local/cuda/ /usr/local/cuda/
ENV PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libopenblas-dev libicu-dev libpcre2-dev nlohmann-json3-dev \
      libcurl4-openssl-dev libssl-dev libarchive-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cmake==3.31.6
COPY inference_worker/ /src/inference_worker/
RUN cmake -S /src/inference_worker -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DKADAN_ENABLE_H3_GENERATION=ON \
    && cmake --build /build --target kadan-worker-video --parallel 4 \
    && cmake --install /build --prefix /out --component kadan-worker-video

FROM build AS development
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r /app/bin/requirements.txt && cp -a /out /opt/kadan
COPY bin/*.py /app/bin/
WORKDIR /app
CMD ["python", "bin/hikari.py", "dev", "worker", "video"]

# Only the video worker encodes media, so only it carries FFmpeg.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/kadan/
CMD ["/opt/kadan/bin/kadan-worker-video"]
