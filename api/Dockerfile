# Build from the repository root; .dockerignore admits API/native source only.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS cuda-toolkit

FROM python:3.12-slim-bookworm AS native-build
COPY --from=cuda-toolkit /usr/local/cuda-12.6 /usr/local/cuda-12.6
ENV PATH=/usr/local/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libopenblas-dev libicu-dev libpcre2-dev nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*
# Bookworm CMake predates CUDA 12 C++20 feature detection.
RUN pip install --no-cache-dir cmake==3.31.6
COPY native/ /src/native/
RUN cmake -S /src/native -B /build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF -DKADAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DKADAN_ENABLE_DECISIONS=ON -DKADAN_ENABLE_H3_GENERATION=ON -DKADAN_ENABLE_WHISPER_BLAS=ON -DKADAN_ENABLE_TTS_WORKER=ON -DKADAN_ENABLE_IMAGE_GENERATION=ON -DKADAN_ENABLE_IMAGE_BLAS=ON \
    && cmake --build /build --target kadan-model-worker kadan-decision-worker kadan-h3-worker kadan-whisper-worker kadan-tts-worker kadan-image-worker --parallel 2 \
    && cmake --install /build --prefix /out --component model-runtime \
    && cmake --install /build --prefix /out --component decision-runtime \
    && cmake --install /build --prefix /out --component h3-runtime \
    && cmake --install /build --prefix /out --component whisper-runtime \
    && cmake --install /build --prefix /out --component tts-runtime \
    && cmake --install /build --prefix /out --component image-runtime \
    && /out/bin/kadan-h3-worker --capabilities | python -c 'import json,sys; assert json.load(sys.stdin)["cuda"] is True'

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY api/requirements*.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY api/ /app/api
COPY --from=native-build /out/ /opt/kadan/
RUN test -x /opt/kadan/bin/kadan-model-worker \
    && LD_LIBRARY_PATH=/opt/kadan/lib ldd /opt/kadan/libexec/kadan-model-worker > /tmp/model-linkage \
    && ! grep "not found" /tmp/model-linkage \
    && test -x /opt/kadan/bin/kadan-decision-worker \
    && LD_LIBRARY_PATH=/opt/kadan/lib ldd /opt/kadan/libexec/kadan-decision-worker > /tmp/decision-linkage \
    && ! grep "not found" /tmp/decision-linkage \
    && test -x /opt/kadan/bin/kadan-h3-worker \
    && LD_LIBRARY_PATH=/opt/kadan/lib ldd /opt/kadan/libexec/kadan-h3-worker > /tmp/h3-linkage \
    && ! grep "not found" /tmp/h3-linkage
RUN ffmpeg -version && ffprobe -version && python -m api.inference.check_install
EXPOSE 8000
CMD ["python", "-m", "api"]
