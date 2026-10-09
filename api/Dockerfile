# Build from the repository root; .dockerignore admits API/native source only.
FROM python:3.12-slim AS decision-build
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake g++ libopenblas-dev libicu-dev libpcre2-dev nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY native/ /src/native/
RUN cmake -S /src/native -B /build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF -DKADAN_ENABLE_CUDA=OFF -DKADAN_ENABLE_DECISIONS=ON \
    && cmake --build /build --target kadan-decision-worker --parallel 2 \
    && cmake --install /build --prefix /out --component decision-runtime

FROM python:3.12-slim AS h3
ENV PYTHONDONTWRITEBYTECODE=1 SGLANG_BUILD_RUST_EXTS=none
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY api/requirements-h3.txt /tmp/requirements-h3.txt
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r /tmp/requirements-h3.txt \
    && uv pip check --system
RUN ln -s lib /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib64 \
    && ln -s libcudart.so.13 /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so
RUN python -c 'from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator; import comfy_kitchen'

FROM h3
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends g++ sox \
    && rm -rf /var/lib/apt/lists/*
COPY api/requirements*.txt /app/
COPY api/qwen_tts_compat /app/api/qwen_tts_compat
RUN uv pip install --system --no-cache torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install --system --no-cache -r requirements.txt && uv pip check --system
COPY api/ /app/api
COPY --from=decision-build /out/ /opt/kadan/
RUN test -x /opt/kadan/bin/kadan-decision-worker \
    && LD_LIBRARY_PATH=/opt/kadan/lib ldd /opt/kadan/libexec/kadan-decision-worker > /tmp/decision-linkage \
    && ! grep "not found" /tmp/decision-linkage
RUN ffmpeg -version && ffprobe -version && python -m api.inference.check_install
EXPOSE 8000
CMD ["python", "-m", "api"]
