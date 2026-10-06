FROM python:3.12-slim AS h3
ENV PYTHONDONTWRITEBYTECODE=1 SGLANG_BUILD_RUST_EXTS=none
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-h3.txt /tmp/requirements-h3.txt
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r /tmp/requirements-h3.txt \
    && uv pip check --system
RUN ln -s lib /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib64 \
    && ln -s libcudart.so.13 /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so
RUN python -c 'from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator; import comfy_kitchen'

FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt /app/
RUN pip install --no-cache-dir torch==2.11.0 --index-url "$TORCH_INDEX_URL" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
COPY --from=h3 /usr/local /opt/kadan-h3
COPY . /app/api
RUN /opt/kadan-h3/bin/python -c 'from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator; import comfy_kitchen'
RUN ffmpeg -version && ffprobe -version && python -m api.inference.check_install
EXPOSE 8000
CMD ["python", "-m", "api"]
