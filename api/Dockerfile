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
COPY . /app/api
RUN ffmpeg -version && ffprobe -version && python -m api.inference.check_install
EXPOSE 8000
CMD ["python", "-m", "api"]
