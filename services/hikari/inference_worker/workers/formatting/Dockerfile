# The transcript formatting worker program (S1-mini): CPU only, no CUDA.
# Build from services/hikari:
#   docker build -f inference_worker/workers/formatting/Dockerfile .
FROM python:3.12.15-slim-bookworm AS build
RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make g++ libopenblas-dev libicu-dev libpcre2-dev nlohmann-json3-dev \
      libcurl4-openssl-dev libssl-dev libarchive-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cmake==3.31.6
COPY inference_worker/ /src/inference_worker/
RUN cmake -S /src/inference_worker -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DKADAN_ENABLE_CUDA=OFF -DKADAN_ENABLE_WHISPER_BLAS=ON \
    && cmake --build /build --target kadan-worker-formatting --parallel 4 \
    && cmake --install /build --prefix /out --component kadan-worker-formatting

FROM build AS runtime
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r /app/bin/requirements.txt && cp -a /out /opt/kadan
COPY bin/*.py /app/bin/
WORKDIR /app
CMD ["python", "bin/hikari.py", "run", "worker", "formatting"]
