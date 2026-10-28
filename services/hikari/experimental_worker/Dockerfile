# Separate trial image: it does not replace the native inference workers.
FROM python:3.12.15-slim-bookworm
WORKDIR /app
COPY experimental_worker/requirements.txt /app/experimental_worker/requirements.txt
RUN pip install --no-cache-dir -r /app/experimental_worker/requirements.txt
COPY experimental_worker/ /app/experimental_worker/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Downloads use the explicit pinned artifact downloader, never implicit Hub loads.
CMD ["python", "-m", "experimental_worker"]
