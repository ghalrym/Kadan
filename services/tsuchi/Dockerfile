FROM python:3.14-slim-bookworm AS app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TSUCHI_DATA_DIR=/data
WORKDIR /app
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir /data
COPY tsuchi/ ./tsuchi/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["uvicorn", "tsuchi.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/tsuchi"]
