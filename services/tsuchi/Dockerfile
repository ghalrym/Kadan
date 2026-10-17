FROM python:3.14-slim-bookworm AS app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TSUCHI_DATA_DIR=/data
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 tsuchi \
    && useradd --uid 10001 --gid tsuchi --no-create-home tsuchi \
    && mkdir /data \
    && chown tsuchi:tsuchi /data
COPY tsuchi/ ./tsuchi/
USER tsuchi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["uvicorn", "tsuchi.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
