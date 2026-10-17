FROM node:24-bookworm-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.14-slim-bookworm AS app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TSUCHI_DATA_DIR=/data \
    TSUCHI_STATIC_DIR=/app/web/dist
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 tsuchi \
    && useradd --uid 10001 --gid tsuchi --no-create-home tsuchi \
    && mkdir /data \
    && chown tsuchi:tsuchi /data
COPY tsuchi/ ./tsuchi/
COPY --from=web /build/dist ./web/dist
USER tsuchi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["uvicorn", "tsuchi.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
