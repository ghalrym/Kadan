FROM python:3.14.9-slim-bookworm AS app
RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
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
