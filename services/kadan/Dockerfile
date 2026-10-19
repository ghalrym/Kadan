FROM python:3.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY kadan/ ./kadan/
COPY AGENT.md ./AGENT.md
USER 10001:10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8630/api/health',timeout=3)"
CMD ["uvicorn", "kadan.api.server:app", "--host", "0.0.0.0", "--port", "8630"]
