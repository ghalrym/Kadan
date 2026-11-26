FROM python:3.14.9-slim-bookworm
RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY kadan/ ./kadan/
COPY AGENT.md ./AGENT.md
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8630/api/health',timeout=3)"
CMD ["uvicorn", "kadan.api.server:app", "--host", "0.0.0.0", "--port", "8630", "--reload", "--reload-dir", "/app/kadan"]
