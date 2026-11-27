# Build with the hikari, tsuchi, kadan and tracing named contexts (see compose.yaml).
# The API coordinates Hikari's worker containers and runs no inference itself.
FROM python:3.14.9-slim-bookworm
RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TSUCHI_DATA_DIR=/data
WORKDIR /app
# Hikari's subscription language providers run through the user's own Claude Code and Codex CLIs.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code@2.1.296 @openai/codex@0.163.0 \
    && npm cache clean --force
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
COPY requirements.txt ./
COPY --from=hikari bin/requirements.txt bin/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r bin/requirements.txt && pip check && mkdir /data
COPY --from=hikari api/ api/
COPY --from=hikari experimental_worker/ experimental_worker/
COPY --from=hikari bin/*.py bin/
COPY --from=tsuchi tsuchi/ tsuchi/
COPY --from=kadan kadan/ kadan/
COPY --from=kadan AGENT.md AGENT.md
COPY combined/ combined/
RUN python bin/hikari.py check install
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; [u.urlopen(f'http://127.0.0.1:8000{p}', timeout=3) for p in ('/hikari/health', '/tsuchi/api/health', '/kadan/api/health')]"
# Worker results, such as generated images and audio, are large WebSocket messages.
CMD ["uvicorn", "combined.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "268435456", \
     "--reload", "--reload-dir", "/app/combined", "--reload-dir", "/app/api", "--reload-dir", "/app/tsuchi", "--reload-dir", "/app/kadan"]
