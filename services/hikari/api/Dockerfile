# Build from services/hikari. The API coordinates worker containers and runs no
# inference, so it carries no native worker and needs no GPU.
FROM python:3.12-slim-bookworm AS development
WORKDIR /app
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY api/requirements*.txt /app/
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r bin/requirements.txt
COPY api/ /app/api
COPY bin/*.py /app/bin/
EXPOSE 8000
CMD ["python", "bin/hikari.py", "dev", "serve"]

# The release image is last so it stays the default build target.
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=tracing . /opt/kadan-tracing
RUN pip install --no-cache-dir /opt/kadan-tracing
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY api/requirements*.txt /app/
COPY bin/requirements.txt /app/bin/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r bin/requirements.txt && pip check
COPY api/ /app/api
COPY bin/*.py /app/bin/
RUN python bin/hikari.py check install
EXPOSE 8000
CMD ["python", "-m", "api"]
