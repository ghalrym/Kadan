# Build from services/hikari. The API coordinates worker containers and runs no
# inference, so it carries no native worker and needs no GPU.
FROM python:3.12.15-slim-bookworm
RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
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
CMD ["python", "bin/hikari.py", "run", "serve"]
