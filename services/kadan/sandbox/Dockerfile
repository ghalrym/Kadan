FROM node:22.23.4-bookworm-slim AS node22
# Install only the official pinned CLI and its platform package in the image build.
RUN npm install --global --ignore-scripts @openai/codex@0.162.1
FROM node:24.22.1-bookworm-slim AS node24
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM ghcr.io/agent-infra/sandbox:1.11.0@sha256:6328d7fd2f0ff0b4c147c3d05b3df1ce331f4a482eb6e550ecd64ed1fcf906e7

COPY --from=node22 /usr/local/bin/node /opt/nodejs/22/bin/node
COPY --from=node22 /usr/local/lib/node_modules/@openai/codex /opt/kadan/codex
RUN ln -sfn /opt/kadan/codex/bin/codex.js /opt/nodejs/22/bin/codex
COPY coding_jobs.py /opt/kadan/coding_jobs.py
COPY --from=node22 /usr/local/bin/node /usr/local/code-server-4.104.0/lib/node
COPY --from=node24 /usr/local/bin/node /opt/nodejs/24/bin/node
COPY --from=uv /uv /usr/local/bin/uv
# Keep the vendor packages while replacing each interpreter within its existing series.
RUN UV_PYTHON_INSTALL_DIR=/tmp/kadan-python UV_PYTHON_BIN_DIR=/tmp/kadan-python/bin \
      uv python install 3.11.17 3.12.15 \
    && cp -a /tmp/kadan-python/cpython-3.11.17-*/. /opt/python3.11/ \
    && cp -a /tmp/kadan-python/cpython-3.12.15-*/. /opt/python3.12/ \
    && rm -rf /tmp/kadan-python
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade \
      python3.10 python3.10-minimal libpython3.10-stdlib libpython3.10-minimal \
    && rm -rf /var/lib/apt/lists/*
RUN for python in /usr/bin/python3 /opt/python3.11/bin/python /opt/python3.12/bin/python; do \
      PIP_BREAK_SYSTEM_PACKAGES=1 "$python" -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
      && PIP_BREAK_SYSTEM_PACKAGES=1 "$python" -m pip install --no-cache-dir --upgrade certifi==2026.10.19 || exit; \
    done

# Keep existing Node 20 entry points usable on supported Node 22.
RUN rm /opt/nodejs/20 \
    && ln -s /opt/nodejs/22 /opt/nodejs/20 \
    && ln -sfn /opt/nodejs/22/bin/node /usr/local/bin/node20 \
    && rm -rf /opt/fnm/node-versions/v20.20.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends tint2 wmctrl gsettings-desktop-schemas \
    && rm -rf /var/lib/apt/lists/*
COPY desktop /opt/kadan/desktop
RUN python3 /opt/kadan/desktop/configure_desktop.py
# Closing the last window must not make the upstream supervisor reopen it.
ENV BROWSER_EXTRA_ARGS="${BROWSER_EXTRA_ARGS} --keep-alive-for-test"
