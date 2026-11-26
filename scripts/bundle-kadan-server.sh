#!/usr/bin/env bash
# Stage all self-contained runtime assets for Kadan Server (single combined API + unified web UI).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT}/apps/kadan-server/src-tauri/resources/runtime"
WORKER_BIN="${1:-${KADAN_WORKER_BIN:-}}"

rm -rf "${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}/services" "${RUNTIME_DIR}/web" "${RUNTIME_DIR}/bin"

echo "[1/4] Building unified frontend (Kadan + Hikari + Tsuchi)..."
(cd "${ROOT}/services/frontend" && npm ci --no-audit --no-fund && npm run build)
cp -a "${ROOT}/services/frontend/dist/." "${RUNTIME_DIR}/web/"

echo "[2/4] Staging Python services..."
mkdir -p "${RUNTIME_DIR}/services/api" \
         "${RUNTIME_DIR}/services/tracing" \
         "${RUNTIME_DIR}/services/tsuchi" \
         "${RUNTIME_DIR}/services/hikari" \
         "${RUNTIME_DIR}/services/kadan"

rsync -a --exclude '__pycache__' --exclude 'tests' --exclude '.pytest_cache' \
  "${ROOT}/services/api/" "${RUNTIME_DIR}/services/api/"
rsync -a --exclude '__pycache__' --exclude 'tests' --exclude '.pytest_cache' \
  "${ROOT}/services/tracing/" "${RUNTIME_DIR}/services/tracing/"
rsync -a --exclude '__pycache__' --exclude 'tests' --exclude '.pytest_cache' \
  "${ROOT}/services/tsuchi/" "${RUNTIME_DIR}/services/tsuchi/"
rsync -a --exclude '__pycache__' --exclude 'tests' --exclude '.pytest_cache' \
  "${ROOT}/services/hikari/api/" "${RUNTIME_DIR}/services/hikari/api/"
rsync -a --exclude '__pycache__' --exclude 'tests' --exclude '.pytest_cache' \
  "${ROOT}/services/kadan/" "${RUNTIME_DIR}/services/kadan/"

echo "[3/4] Creating standalone Python runtime..."
if command -v uv >/dev/null 2>&1; then
  uv venv --relocatable --python 3.12 "${RUNTIME_DIR}/venv"
  uv pip install --python "${RUNTIME_DIR}/venv/bin/python" \
    "${RUNTIME_DIR}/services/tracing" \
    -r "${ROOT}/services/api/requirements.txt"
else
  python3 -m venv "${RUNTIME_DIR}/venv"
  "${RUNTIME_DIR}/venv/bin/pip" install --upgrade pip
  "${RUNTIME_DIR}/venv/bin/pip" install \
    "${RUNTIME_DIR}/services/tracing" \
    -r "${ROOT}/services/api/requirements.txt"
fi

echo "[4/4] Staging native C++ kadan-worker runtime..."
if [[ -n "${WORKER_BIN}" && -d "${WORKER_BIN}" ]]; then
  for sub in bin libexec lib share; do
    if [[ -d "${WORKER_BIN}/${sub}" ]]; then
      rsync -a "${WORKER_BIN}/${sub}/" "${RUNTIME_DIR}/${sub}/"
    fi
  done
  echo "  Staged kadan-worker install prefix from ${WORKER_BIN}"
elif [[ -n "${WORKER_BIN}" && -f "${WORKER_BIN}" ]]; then
  cp "${WORKER_BIN}" "${RUNTIME_DIR}/bin/kadan-worker"
  chmod +x "${RUNTIME_DIR}/bin/kadan-worker"
  echo "  Staged kadan-worker from ${WORKER_BIN}"
else
  echo "  Note: no KADAN_WORKER_BIN provided; set KADAN_WORKER_BIN or pass path/prefix as \$1 to bundle kadan-worker."
fi

echo "Done! Staged Kadan Server runtime in ${RUNTIME_DIR}"
