#!/usr/bin/env bash
# Stage the runtime Kadan bundles to run its services on this computer (combined API + web UI + workers).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT}/apps/kadan/src-tauri/resources/runtime"
WORKER_BIN="${1:-${KADAN_WORKER_BIN:-}}"
REQUIRED_WORKLOADS=("llm" "speech" "tts" "image")

rm -rf "${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}/services" "${RUNTIME_DIR}/web" "${RUNTIME_DIR}/bin"

echo "[1/4] Building the web app..."
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
if [[ -z "${WORKER_BIN}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    BUILD_DIR="${KADAN_WORKER_BUILD_DIR:-${ROOT}/build/kadan-worker-release}"
    INSTALL_DIR="${BUILD_DIR}-install"
    CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-80;86;89;90}"
    echo "  Building native kadan-worker with all CUDA workloads (${CUDA_ARCHS})..."
    cmake -S "${ROOT}/services/hikari/inference_worker" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DKADAN_ENABLE_CUDA=ON \
      -DKADAN_ENABLE_TTS_WORKER=ON \
      -DKADAN_ENABLE_IMAGE_GENERATION=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
    cmake --build "${BUILD_DIR}" --target kadan-worker --parallel "$(nproc)"
    rm -rf "${INSTALL_DIR}"
    cmake --install "${BUILD_DIR}" --component kadan-worker --prefix "${INSTALL_DIR}"
    WORKER_BIN="${INSTALL_DIR}"
  else
    echo "ERROR: No KADAN_WORKER_BIN provided and nvcc was not found on PATH." >&2
    echo "Install the CUDA toolkit or pass a prebuilt kadan-worker install prefix as \$1." >&2
    exit 1
  fi
fi

if [[ -d "${WORKER_BIN}" ]]; then
  for sub in bin libexec lib share; do
    if [[ -d "${WORKER_BIN}/${sub}" ]]; then
      rsync -a "${WORKER_BIN}/${sub}/" "${RUNTIME_DIR}/${sub}/"
    fi
  done
  echo "  Staged kadan-worker install prefix from ${WORKER_BIN}"
elif [[ -f "${WORKER_BIN}" ]]; then
  cp "${WORKER_BIN}" "${RUNTIME_DIR}/bin/kadan-worker"
  chmod +x "${RUNTIME_DIR}/bin/kadan-worker"
  echo "  Staged kadan-worker from ${WORKER_BIN}"
else
  echo "ERROR: KADAN_WORKER_BIN path does not exist: ${WORKER_BIN}" >&2
  exit 1
fi

echo "  Verifying all required workloads are compiled into kadan-worker..."
WORKLOAD_JSON="$("${RUNTIME_DIR}/bin/kadan-worker" --list-workloads)"
echo "  ${WORKLOAD_JSON}"
"${RUNTIME_DIR}/venv/bin/python" - "${WORKLOAD_JSON}" "${REQUIRED_WORKLOADS[@]}" <<'PY'
import json, sys
caps = json.loads(sys.argv[1])
required = set(sys.argv[2:])
found = {w["id"] for w in caps.get("workloads", [])}
missing = sorted(required - found)
if missing:
    sys.stderr.write(
        f"ERROR: kadan-worker is missing required workloads: {', '.join(missing)} "
        f"(found: {sorted(found)})\n"
    )
    sys.exit(1)
PY

echo "Done! Staged complete Kadan runtime (all workloads verified) in ${RUNTIME_DIR}"
