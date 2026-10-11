#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/libexec/kadan-model-worker" "$@"
