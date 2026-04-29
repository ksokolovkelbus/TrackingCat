#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec .venv/bin/python tools/export_openvino_models.py "$@"
