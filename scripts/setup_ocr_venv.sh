#!/usr/bin/env bash
# PaddleOCR requires Python 3.9–3.12 (not 3.13+). Creates venv-ocr/ in project root.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON312:-}"
if [[ -z "$PY" ]]; then
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
      PY="$candidate"
      break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "No Python 3.12 found. Install one, e.g.: brew install python@3.12"
  echo "Then: PYTHON312=\$(brew --prefix python@3.12)/bin/python3.12 $0"
  exit 1
fi

echo "Using: $($PY --version)"
rm -rf venv-ocr
"$PY" -m venv venv-ocr
venv-ocr/bin/python -m pip install --upgrade pip
venv-ocr/bin/pip install paddlepaddle paddleocr pillow numpy

echo ""
echo "Done. OCR will use: $(pwd)/venv-ocr/bin/python"
echo "Or: export OCR_PYTHON=$(pwd)/venv-ocr/bin/python"
