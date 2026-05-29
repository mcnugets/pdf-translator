"""
Standalone PaddleOCR worker (run with Python 3.9–3.12).

Usage:
    python ocr_worker.py image.png --zoom 2 --lang en --min-confidence 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    ocr = PaddleOCR(use_angle_cls=True, lang=args.lang, show_log=False)
    raw = ocr.ocr(img, cls=True)
    lines = raw[0] if raw else []

    out = []
    for quad, (text, conf) in lines or []:
        text = str(text).strip()
        conf = float(conf)
        if not text or conf < args.min_confidence:
            continue
        xs = [float(p[0]) / args.zoom for p in quad]
        ys = [float(p[1]) / args.zoom for p in quad]
        out.append(
            {
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "text": text,
                "confidence": conf,
            }
        )

    json.dump(out, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
