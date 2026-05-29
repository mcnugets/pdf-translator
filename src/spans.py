"""Extract and merge text spans from PDF text layers and PaddleOCR."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf as fitz

Source = Literal["pdf", "ocr"]


@dataclass
class SpanRecord:
    id: str
    bbox: tuple[float, float, float, float]
    text: str
    font: str
    size: float
    color: int
    flags: int
    source: Source
    bg_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    ocr_confidence: float | None = None


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _normalize_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _estimate_font_size(bbox: tuple[float, float, float, float]) -> float:
    _, y0, _, y1 = bbox
    return max(6.0, round((y1 - y0) * 0.75, 1))


def extract_pdf_spans(page: fitz.Page) -> list[SpanRecord]:
    """Spans from the PDF text layer (selectable text)."""
    records: list[SpanRecord] = []
    index = 0
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                records.append(
                    SpanRecord(
                        id=f"span_{index}",
                        bbox=_normalize_bbox(tuple(span["bbox"])),
                        text=text,
                        font=str(span.get("font", "")),
                        size=float(span["size"]),
                        color=int(span["color"]),
                        flags=int(span["flags"]),
                        source="pdf",
                    )
                )
                index += 1
    return records


def extract_image_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 1:
            continue
        images.append(
            {
                "xref": block.get("xref", 0),
                "bbox": list(_normalize_bbox(tuple(block["bbox"]))),
            }
        )
    return images


def page_to_numpy(page: fitz.Page, zoom: float) -> tuple[Any, float]:
    """Render page to RGB numpy array; return (array, zoom factor)."""
    import numpy as np

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    channels = pix.n
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, channels)
    if channels == 4:
        img = img[:, :, :3]
    return img, zoom


_OCR_ENGINES: dict[str, Any] = {}


def _get_paddle_ocr(lang: str) -> Any:
    if lang not in _OCR_ENGINES:
        from paddleocr import PaddleOCR

        try:
            _OCR_ENGINES[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except TypeError:
            _OCR_ENGINES[lang] = PaddleOCR(use_textline_orientation=True, lang=lang)
    return _OCR_ENGINES[lang]


def _run_paddle_ocr(ocr: Any, img: Any) -> list[Any]:
    if hasattr(ocr, "ocr"):
        raw = ocr.ocr(img, cls=True)
        if not raw:
            return []
        first = raw[0]
        return first if isinstance(first, list) else raw
    if hasattr(ocr, "predict"):
        return ocr.predict(img) or []
    raise RuntimeError("Unsupported PaddleOCR API: no ocr() or predict() method")


def _iter_ocr_items(raw: Any) -> list[tuple[list[list[float]], str, float]]:
    """Normalize PaddleOCR 2.x / 3.x output to (quad, text, confidence)."""
    items: list[tuple[list[list[float]], str, float]] = []
    if not raw:
        return items

    # 2.x: [ [quad, (text, conf)], ... ]
    if isinstance(raw, list) and raw and isinstance(raw[0], list) and len(raw[0]) == 2:
        if isinstance(raw[0][1], (list, tuple)):
            for entry in raw:
                if not entry or len(entry) < 2:
                    continue
                quad, payload = entry[0], entry[1]
                if not payload or len(payload) < 2:
                    continue
                items.append((quad, str(payload[0]).strip(), float(payload[1])))
            return items

    # 3.x predict: list of dicts
    for block in raw if isinstance(raw, list) else []:
        if isinstance(block, dict):
            texts = block.get("rec_texts") or block.get("texts") or []
            scores = block.get("rec_scores") or block.get("scores") or []
            polys = block.get("dt_polys") or block.get("rec_polys") or block.get("boxes") or []
            for i, text in enumerate(texts):
                if not str(text).strip():
                    continue
                conf = float(scores[i]) if i < len(scores) else 1.0
                poly = polys[i] if i < len(polys) else None
                if poly is None:
                    continue
                items.append((poly, str(text).strip(), conf))
    return items


def _ocr_python() -> str | None:
    """Interpreter for PaddleOCR (main venv or OCR_PYTHON / venv-ocr)."""
    env = os.environ.get("OCR_PYTHON")
    if env and Path(env).is_file():
        return env
    venv_ocr = Path(__file__).resolve().parents[1] / "venv-ocr" / "bin" / "python"
    if venv_ocr.is_file():
        return str(venv_ocr)
    return None


def _extract_ocr_spans_subprocess(
    page: fitz.Page,
    *,
    zoom: float,
    lang: str,
    min_confidence: float,
) -> list[SpanRecord]:
    python = _ocr_python()
    if not python:
        raise ImportError(
            "PaddleOCR needs Python 3.9–3.12. Create venv-ocr:\n"
            "  brew install python@3.12\n"
            "  python3.12 -m venv venv-ocr\n"
            "  venv-ocr/bin/pip install paddlepaddle paddleocr pillow numpy\n"
            "Or set OCR_PYTHON=/path/to/python3.12"
        )

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    worker = Path(__file__).resolve().parent / "ocr_worker.py"

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "page.png"
        pix.save(str(img_path))
        proc = subprocess.run(
            [
                python,
                str(worker),
                str(img_path),
                "--zoom",
                str(zoom),
                "--lang",
                lang,
                "--min-confidence",
                str(min_confidence),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"OCR worker failed:\n{proc.stderr or proc.stdout}")

        hits = json.loads(proc.stdout or "[]")

    records: list[SpanRecord] = []
    for index, hit in enumerate(hits):
        bbox = _normalize_bbox(tuple(hit["bbox"]))
        records.append(
            SpanRecord(
                id=f"ocr_{index}",
                bbox=bbox,
                text=str(hit["text"]),
                font="OCR",
                size=_estimate_font_size(bbox),
                color=0,
                flags=0,
                source="ocr",
                ocr_confidence=float(hit["confidence"]),
            )
        )
    return records


def extract_ocr_spans(
    page: fitz.Page,
    *,
    zoom: float = 2.0,
    lang: str = "en",
    min_confidence: float = 0.5,
) -> list[SpanRecord]:
    """Spans from PaddleOCR on a rendered page image."""
    try:
        img, scale = page_to_numpy(page, zoom)
        ocr = _get_paddle_ocr(lang)
        raw = _run_paddle_ocr(ocr, img)
        parsed = _iter_ocr_items(raw if isinstance(raw, list) else [])

        records: list[SpanRecord] = []
        for index, (quad, text, confidence) in enumerate(parsed):
            if not text or confidence < min_confidence:
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            bbox = _normalize_bbox(
                (min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale)
            )
            records.append(
                SpanRecord(
                    id=f"ocr_{index}",
                    bbox=bbox,
                    text=text,
                    font="OCR",
                    size=_estimate_font_size(bbox),
                    color=0,
                    flags=0,
                    source="ocr",
                    ocr_confidence=confidence,
                )
            )
        return records
    except (ImportError, ModuleNotFoundError):
        return _extract_ocr_spans_subprocess(
            page, zoom=zoom, lang=lang, min_confidence=min_confidence
        )


def merge_spans(
    pdf_spans: list[SpanRecord],
    ocr_spans: list[SpanRecord],
    *,
    iou_threshold: float = 0.35,
) -> list[SpanRecord]:
    """Keep all PDF spans; add OCR spans that do not overlap existing text."""
    merged = list(pdf_spans)
    next_index = len(pdf_spans)

    for ocr in ocr_spans:
        duplicate = False
        for pdf in pdf_spans:
            if bbox_iou(ocr.bbox, pdf.bbox) >= iou_threshold:
                duplicate = True
                break
            # Same text already covered by a nearby PDF span
            if ocr.text.lower() in pdf.text.lower() and bbox_iou(ocr.bbox, pdf.bbox) > 0.1:
                duplicate = True
                break
        if duplicate:
            continue
        merged.append(
            SpanRecord(
                id=f"span_{next_index}",
                bbox=ocr.bbox,
                text=ocr.text,
                font=ocr.font,
                size=ocr.size,
                color=ocr.color,
                flags=ocr.flags,
                source="ocr",
                bg_color=ocr.bg_color,
                ocr_confidence=ocr.ocr_confidence,
            )
        )
        next_index += 1

    return merged


def extract_page_spans(
    page: fitz.Page,
    *,
    use_ocr: bool = True,
    ocr_zoom: float = 2.0,
    ocr_lang: str = "en",
    min_ocr_confidence: float = 0.5,
) -> tuple[list[SpanRecord], list[dict[str, Any]]]:
    pdf_spans = extract_pdf_spans(page)
    images = extract_image_blocks(page)
    if not use_ocr:
        return pdf_spans, images

    ocr_spans = extract_ocr_spans(
        page,
        zoom=ocr_zoom,
        lang=ocr_lang,
        min_confidence=min_ocr_confidence,
    )
    return merge_spans(pdf_spans, ocr_spans), images


def build_layout_json(
    page_num: int,
    page: fitz.Page,
    spans: list[SpanRecord],
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    layout_spans = []
    for span in spans:
        entry: dict[str, Any] = {
            "id": span.id,
            "bbox": list(span.bbox),
            "font": span.font,
            "size": span.size,
            "color": span.color,
            "flags": span.flags,
            "bg_color": span.bg_color,
            "source": span.source,
        }
        if span.ocr_confidence is not None:
            entry["ocr_confidence"] = span.ocr_confidence
        layout_spans.append(entry)

    return {
        "page": page_num,
        "width": float(page.rect.width),
        "height": float(page.rect.height),
        "spans": layout_spans,
        "images": images,
    }


def build_text_json(page_num: int, spans: list[SpanRecord], lang: str = "en") -> dict[str, Any]:
    return {
        "page": page_num,
        "lang": lang,
        "spans": {span.id: span.text for span in spans},
    }


def write_page_bundle(
    out_dir: Path,
    page_num: int,
    page: fitz.Page,
    spans: list[SpanRecord],
    images: list[dict[str, Any]],
    lang: str = "en",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_path = out_dir / "layout.json"
    text_path = out_dir / f"text.{lang}.json"

    layout = build_layout_json(page_num, page, spans, images)
    text = build_text_json(page_num, spans, lang=lang)

    layout_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(json.dumps(text, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return layout_path, text_path


def span_record_from_layout(layout_span: dict[str, Any], text: str) -> SpanRecord:
    return SpanRecord(
        id=str(layout_span["id"]),
        bbox=tuple(layout_span["bbox"]),
        text=text,
        font=str(layout_span.get("font", "")),
        size=float(layout_span.get("size", 12.0)),
        color=int(layout_span.get("color", 0)),
        flags=int(layout_span.get("flags", 0)),
        source=layout_span.get("source", "pdf"),
        bg_color=list(layout_span.get("bg_color", [255, 255, 255])),
        ocr_confidence=layout_span.get("ocr_confidence"),
    )


def load_page_bundle(page_dir: Path, lang: str = "en") -> tuple[dict[str, Any], dict[str, Any]]:
    layout = json.loads((page_dir / "layout.json").read_text(encoding="utf-8"))
    text = json.loads((page_dir / f"text.{lang}.json").read_text(encoding="utf-8"))
    return layout, text
