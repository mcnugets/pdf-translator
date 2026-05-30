"""Extract and merge text spans from PDF text layers and OCR (Tesseract / PaddleOCR)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pymupdf as fitz
from PIL import Image

import pytesseract

_TESS_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
if os.path.isfile(_TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = _TESS_CMD
os.environ.setdefault(
    "TESSDATA_PREFIX",
    r"C:\Program Files\Tesseract-OCR\tessdata",
)

Source = Literal["pdf", "ocr"]
OcrBackend = Literal["auto", "tesseract", "paddle"]


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
    seen_xrefs: set[int] = set()
    for xref, *_ in page.get_images():
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        rects = page.get_image_rects(xref)
        for rect in rects:
            images.append(
                {
                    "xref": xref,
                    "bbox": list(_normalize_bbox((rect.x0, rect.y0, rect.x1, rect.y1))),
                }
            )
    return images


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

_TESS_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "ru": "rus",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
}


def _extract_tesseract_spans(
    page: fitz.Page,
    *,
    zoom: float,
    lang: str,
    min_confidence: float,
) -> list[SpanRecord]:
    import pytesseract as _pt

    img_pil, scale = _page_to_pil(page, zoom)
    tess_lang = _TESS_LANG_MAP.get(lang, lang)

    data = _pt.image_to_data(
        img_pil, lang=tess_lang, output_type=_pt.Output.DICT
    )

    n = len(data["text"])
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = int(data["conf"][i])
        if not text or conf < min_confidence * 100:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    records: list[SpanRecord] = []
    for idx, (key, indices) in enumerate(lines.items()):
        texts: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        confs: list[int] = []
        for i in indices:
            texts.append(data["text"][i])
            confs.append(int(data["conf"][i]))
            xs.append(float(data["left"][i]))
            xs.append(float(data["left"][i] + data["width"][i]))
            ys.append(float(data["top"][i]))
            ys.append(float(data["top"][i] + data["height"][i]))

        bbox_px = (min(xs), min(ys), max(xs), max(ys))
        bbox = _normalize_bbox(
            (bbox_px[0] / scale, bbox_px[1] / scale, bbox_px[2] / scale, bbox_px[3] / scale)
        )
        avg_conf = sum(confs) / len(confs) / 100.0

        records.append(
            SpanRecord(
                id=f"ocr_{idx}",
                bbox=bbox,
                text=" ".join(texts),
                font="OCR",
                size=_estimate_font_size(bbox),
                color=0,
                flags=0,
                source="ocr",
                ocr_confidence=avg_conf,
            )
        )

    return records


# ---------------------------------------------------------------------------
# PaddleOCR backend
# ---------------------------------------------------------------------------

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
    items: list[tuple[list[list[float]], str, float]] = []
    if not raw:
        return items

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
    env = os.environ.get("OCR_PYTHON")
    if env and Path(env).is_file():
        return env
    venv_ocr = Path(__file__).resolve().parents[1] / "venv-ocr" / "bin" / "python"
    if venv_ocr.is_file():
        return str(venv_ocr)
    return None


def _extract_paddle_spans(
    page: fitz.Page,
    *,
    zoom: float,
    lang: str,
    min_confidence: float,
) -> list[SpanRecord]:
    try:
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        channels = pix.n
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, channels)
        if channels == 4:
            img = img[:, :, :3]

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
                (min(xs) / zoom, min(ys) / zoom, max(xs) / zoom, max(ys) / zoom)
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
        return _extract_ocr_spans_subprocess(page, zoom=zoom, lang=lang, min_confidence=min_confidence)


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
            [python, str(worker), str(img_path), "--zoom", str(zoom), "--lang", lang, "--min-confidence", str(min_confidence)],
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


# ---------------------------------------------------------------------------
# Backend auto-detection
# ---------------------------------------------------------------------------


def detect_ocr_backend() -> OcrBackend:
    """Auto-detect best available OCR backend based on hardware and deps."""
    # 1. Check PaddleOCR — GPU-capable if paddlepaddle-gpu is installed
    paddle_ok = False
    try:
        import paddleocr  # noqa: F401
        paddle_ok = True
    except ImportError:
        paddle_ok = bool(_ocr_python())

    # 2. Check Tesseract binary
    tesseract_ok = os.path.isfile(_TESS_CMD) or (
        os.environ.get("TESSERACT_CMD") is not None
    )
    if not tesseract_ok:
        try:
            import shutil
            tesseract_ok = shutil.which("tesseract") is not None
        except Exception:
            tesseract_ok = False

    # 3. Decide: prefer PaddleOCR when GPU is available, else Tesseract
    if paddle_ok:
        import importlib
        has_gpu = importlib.util.find_spec("paddle") is not None
        if has_gpu:
            try:
                import paddle
                has_gpu = paddle.device.is_compiled_with_cuda()
            except Exception:
                has_gpu = False
        if has_gpu:
            return "paddle"
        if tesseract_ok:
            return "tesseract"
        return "paddle"

    if tesseract_ok:
        return "tesseract"

    msg = "No OCR backend available. Install Tesseract or PaddleOCR."
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _page_to_pil(page: fitz.Page, zoom: float) -> tuple[Image.Image, float]:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGBA" if pix.n == 4 else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img, zoom


def extract_ocr_spans(
    page: fitz.Page,
    *,
    zoom: float = 2.0,
    lang: str = "en",
    min_confidence: float = 0.5,
    backend: OcrBackend = "auto",
) -> list[SpanRecord]:
    if backend == "auto":
        backend = detect_ocr_backend()

    if backend == "tesseract":
        return _extract_tesseract_spans(
            page, zoom=zoom, lang=lang, min_confidence=min_confidence
        )
    elif backend == "paddle":
        return _extract_paddle_spans(
            page, zoom=zoom, lang=lang, min_confidence=min_confidence
        )
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")


def _bbox_contains_center(outer: tuple[float, ...], inner: tuple[float, ...]) -> bool:
    """Return True if the center of *inner* bbox lies inside *outer* bbox."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _is_low_quality_ocr(ocr: SpanRecord) -> bool:
    """Check if an OCR span is too low-quality to include.

    Drops obvious text fragments (very short, non-numeric) and very
    low-confidence results that would produce garbage translations.
    """
    text = ocr.text.strip()
    if not text:
        return True
    if ocr.ocr_confidence is not None and ocr.ocr_confidence < 0.70:
        return True
    if len(text) < 4 and not text.isdigit():
        return True
    return False


def merge_spans(
    pdf_spans: list[SpanRecord],
    ocr_spans: list[SpanRecord],
    *,
    iou_threshold: float = 0.35,
) -> list[SpanRecord]:
    """Keep all PDF spans; add OCR spans that do not overlap existing text.

    Low-quality OCR spans (very short, low confidence) are dropped outright.
    """
    merged = list(pdf_spans)
    next_index = len(pdf_spans)

    for ocr in ocr_spans:
        if _is_low_quality_ocr(ocr):
            continue
        duplicate = False
        for pdf in pdf_spans:
            if bbox_iou(ocr.bbox, pdf.bbox) >= iou_threshold:
                duplicate = True
                break
            if ocr.text.lower() in pdf.text.lower() and bbox_iou(ocr.bbox, pdf.bbox) > 0.1:
                duplicate = True
                break
            if pdf.text.lower() in ocr.text.lower() and bbox_iou(ocr.bbox, pdf.bbox) > 0.1:
                duplicate = True
                break
            # If OCR span overlaps a PDF span and shares >80% of its
            # characters with the PDF text, it's a fragment → drop.
            # Use both IoU and centroid-inside-bbox checks to catch
            # OCR fragments that barely overlap their parent span.
            if len(ocr.text) > 3:
                shared = sum(1 for c in ocr.text.lower() if c in pdf.text.lower())
                char_ratio = shared / len(ocr.text)
                if char_ratio > 0.8 and (
                    bbox_iou(ocr.bbox, pdf.bbox) > 0.05
                    or _bbox_contains_center(pdf.bbox, ocr.bbox)
                ):
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


def _sample_bg_colors(page: fitz.Page, spans: list[SpanRecord]) -> list[SpanRecord]:
    pix = page.get_pixmap(dpi=72, alpha=False)
    w, h = pix.width, pix.height
    page_w, page_h = page.rect.width, page.rect.height
    scale_x = w / page_w
    scale_y = h / page_h

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, 3)

    for span in spans:
        x0, y0, x1, y1 = span.bbox
        px0 = max(0, int(x0 * scale_x))
        py0 = max(0, int(y0 * scale_y))
        px1 = min(w, int(x1 * scale_x))
        py1 = min(h, int(y1 * scale_y))

        if px1 <= px0 or py1 <= py0:
            continue

        region = arr[py0:py1, px0:px1]
        margin = max(1, min(2, (px1 - px0) // 8, (py1 - py0) // 8))

        samples: list[np.ndarray] = []
        if py1 - py0 > margin * 2:
            samples.append(region[:margin, :, :])
            samples.append(region[-margin:, :, :])
        if px1 - px0 > margin * 2:
            samples.append(region[:, :margin, :])
            samples.append(region[:, -margin:, :])

        if not samples:
            continue

        all_pixels = np.concatenate([s.reshape(-1, 3) for s in samples])
        bg = all_pixels.mean(axis=0).round().astype(int).tolist()
        span.bg_color = bg

    return spans


def extract_page_spans(
    page: fitz.Page,
    *,
    use_ocr: bool = True,
    ocr_backend: OcrBackend = "auto",
    ocr_zoom: float = 2.0,
    ocr_lang: str = "en",
    min_ocr_confidence: float = 0.5,
) -> tuple[list[SpanRecord], list[dict[str, Any]]]:
    pdf_spans = extract_pdf_spans(page)
    images = extract_image_blocks(page)
    if not use_ocr:
        return _sample_bg_colors(page, pdf_spans), images

    ocr_spans = extract_ocr_spans(
        page,
        zoom=ocr_zoom,
        lang=ocr_lang,
        min_confidence=min_ocr_confidence,
        backend=ocr_backend,
    )
    merged = merge_spans(pdf_spans, ocr_spans)
    return _sample_bg_colors(page, merged), images


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


def extract_images_to_disk(
    images_dir: Path,
    doc: fitz.Document,
    images: list[dict[str, Any]],
) -> None:
    """Save page images as xref_N.png to *images_dir*."""
    for img in images:
        xref = img.get("xref", 0)
        if xref <= 0:
            continue
        img_path = images_dir / f"xref_{xref}.png"
        if img_path.is_file():
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(str(img_path))
        except Exception:
            pass


def _render_text_free(page: fitz.Page, dpi: float = 300.0) -> Image.Image:
    """Render page as an image with all text removed.

    Strips BT...ET blocks (PDF text showing operators) from the content
    stream before rendering, producing a clean background image that
    preserves vectors, images, gradients, and colors.
    """
    doc = page.parent
    src_doc = doc  # the source document
    tmp_doc = fitz.Document()
    try:
        tmp_doc.insert_pdf(src_doc, from_page=page.number, to_page=page.number)
        tp = tmp_doc[0]
        tp.clean_contents()
        xrefs = tp.get_contents()
        xref = xrefs[0] if isinstance(xrefs, list) else xrefs
        content = tmp_doc.xref_stream(xref)
        content = re.sub(rb'BT.*?ET', b'', content, flags=re.DOTALL)
        tmp_doc.update_stream(xref, content)
        zoom = dpi / 72.0
        pix = tp.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.pil_image()
    finally:
        tmp_doc.close()


def render_fullpage_background(page: fitz.Page, images_dir: Path, page_num: int) -> None:
    """Render the full page to a JPEG at 150 DPI with text removed.

    The output is a clean background image (no text) that preserves all
    visual content (raster images, vector graphics, gradients, colors).
    Used by the stateless rebuild path as a base layer.
    """
    try:
        img = _render_text_free(page, dpi=300.0)
        path = images_dir / f"fullpage_{page_num}.jpg"
        images_dir.mkdir(parents=True, exist_ok=True)
        img.save(str(path), format="JPEG", quality=85, optimize=True)
    except Exception:
        pass


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

    doc = page.parent
    if doc is not None:
        images_dir = out_dir.parent / "images"
        extract_images_to_disk(images_dir, doc, images)
        render_fullpage_background(page, images_dir, page_num)

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
