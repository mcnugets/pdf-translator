"""
Phase 0: dump text spans from one PDF page (bbox, text, size, color).

Usage:
    python -m src.inspect_spans path/to/brochure.pdf
    python -m src.inspect_spans path/to/brochure.pdf --page 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import pymupdf as fitz

from src.spans import OcrBackend, SpanRecord, extract_page_spans


def _color_hex(color: int) -> str:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return f"#{r:02x}{g:02x}{b:02x}"


def format_span_line(span: SpanRecord, index: int) -> str:
    x0, y0, x1, y1 = span.bbox
    bbox = f"{x0:6.1f},{y0:6.1f},{x1:6.1f},{y1:6.1f}"
    preview = span.text.replace("\n", "\\n")
    if len(preview) > 48:
        preview = preview[:45] + "..."
    src = f" [{span.source}]" if span.source == "ocr" else ""
    conf = f" conf={span.ocr_confidence:.2f}" if span.ocr_confidence is not None else ""
    return (
        f"[{index:3d}] bbox=({bbox})  "
        f"size={span.size:5.1f}  color={_color_hex(span.color)}  "
        f"flags={span.flags:3d}  font={span.font!r}{src}{conf}\n"
        f"       text={preview!r}"
    )


def inspect_page(
    doc: fitz.Document,
    page_num: int,
    *,
    use_ocr: bool = False,
    ocr_backend: OcrBackend = "auto",
    ocr_zoom: float = 2.0,
    ocr_lang: str = "en",
) -> list[SpanRecord]:
    if page_num < 0 or page_num >= doc.page_count:
        raise ValueError(f"Page {page_num} out of range (document has {doc.page_count} page(s))")
    page = doc[page_num]
    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    text_blocks = sum(1 for b in page_dict["blocks"] if b["type"] == 0)
    image_blocks = sum(1 for b in page_dict["blocks"] if b["type"] == 1)

    spans, _ = extract_page_spans(
        page,
        use_ocr=use_ocr,
        ocr_backend=ocr_backend,
        ocr_zoom=ocr_zoom,
        ocr_lang=ocr_lang,
    )

    print(f"File: {doc.name}")
    print(f"Page: {page_num} / {doc.page_count - 1}  size={page.rect.width:.1f} x {page.rect.height:.1f} pt")
    print(f"Blocks: {text_blocks} text, {image_blocks} image")
    pdf_n = sum(1 for s in spans if s.source == "pdf")
    ocr_n = sum(1 for s in spans if s.source == "ocr")
    print(f"Spans: {len(spans)} ({pdf_n} PDF, {ocr_n} OCR)\n")

    if not spans:
        print("No text spans found on this page.")
        return spans

    print("-" * 72)
    for i, span in enumerate(spans):
        print(format_span_line(span, i))
        print()
    print("-" * 72)
    print(f"Total: {len(spans)} span(s)")
    return spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0: print text spans (bbox, text, size, color) for one PDF page.",
    )
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("-p", "--page", type=int, default=0, help="Page index, 0-based")
    parser.add_argument("--json", action="store_true", help="Also print spans as JSON")
    parser.add_argument("--ocr", action="store_true", help="Include OCR spans")
    parser.add_argument(
        "--ocr-backend",
        default="auto",
        choices=["auto", "tesseract", "paddle"],
        help="OCR backend (default: auto)",
    )
    parser.add_argument("--ocr-zoom", type=float, default=2.0)
    parser.add_argument("--ocr-lang", default="en")
    args = parser.parse_args(argv)

    try:
        doc = fitz.open(args.pdf)
    except Exception as exc:
        print(f"Error opening PDF: {exc}", file=sys.stderr)
        return 1

    try:
        spans = inspect_page(
            doc,
            args.page,
            use_ocr=args.ocr,
            ocr_backend=args.ocr_backend,
            ocr_zoom=args.ocr_zoom,
            ocr_lang=args.ocr_lang,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        doc.close()

    if args.json:
        payload = []
        for span in spans:
            row = {**asdict(span)}
            row["bbox"] = list(span.bbox)
            payload.append(row)
        print("\nJSON:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
