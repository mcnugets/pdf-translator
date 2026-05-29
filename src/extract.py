"""
Phase 1: extract layout.json + text.en.json for PDF page(s).

Combines PyMuPDF text-layer spans with PaddleOCR for text embedded in images.

Usage:
    python -m src.extract "input/brochure.pdf" --page 4 --out pages/page_4
    python -m src.extract "input/brochure.pdf" --page 4 --out pages/page_4 --no-ocr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf as fitz

from src.spans import extract_page_spans, write_page_bundle


def extract_one_page(
    doc: fitz.Document,
    page_num: int,
    out_dir: Path,
    *,
    lang: str,
    use_ocr: bool,
    ocr_zoom: float,
    ocr_lang: str,
    min_ocr_confidence: float,
) -> tuple[int, int, int]:
    page = doc[page_num]
    spans, images = extract_page_spans(
        page,
        use_ocr=use_ocr,
        ocr_zoom=ocr_zoom,
        ocr_lang=ocr_lang,
        min_ocr_confidence=min_ocr_confidence,
    )
    write_page_bundle(out_dir, page_num, page, spans, images, lang=lang)

    pdf_count = sum(1 for s in spans if s.source == "pdf")
    ocr_count = sum(1 for s in spans if s.source == "ocr")
    return len(spans), pdf_count, ocr_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: write layout.json and text.{lang}.json for PDF page(s).",
    )
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("-p", "--page", type=int, help="Single page index (0-based)")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Extract every page (writes pages/page_N/ under --out)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Output directory (page dir or parent when using --all-pages)",
    )
    parser.add_argument("--lang", default="en", help="Language code for text file (default: en)")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip PaddleOCR (PDF text layer only)",
    )
    parser.add_argument(
        "--ocr-zoom",
        type=float,
        default=2.0,
        help="Render scale for OCR (default: 2.0)",
    )
    parser.add_argument(
        "--ocr-lang",
        default="en",
        help="PaddleOCR language code (default: en)",
    )
    parser.add_argument(
        "--min-ocr-confidence",
        type=float,
        default=0.5,
        help="Drop OCR hits below this confidence (default: 0.5)",
    )
    args = parser.parse_args(argv)

    if args.all_pages and args.page is not None:
        print("Use either --page or --all-pages, not both.", file=sys.stderr)
        return 1
    if not args.all_pages and args.page is None:
        print("Specify --page N or --all-pages.", file=sys.stderr)
        return 1

    try:
        doc = fitz.open(args.pdf)
    except Exception as exc:
        print(f"Error opening PDF: {exc}", file=sys.stderr)
        return 1

    pages = range(doc.page_count) if args.all_pages else [args.page]

    try:
        for page_num in pages:
            if page_num < 0 or page_num >= doc.page_count:
                print(f"Page {page_num} out of range.", file=sys.stderr)
                return 1

            out_dir = args.out if not args.all_pages else args.out / f"page_{page_num}"
            total, pdf_n, ocr_n = extract_one_page(
                doc,
                page_num,
                out_dir,
                lang=args.lang,
                use_ocr=not args.no_ocr,
                ocr_zoom=args.ocr_zoom,
                ocr_lang=args.ocr_lang,
                min_ocr_confidence=args.min_ocr_confidence,
            )
            ocr_msg = f", {ocr_n} from OCR" if not args.no_ocr else ""
            print(
                f"Page {page_num}: {total} spans ({pdf_n} PDF{ocr_msg}) → {out_dir}/"
            )
    finally:
        doc.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
