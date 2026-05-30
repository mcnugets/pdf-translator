"""
Phase 3: translate text.{source}.json → text.{target}.json
with per-paragraph context batching for better quality.

Usage:
    python -m src.translate pages/page_0 --from en --to ru
    python -m src.translate pages/page_0 --from en --to ru --force
    python -m src.translate pages/ --all-pages --from en --to ru
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator


def _group_spans_into_paragraphs(
    layout: dict,
    text: dict,
) -> list[list[tuple[str, str]]]:
    """Group spans into paragraphs based on y-proximity and x-range.

    Returns list of paragraphs, each paragraph is a list of
    (span_id, text) tuples in reading order.
    """
    span_map = text["spans"]
    bbox_map: dict[str, tuple[float, float, float, float]] = {}
    font_map: dict[str, str] = {}
    size_map: dict[str, float] = {}
    for s in layout["spans"]:
        sid = s["id"]
        if sid in span_map:
            bbox_map[sid] = tuple(s["bbox"])
            font_map[sid] = str(s.get("font", ""))
            size_map[sid] = float(s.get("size", 12.0))

    sorted_ids = sorted(
        [sid for sid in span_map if sid in bbox_map],
        key=lambda sid: (bbox_map[sid][1], bbox_map[sid][0]),
    )

    paragraphs: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    prev_bottom = 0.0
    prev_y0 = 0.0
    prev_x0 = 0.0
    prev_size = 12.0
    prev_font = ""

    for sid in sorted_ids:
        bbox = bbox_map[sid]
        size = size_map.get(sid, 12.0)
        font = font_map.get(sid, "")

        gap_threshold = max(prev_size, size) * 1.5

        x0s = [bbox_map[s][0] for s, _ in current]
        x_range = max(x0s) - min(x0s) if x0s else 0

        starts_new = False
        if current:
            y_gap = bbox[1] - prev_bottom
            if y_gap > gap_threshold:
                starts_new = True
            elif y_gap < 2 and (bbox[0] - prev_x0) > 50:
                starts_new = True
            elif (max(x0s + [bbox[0]]) - min(x0s + [bbox[0]])) - x_range > 100:
                starts_new = True
            elif prev_size > 0 and size > 0 and max(prev_size, size) / min(prev_size, size) >= 2.0:
                starts_new = True

        if starts_new:
            paragraphs.append(current)
            current = []

        current.append((sid, span_map[sid]))
        prev_bottom = bbox[3]
        prev_y0 = bbox[1]
        prev_x0 = bbox[0]
        prev_size = size
        prev_font = font

    if current:
        paragraphs.append(current)

    return paragraphs


def _split_by_char_proportion(
    text: str,
    proportions: list[float],
) -> list[str]:
    """Split *text* into parts proportional to *proportions*,
    respecting word boundaries (split after spaces when possible)."""
    if not text or not proportions:
        return [text] if text else [""]

    total_len = len(text)
    parts: list[str] = []
    start = 0

    for i, prop in enumerate(proportions):
        if i == len(proportions) - 1:
            parts.append(text[start:])
            break

        target = start + max(1, round(total_len * prop))
        target = min(target, len(text))

        if target >= len(text):
            target = len(text)

        if target > start and target < len(text):
            space = text.rfind(" ", start + 1, target + 1)
            if space > start:
                target = space + 1

        parts.append(text[start:target])
        start = target

    return parts


def _translate_paragraph(
    translator: GoogleTranslator,
    para: list[tuple[str, str]],
    source: str,
    target: str,
    layout: dict,
    *,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[str] | None:
    """Translate a paragraph by merging text, then split back.

    - For headings (short spans, large font): merge → translate →
      entire result goes to first span, rest empty.
    - For body text: merge → translate → character-proportional split
      with word-boundary awareness.
    """
    texts = [t for _, t in para]
    merged = " ".join(t.strip() for t in texts)

    if not merged.strip():
        return None

    merged = merged.strip()
    result = None
    for attempt in range(max_retries):
        try:
            result = translator.translate(merged)
            if result:
                result = result.strip()
                break
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"  Paragraph failed: {exc}", file=sys.stderr)
                return None
    else:
        return None

    if result is None:
        return None

    # Always use character-proportional split with word-boundary awareness.
    # This works for both word-wrapped headings (multi-line title) and
    # heading+subtitle pairs — much better than the old approach of putting
    # all text in the first span and emptying the rest.
    char_counts = [max(1, len(t.strip())) for t in texts]
    total_chars = sum(char_counts)
    proportions = [c / total_chars for c in char_counts]
    return _split_by_char_proportion(result, proportions)


def translate_span(
    translator: GoogleTranslator,
    text: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    text = text.strip()
    if not text:
        return text
    for attempt in range(max_retries):
        try:
            result = translator.translate(text)
            if result:
                return result.strip()
            return text
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"  Failed after {max_retries} retries: {exc}", file=sys.stderr)
                return text
    return text


def translate_page(page_dir: Path, source: str, target: str, force: bool) -> int:
    src_path = page_dir / f"text.{source}.json"
    if not src_path.is_file():
        print(f"  Skip {page_dir.name}: no text.{source}.json", file=sys.stderr)
        return 0

    tgt_path = page_dir / f"text.{target}.json"
    if tgt_path.is_file() and not force:
        print(f"  Skip {page_dir.name}: text.{target}.json exists (use --force)")
        return 0

    src_data = json.loads(src_path.read_text(encoding="utf-8"))
    src_spans: dict[str, str] = src_data["spans"]

    layout_path = page_dir / "layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    else:
        layout = None

    total = len(src_spans)
    print(f"  Translating {total} span(s) in {page_dir.name}: {source} → {target}")
    translator = GoogleTranslator(source=source, target=target)

    tgt_spans: dict[str, str] = {}

    if layout:
        paragraphs = _group_spans_into_paragraphs(layout, src_data)
        tgt_spans = {}
        order: list[str] = []
        for para in paragraphs:
            for sid, _ in para:
                order.append(sid)

        for para in paragraphs:
            if len(para) <= 1:
                sid, text = para[0]
                translated = translate_span(translator, text)
                tgt_spans[sid] = translated
                print(f"    [{order.index(sid) + 1}/{total}] {text!r} → {translated!r}")
            else:
                result = _translate_paragraph(translator, para, source, target, layout)
                if result and len(result) == len(para):
                    for (sid, text), translated in zip(para, result):
                        tgt_spans[sid] = translated
                        print(f"    [{order.index(sid) + 1}/{total}] {text!r} → {translated!r}")
                else:
                    for sid, text in para:
                        translated = translate_span(translator, text)
                        tgt_spans[sid] = translated
                        print(f"    [{order.index(sid) + 1}/{total}] {text!r} → {translated!r} (fallback)")
    else:
        for idx, (span_id, text) in enumerate(src_spans.items()):
            translated = translate_span(translator, text)
            tgt_spans[span_id] = translated
            print(f"    [{idx + 1}/{total}] {text!r} → {translated!r}")

    tgt_data = {
        "page": src_data["page"],
        "lang": target,
        "spans": tgt_spans,
    }
    tgt_path.write_text(json.dumps(tgt_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {tgt_path}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: translate text.{source}.json → text.{target}.json",
    )
    parser.add_argument("page_dir", type=Path, help="Page directory or parent directory with --all-pages")
    parser.add_argument("--from", dest="source", default="en", help="Source language code")
    parser.add_argument("--to", dest="target", required=True, help="Target language code")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process all page_N/ subdirectories under page_dir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate even if target file exists",
    )
    args = parser.parse_args(argv)

    if args.all_pages:
        dirs = sorted(
            [d for d in args.page_dir.iterdir() if d.is_dir() and d.name.startswith("page_")],
            key=lambda d: int(d.name.replace("page_", "")),
        )
        if not dirs:
            print(f"No page_N/ directories found under {args.page_dir}", file=sys.stderr)
            return 1
        total_spans = 0
        for d in dirs:
            total_spans += translate_page(d, args.source, args.target, args.force)
        print(f"Done: {len(dirs)} page(s), {total_spans} span(s) translated")
        return 0

    translate_page(args.page_dir, args.source, args.target, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
