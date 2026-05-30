"""
Phase 4/7: PDF reconstructor with multi-line textbox fitting and pixmap patch backgrounds.

Two modes:
  Source-based (default) — overlay on source PDF canvas with pixmap background patches.
  Stateless (--stateless) — create PDF from scratch using only layout + text + images.

Usage:
    python -m src.rebuild source.pdf pages/page_0 -o out.pdf
    python -m src.rebuild source.pdf pages/page_0 -o out.pdf --lang ru
    python -m src.rebuild source.pdf pages/ -o out.pdf --lang ru --all-pages
    python -m src.rebuild -o out.pdf pages/page_0 --lang ru --stateless
    python -m src.rebuild -o out.pdf pages/ --lang ru --all-pages --stateless
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pymupdf as fitz

from src.spans import load_page_bundle

_FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
_NOTO_PATH = _FONTS_DIR / "NotoSans-Regular.ttf"
_MIN_FONT_SIZE = 4.0

try:
    from PIL import Image, ImageFilter

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _word_wrap_lines(text: str, font: fitz.Font, font_size: float, max_width: float) -> list[str]:
    """Simulate word-wrapping, returning the lines that would fit.

    Uses 0.95 * max_width as the effective constraint to account for
    differences between font.text_length() and insert_textbox's internal layout.
    """
    effective = max_width * 0.90
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if font.text_length(candidate, fontsize=font_size) <= effective:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_font_size_multiline(
    text: str,
    font: fitz.Font,
    bbox_width: float,
    bbox_height: float,
    max_size: float,
    min_size: float = _MIN_FONT_SIZE,
) -> float:
    if max_size <= min_size:
        return max_size
    lo, hi = min_size, max_size
    best = min_size
    while hi - lo > 0.5:
        mid = (lo + hi) / 2
        lines = _word_wrap_lines(text, font, mid, bbox_width)
        line_height = mid * 1.2
        total_height = len(lines) * line_height
        if total_height <= bbox_height * 0.9:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def _needs_fallback(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if cp > 0xFF and cp != 0x2022:
            return True
    return False


_NOTO_NAME = "notosans"


def _ensure_noto(page: fitz.Page) -> str | None:
    if not _NOTO_PATH.is_file():
        return None
    try:
        page.insert_font(fontname=_NOTO_NAME, fontfile=str(_NOTO_PATH))
        return _NOTO_NAME
    except Exception:
        return None


_REUSED_FONTS: dict[str, fitz.Font] = {}


def _get_font(name: str) -> fitz.Font:
    if name not in _REUSED_FONTS:
        if name == "helv":
            _REUSED_FONTS[name] = fitz.Font(fontname="helv")
        else:
            _REUSED_FONTS[name] = fitz.Font(fontfile=str(_NOTO_PATH))
    return _REUSED_FONTS[name]


def _check_edge_dirty(
    img: Image.Image,
    threshold: float = 15.0,
) -> dict[str, bool]:
    """Check each edge of the cleaned image for residual text pixels.

    Compares the mean color of the outermost 1-pixel strip on each side
    to the mean of the interior (2px inset).  If the distance exceeds
    *threshold*, that side is considered dirty (text bleeds to the edge).

    Returns a dict with keys 'left', 'right', 'top', 'bottom'.
    """
    try:
        from PIL import ImageStat
    except ImportError:
        return {"left": False, "right": False, "top": False, "bottom": False}

    w, h = img.size
    if w < 5 or h < 5:
        return {"left": False, "right": False, "top": False, "bottom": False}

    interior = img.crop((2, 2, w - 2, h - 2))
    interior_mean = ImageStat.Stat(interior).mean[:3]

    edges: dict[str, bool] = {}
    for side, box in (
        ("left", (0, 0, 1, h)),
        ("right", (w - 1, 0, w, h)),
        ("top", (0, 0, w, 1)),
        ("bottom", (0, h - 1, w, h)),
    ):
        strip = img.crop(box)
        edge_mean = ImageStat.Stat(strip).mean[:3]
        dist = sum((e - i) ** 2 for e, i in zip(edge_mean, interior_mean)) ** 0.5
        edges[side] = dist > threshold

    return edges


def _pixmap_clean_and_insert(
    page: fitz.Page,
    rect: fitz.Rect,
    full_pix: fitz.Pixmap | None = None,
    text_color_rgb: tuple[float, float, float] = (0, 0, 0),
    bg_color_rgb: tuple[float, float, float] = (255, 255, 255),
    font_size: float = 12,
) -> bool:
    """Render/crop bbox region, morphologically erode text strokes, insert as background.

    Uses PIL MinFilter (light text) or MaxFilter (dark text) with kernel size
    scaled to *font_size* — erodes strokes into surrounding background.

    Edge bleed prevention: after filtering, each of the 4 edges is checked
    for residual text pixels.  Dirty edges get an additional 1px margin
    (up to 3px) on that side only, then re-cropped and re-filtered.

    Returns True on success (caller should skip the solid whiteout rect).
    """
    if not _HAS_PIL:
        return False

    try:
        if full_pix is not None:
            px_w = full_pix.width
            px_h = full_pix.height
            p_w = page.rect.width
            p_h = page.rect.height
            scale_x = px_w / p_w
            scale_y = px_h / p_h
            full_img = full_pix.pil_image()
        else:
            full_img = None

        # Per-side adaptive margin: only expand edges that show residual text
        margin = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        max_margin = 3

        for iteration in range(max_margin + 1):
            # Crop or render sub-image with current margins
            if full_pix is not None and full_img is not None:
                x0 = max(0, rect.x0 * scale_x - margin["left"])
                y0 = max(0, rect.y0 * scale_y - margin["top"])
                x1 = min(px_w, rect.x1 * scale_x + margin["right"])
                y1 = min(px_h, rect.y1 * scale_y + margin["bottom"])
                if x1 <= x0 or y1 <= y0:
                    return False
                sub_img = full_img.crop((x0, y0, x1, y1))
            else:
                # Render bbox region directly.  At 2x zoom 1 px = 0.5 pt.
                clip = fitz.Rect(
                    rect.x0 - margin["left"] * 0.5,
                    rect.y0 - margin["top"] * 0.5,
                    rect.x1 + margin["right"] * 0.5,
                    rect.y1 + margin["bottom"] * 0.5,
                )
                pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2))
                sub_img = pix.pil_image()

            w, h = sub_img.size
            if w < 3 or h < 3:
                return False

            # Determine if text is lighter than background
            tl = text_color_rgb[0] * 0.299 + text_color_rgb[1] * 0.587 + text_color_rgb[2] * 0.114
            bl = bg_color_rgb[0] * 0.299 + bg_color_rgb[1] * 0.587 + bg_color_rgb[2] * 0.114

            # Kernel size scales with font pt
            ks = max(3, min(31, int(font_size / 2)))
            if ks % 2 == 0:
                ks += 1
            ks = min(ks, w, h)

            if tl > bl:
                cleaned = sub_img.filter(ImageFilter.MinFilter(size=ks))
            else:
                cleaned = sub_img.filter(ImageFilter.MaxFilter(size=ks))

            # Check for residual text at edges
            dirty = _check_edge_dirty(cleaned)
            if not any(dirty.values()):
                break  # All edges clean

            # Expand only dirty sides (cap at max_margin)
            expanded = False
            for side in dirty:
                if dirty[side] and margin[side] < max_margin:
                    margin[side] += 1
                    expanded = True

            if not expanded:
                break  # Can't expand further

        # Insert the final cleaned image
        buf = io.BytesIO()
        cleaned.save(buf, format="PNG")
        clean_pix = fitz.Pixmap(buf.getvalue())
        page.insert_image(rect, pixmap=clean_pix)
        return True
    except Exception:
        return False


def _whiteout_and_insert(
    page: fitz.Page,
    layout: dict,
    text: dict,
    noto_ready: bool,
    use_pixmap_patch: bool = True,
) -> fitz.Page:
    """Insert translated text with background preservation on an existing page.

    When *use_pixmap_patch* is True, each bbox region is rendered to a pixmap,
    median-filtered to remove original text strokes, and inserted as a background
    image — preserving gradients/images behind text.  Falls back to solid whiteout
    when the pixmap approach fails or PIL is unavailable.
    """
    text_map: dict[str, str] = text["spans"]

    bg_pix: fitz.Pixmap | None = None
    if use_pixmap_patch and _HAS_PIL:
        try:
            bg_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        except Exception:
            bg_pix = None

    for span in layout["spans"]:
        span_id = span["id"]
        bbox = span["bbox"]
        translated = text_map.get(span_id, "")

        if not translated.strip():
            continue
        translated = translated.strip()

        rect = fitz.Rect(*bbox)
        bbox_width = rect.width
        bbox_height = rect.height
        original_size = float(span["size"])

        if _needs_fallback(translated):
            if not noto_ready:
                noto_ready = bool(_ensure_noto(page))
            font_name = _NOTO_NAME if noto_ready else "helv"
        else:
            font_name = "helv"

        font_obj = _get_font(font_name)
        fitted_size = _fit_font_size_multiline(
            translated, font_obj, bbox_width, bbox_height, original_size
        )

        color_int = int(span.get("color", 0))
        tr, tg, tb = (color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255
        text_color = (tr / 255, tg / 255, tb / 255)

        bg_color = span.get("bg_color", [255, 255, 255])
        patched = _pixmap_clean_and_insert(
            page, rect, bg_pix,
            text_color_rgb=text_color,
            bg_color_rgb=(bg_color[0] / 255, bg_color[1] / 255, bg_color[2] / 255),
            font_size=original_size,
        )

        if not patched:
            fill_color = (bg_color[0] / 255, bg_color[1] / 255, bg_color[2] / 255)
            page.draw_rect(rect, color=None, fill=fill_color, width=0)

        # Insert with retry — if insert_textbox overflows (negative return),
        # shrink font and retry (re-register font after clean_contents).
        attempt = fitted_size
        for _ in range(5):
            remaining = page.insert_textbox(
                rect, translated, fontname=font_name,
                fontsize=attempt, color=text_color, lineheight=1.2,
            )
            if remaining >= 0:
                break
            page.clean_contents()
            if font_name == _NOTO_NAME:
                _ensure_noto(page)
            attempt *= 0.85

    return page


def _insert_images(page: fitz.Page, images: list[dict], images_dir: Path) -> None:
    """Insert image PNGs at their original bbox positions."""
    for img in images:
        xref = img.get("xref", 0)
        if xref <= 0:
            continue
        img_path = images_dir / f"xref_{xref}.png"
        if not img_path.is_file():
            continue
        rect = fitz.Rect(*img["bbox"])
        page.insert_image(rect, filename=str(img_path))


def _available_space(
    rect: tuple[float, float, float, float],
    occupied: list[tuple[float, float, float, float]],
    page_rect: tuple[float, float, float, float],
) -> dict[str, float]:
    """Scan all 4 directions for empty space around *rect*.

    Returns pt available below, above, right, and left (with 2pt padding).
    Respects page edges as boundaries.  Rects that partially overlap with
    *rect* are treated as blockers on the overlapping side.
    """
    x0, y0, x1, y1 = rect
    px0, py0, px1, py1 = page_rect
    PAD = 2.0

    max_below = py1
    max_above = py0
    max_right = px1
    max_left = px0

    for ob in occupied:
        ox0, oy0, ox1, oy1 = ob
        if not (ox0 < x1 and ox1 > x0):
            continue  # no x-overlap — skip

        # Fully below
        if oy0 >= y1:
            max_below = min(max_below, oy0)
        # Fully above
        elif oy1 <= y0:
            max_above = max(max_above, oy1)
        # Partially overlapping — rect spans into our vertical range
        else:
            # Blocks upward extension: rect's bottom limits our upward growth
            if oy0 < y0 and oy1 > y0:
                max_above = max(max_above, oy1)
            # Blocks downward extension: rect's top limits our downward growth
            if oy0 < y1 and oy1 > y1:
                max_below = min(max_below, oy0)

        if ox0 >= x1 and oy0 < y1 and oy1 > y0:
            max_right = min(max_right, ox0)
        if ox1 <= x0 and oy0 < y1 and oy1 > y0:
            max_left = max(max_left, ox1)

    return {
        "below": max(0, max_below - y1 - PAD),
        "above": max(0, y0 - max_above - PAD),
        "right": max(0, max_right - x1 - PAD),
        "left": max(0, x0 - max_left - PAD),
    }


_SNAP_SIZES = [8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 36.0, 48.0, 60.0]


def _snap_font_size(size: float) -> float:
    for s in reversed(_SNAP_SIZES):
        if s <= size:
            return s
    return size


def _insert_text_only(
    page: fitz.Page,
    layout: dict,
    text: dict,
    noto_ready: bool,
) -> fitz.Page:
    """Place translated text at each span's bbox.

    Groups consecutive spans into paragraphs and inserts merged text at a
    uniform font size.  When text overflows the bbox, extends into available
    empty space (detected via 4-direction collision scan) before shrinking.

    Dynamic rect tracking prevents inserted paragraphs from overlapping with
    subsequent ones.  Final font sizes are snapped to a discrete set for
    visual consistency.
    """
    text_map: dict[str, str] = text["spans"]
    page_rect = (0, 0, float(layout.get("width", 612)), float(layout.get("height", 792)))

    # Dynamic occupied — tracks actual rects as paragraphs are placed
    dynamic_occupied: list[tuple[float, float, float, float]] = [
        tuple(s["bbox"]) for s in layout["spans"]
    ]
    for im in layout.get("images", []):
        dynamic_occupied.append(tuple(im["bbox"]))
    placed_rects: list[tuple[float, float, float, float]] = []

    spans_sorted = sorted(layout["spans"], key=lambda s: (s["bbox"][1], s["bbox"][0]))

    # Group into paragraphs by y-proximity
    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    for s in spans_sorted:
        tid = s["id"]
        if not text_map.get(tid, "").strip():
            if current:
                paragraphs.append(current)
                current = []
            continue
        if not current:
            current.append(s)
            continue
        prev = current[-1]
        gap = s["bbox"][1] - prev["bbox"][3]
        threshold = float(prev.get("size", 10)) * 1.5
        same_size = abs(float(s.get("size", 10)) - float(prev.get("size", 10))) < 0.5
        if same_size and gap < threshold and s["bbox"][0] < prev["bbox"][2]:
            current.append(s)
        else:
            paragraphs.append(current)
            current = [s]
    if current:
        paragraphs.append(current)

    for para in paragraphs:
        merged_text = "".join(text_map.get(s["id"], "") for s in para).strip()
        if not merged_text:
            continue

        original_size = _snap_font_size(float(para[0].get("size", 10)))

        x0 = min(s["bbox"][0] for s in para)
        y0 = min(s["bbox"][1] for s in para)
        x1 = max(s["bbox"][2] for s in para)
        y1 = max(s["bbox"][3] for s in para)
        rect = fitz.Rect(x0, y0, x1, y1)

        # Build occupied: everything in dynamic_occupied except this paragraph's own bboxes
        para_bboxes = {tuple(s["bbox"]) for s in para}
        occupied = [ob for ob in dynamic_occupied if ob not in para_bboxes]

        if _needs_fallback(merged_text):
            if not noto_ready:
                noto_ready = bool(_ensure_noto(page))
            font_name = _NOTO_NAME if noto_ready else "helv"
        else:
            font_name = "helv"

        color_int = int(para[0].get("color", 0))
        tr, tg, tb = (color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255
        text_color = (tr / 255, tg / 255, tb / 255)

        # Nudge down if rect overlaps any previously placed rect
        page_height = float(layout.get("height", 792))
        for _ in range(5):
            target_shift = 0
            for pr in placed_rects:
                if (rect.x0 < pr[2] and rect.x1 > pr[0] and
                    rect.y0 < pr[3] and rect.y1 > pr[1]):
                    shift = pr[3] - rect.y0 + 2
                    if rect.y1 + shift <= page_height - 20:
                        target_shift = shift
                        break
            if target_shift > 0:
                rect = fitz.Rect(rect.x0, rect.y0 + target_shift,
                                 rect.x1, rect.y1 + target_shift)
            else:
                break

        attempt = original_size
        for _ in range(10):
            remaining = page.insert_textbox(
                rect, merged_text, fontname=font_name,
                fontsize=attempt, color=text_color, lineheight=1.2,
            )
            if remaining >= 0:
                break

            # Overflow — try extending bbox before shrinking
            page.clean_contents()
            if font_name == _NOTO_NAME:
                _ensure_noto(page)

            deficit = abs(remaining)
            space = _available_space(
                (rect.x0, rect.y0, rect.x1, rect.y1), occupied, page_rect
            )

            if space["below"] >= deficit:
                rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 + deficit)
                continue
            if space["below"] + space["above"] >= deficit:
                take_below = space["below"]
                rect = fitz.Rect(
                    rect.x0, rect.y0 - (deficit - take_below),
                    rect.x1, rect.y1 + take_below,
                )
                continue
            if rect.width < 150 and space["right"] >= deficit * 0.3:
                take_right = min(space["right"], deficit)
                rect = fitz.Rect(
                    rect.x0, rect.y0,
                    rect.x1 + take_right, rect.y1 + (deficit - take_right),
                )
                continue

            attempt = _snap_font_size(attempt * 0.85)

        # Update dynamic occupied with the actual rect used
        for ob in para_bboxes:
            while ob in dynamic_occupied:
                dynamic_occupied.remove(ob)
        dynamic_occupied.append((rect.x0, rect.y0, rect.x1, rect.y1))
        placed_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))

    return page


def rebuild_page_source(
    doc: fitz.Document,
    page_num: int,
    layout: dict,
    text: dict,
) -> fitz.Page:
    """Rebuild by modifying source PDF page in-place."""
    page = doc[page_num]
    return _whiteout_and_insert(page, layout, text, noto_ready=False, use_pixmap_patch=True)


def rebuild_page_stateless(
    out: fitz.Document,
    layout: dict,
    text: dict,
    images_dir: Path | None,
) -> fitz.Page:
    """Create a fresh page in *out* from layout + text, no source PDF needed."""
    page = out.new_page(
        width=float(layout["width"]),
        height=float(layout["height"]),
    )

    fp = (images_dir or Path()) / f"fullpage_{layout['page']}.jpg"
    if fp.is_file():
        page.insert_image(
            fitz.Rect(0, 0, float(layout["width"]), float(layout["height"])),
            filename=str(fp),
        )
    else:
        _insert_images(page, layout.get("images", []), images_dir or Path())

    noto_ready = bool(_ensure_noto(page))
    return _insert_text_only(page, layout, text, noto_ready=noto_ready)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4/7: reconstruct PDF from layout + text (source-based or stateless).",
    )
    parser.add_argument("source_pdf", nargs="?", type=Path, default=None,
                        help="Original PDF (not needed with --stateless)")
    parser.add_argument("page_dir", type=Path, help="Page directory or parent with --all-pages")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process all page_N/ subdirs under page_dir",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Create PDF from scratch (no source PDF needed)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Images directory (default: <page_dir parent>/images)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated page numbers to skip (e.g. '1,3,5')",
    )
    args = parser.parse_args(argv)

    skip_pages = set()
    if args.exclude:
        for part in args.exclude.split(","):
            part = part.strip()
            if part:
                try:
                    skip_pages.add(int(part))
                except ValueError:
                    pass

    if args.all_pages:
        dirs = sorted(
            [d for d in args.page_dir.iterdir() if d.is_dir() and d.name.startswith("page_")],
            key=lambda d: int(d.name.replace("page_", "")),
        )
        if not dirs:
            print(f"No page_N/ directories found under {args.page_dir}", file=sys.stderr)
            return 1
    else:
        layout_path = args.page_dir / "layout.json"
        text_path = args.page_dir / f"text.{args.lang}.json"
        if not layout_path.is_file() or not text_path.is_file():
            print(f"Missing layout.json or text.{args.lang}.json in {args.page_dir}", file=sys.stderr)
            return 1
        dirs = [args.page_dir]

    # Images live as a sibling of the page_N/ directories.
    if args.all_pages:
        page_parent = args.page_dir
    else:
        page_parent = args.page_dir.parent
    images_dir = args.image_dir or (page_parent / "images")

    if args.stateless:
        out = fitz.open()
        try:
            for page_dir in dirs:
                layout_path = page_dir / "layout.json"
                text_path = page_dir / f"text.{args.lang}.json"
                if not layout_path.is_file() or not text_path.is_file():
                    print(f"  Skip {page_dir.name}: missing files", file=sys.stderr)
                    continue
                layout, text = load_page_bundle(page_dir, lang=args.lang)
                if layout["page"] in skip_pages:
                    print(f"  Skip {page_dir.name} (excluded)")
                    # Still copy the original page for stateless
                    continue
                rebuild_page_stateless(out, layout, text, images_dir)
                print(f"  Rebuilt page {layout['page']} from {page_dir.name} (stateless)")
            out.save(str(args.output), garbage=4, deflate=True)
            print(f"Wrote {args.output}")
        except Exception as exc:
            print(f"Error during stateless rebuild: {exc}", file=sys.stderr)
            return 1
        finally:
            out.close()
        return 0

    if not args.source_pdf or not args.source_pdf.is_file():
        print(f"Source PDF required (use --stateless or provide a valid source path).", file=sys.stderr)
        return 1

    try:
        doc = fitz.open(args.source_pdf)
    except Exception as exc:
        print(f"Error opening source PDF: {exc}", file=sys.stderr)
        return 1

    try:
        for page_dir in dirs:
            layout_path = page_dir / "layout.json"
            text_path = page_dir / f"text.{args.lang}.json"
            if not layout_path.is_file() or not text_path.is_file():
                print(f"  Skip {page_dir.name}: missing files", file=sys.stderr)
                continue
            layout, text = load_page_bundle(page_dir, lang=args.lang)
            page_num = layout["page"]
            if page_num in skip_pages:
                print(f"  Skip page {page_num} from {page_dir.name} (original)")
            else:
                rebuild_page_source(doc, page_num, layout, text)
                print(f"  Rebuilt page {page_num} from {page_dir.name}")
        doc.save(str(args.output), garbage=4, deflate=True)
        print(f"Wrote {args.output}")
    except Exception as exc:
        print(f"Error during rebuild: {exc}", file=sys.stderr)
        return 1
    finally:
        doc.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
