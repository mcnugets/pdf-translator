# Project Progress

Generated: 2026-05-29

## Phase Completion

| Phase | Status | Notes |
|-------|--------|-------|
| **0** — Inspect spans | ✅ Done | `inspect_spans.py` — debug dump of text spans per page |
| **1** — Extract JSON | ✅ Done | `layout.json` + `text.en.json` per page; bg_color sampling |
| **2** — Dumb rebuild | ✅ Done | Whiteout + overlay on source PDF canvas |
| **3** — Translate | ⚠️ Partial | Span-by-span GoogleTranslate works; **per-page `[0]…[N]` batching** for context-aware marketing copy not implemented |
| **4** — Fit font | ✅ Done | Binary search to bbox width, 4pt min, Noto Sans fallback for non-Latin-1 |
| **5** — Smarter whiteout | ⚠️ Partial | Solid bg color sampling works (edge-pixel mean). **Pixmap clip-patch** for gradients/text-over-image not implemented |
| **6** — Full doc | ⚠️ Partial | `--all-pages` flags work for all 3 CLIs. **Jinja2 HTML preview** and **asyncio** not implemented (libs installed but unused) |
| **7** — Stateless rebuild | ✅ Done | `--stateless` flag; images extracted to `images/xref_N.png`; no source PDF at rebuild |

## Known Gaps

### Font style ignored
`layout.json` stores `flags` (bold/italic/underline) but `rebuild.py` always uses `helv` or `notosans` — never `Helvetica-Bold`, `Helvetica-Oblique`, etc.

### No reading order inference
Multi-line paragraphs break into independent spans, translated separately. No line-merging or paragraph grouping.

### No word-wrapping
`_fit_font_size` scales the whole line; long words just shrink down to 4pt min, potentially clipping.

### Background color sampling fragile
Samples a thin 2px margin at 72 DPI. Wrong if span is adjacent to a different-colored region. No fallback for gradients (Phase 5 step 2).

### `_needs_fallback` heuristic crude
Triggers on any code point > `0xFF` except `0x2022` (bullet). Misses Latin-1 Supplement chars that Helvetica can handle; also won't detect Noto lacking a glyph.

### Translation is span-by-span
Marketing copy loses context. Per-page `[0]…[N]` batching would preserve tone (arch doc Challenge #5).

### No test framework
No pytest config, no test scripts, no CI.

## Project Config Issues

- `pyproject.toml` missing `numpy` and `pytesseract` from dependencies
- `pyproject.toml` sets `pythonVersion = "3.14"` — conflicts with PaddleOCR needing 3.9–3.12
- Hardcoded Tesseract path in `spans.py:19-21` — override via `TESSERACT_CMD` env var
- `aiofiles` and `Jinja2` installed but never used (dead weight)
- On Windows, `PYTHONIOENCODING=utf-8` prefix needed to avoid `UnicodeEncodeError`

## What's Left (Priority Order)

1. **Phase 5 step 2** — pixmap clip-patch for gradient/image backgrounds
2. **Font bold/italic** — map `flags` to Helvetica variants in rebuild
3. **Per-page translation batching** — `[0]…[N]` markers for marketing context
4. **Test framework** — pytest setup with golden-page fixture
5. **Reading order** — merge lines into paragraphs before translation
6. **Jinja2 HTML preview** — QA step from layout + text
7. **Pyproject.toml cleanup** — deps, python version, type-check paths
