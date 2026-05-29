# PDF Translation Architecture

## The Problem

PDFs are not documents in the word-processor sense — they are absolute-position rendering instructions. Text exists as scattered character and span objects with baked-in coordinates. There is no native concept of a paragraph or text box. In-place editing is therefore genuinely hard.

For design-heavy brochures (e.g. product catalogs), the goal is to:

- Preserve images, vector graphics, and layout exactly
- Replace source-language copy with translated copy in the same visual positions
- Support multiple target languages from a single extraction pass

The core challenge is reconciling **layout fidelity** (coordinates, fonts, colors) with **translation semantics** (context, tone, length differences between languages).

---

## Tool Stack

| Concern | Library / approach |
|--------|---------------------|
| PDF read/write | **PyMuPDF** (`fitz`) — extraction, overlay, reconstruction |
| Translation | **deep-translator** (`GoogleTranslator` — unofficial Google Translate; no API key); **Anthropic / OpenAI** optional for context-aware marketing copy |
| Background color sampling | PyMuPDF + **PIL** for whiteout rects that match page background |
| Universal font (Cyrillic, CJK, etc.) | Bundle **Noto Sans** as fallback when embedded subsets lack glyphs |
| Async pipeline | **asyncio** + **aiofiles** for batched, rate-limited API calls |
| QA preview (optional) | **Jinja2** → HTML positioned preview (not used for PDF output) |

### Recommended pipeline (overlay / whiteout)

Best for brochures where images and graphics must remain untouched:

```
Input PDF
    → [1] TextExtractor     — pymupdf get_text("dict") → TextSpan[] per page
    → [2] TranslationBatcher — per-page tagged blocks → deep-translator / optional LLM → translated strings
    → [3] PDFReconstructor  — sample bg, whiteout rects, auto-scale font, insert text
    → Output PDF (vectors/images unchanged)
```

**Do not translate span-by-span** — marketing copy loses context. Batch per page with position markers (`[0]`, `[1]`, …) and parse translations back in the same format.

---

## Strategies (Complexity vs Robustness)

### Strategy 1: Overlay / whiteout (recommended for brochures)

1. Extract text spans (bbox, text, color, font size, flags)
2. Translate batched blocks per page
3. Paint matching-color rectangles over original text
4. Insert translated text with auto-scaled font
5. Write output PDF without modifying the underlying content stream

**Why it wins:** Images, vectors, and bleeds stay bit-for-bit intact; you only paint on top.

### Strategy 2: Direct stream edit (pikepdf)

Edit the PDF content stream directly. More faithful to original typography but fragile: subset-embedded fonts may lack glyphs for the target script (e.g. EN → RU, EN → ZH). Only viable when source and target share a similar charset (e.g. EN → DE, EN → FR) unless font substitution is implemented.

### Strategy 3: Render → OCR → rebuild (nuclear)

Rasterize pages, OCR bounding boxes, overlay translated text, repackage as PDF. Fully rasterized output; no vector crispness. Use only for scanned or image-only PDFs.

---

## Stateless Data Model: Two JSON Files

Store layout and text separately so reconstruction is deterministic and language-agnostic.

| File | Role |
|------|------|
| `layout.json` | Immutable after extraction — geometry, fonts, colors, image refs |
| `text.{lang}.json` | Swappable per language — span ID → string |

**Why not Jinja2 for PDF?** Reconstruction needs keyed lookup of each span by ID at specific coordinates, not a rendered string. Jinja2 fits an optional **HTML QA preview** step only:

```
layout.json + text.ru.json → Jinja2 → review.html   (QA)
layout.json + text.ru.json → reconstructor → out.pdf (production)
```

### Example `layout.json` (per page)

```json
{
  "page": 0,
  "width": 595.0,
  "height": 842.0,
  "spans": [
    {
      "id": "span_0",
      "bbox": [72.0, 100.0, 400.0, 124.0],
      "font": "Helvetica-Bold",
      "size": 24.0,
      "color": 0,
      "flags": 20,
      "bg_color": [255, 255, 255]
    }
  ],
  "images": [
    { "xref": 5, "bbox": [0, 0, 595, 200] }
  ]
}
```

### Example `text.{lang}.json`

```json
{
  "page": 0,
  "lang": "ru",
  "spans": {
    "span_0": "Купить сейчас",
    "span_1": "Ограниченное предложение"
  }
}
```

### Directory layout

```
brochures/
  campaign_a/
    pages/
      page_0/
        layout.json
        text.en.json
        text.ru.json
        text.de.json
      page_1/
        ...
    images/
      xref_5.png
      xref_8.png
    output/
      campaign_a.ru.pdf
```

Images are extracted once and referenced by `xref` in layout; the reconstructor loads them independently. One layout, N language files; git-diffable strings; patch individual spans without re-extraction.

Reconstruction is a pure function:

```python
def reconstruct(layout: PageLayout, text: PageText) -> fitz.Document:
    ...
```

---

## Challenges

### 1. No document structure in PDFs

Text is a flat list of positioned spans. Grouping, reading order, and “logical” blocks must be inferred or tagged explicitly for translation batching.

### 2. Translation length and font scaling

Russian is often ~30% longer than English; German compounds can be much longer. Translated text must be auto-scaled to fit the original bbox (binary search on font size against bbox width). A bundled universal font (Noto Sans) is required when embedded subsets do not cover the target script.

### 3. Background whiteout on non-solid areas

Solid-color background sampling works well for simple fills. On **gradients** or **text over images**, a flat rectangle will visibly block the background. Mitigations:

- Use `page.get_pixmap(clip=bbox)` to sample the exact pixel region and match fill, or
- Render that region as a sub-image and composite it behind the new text

More work, but necessary for high-quality brochures.

### 4. Font and glyph coverage

Embedded fonts are often subset. Direct stream replacement breaks silently when Cyrillic/CJK glyphs are missing. Overlay strategy + Noto fallback avoids editing streams but requires consistent placement and scaling.

### 5. Translation quality vs structure

Per-span calls lose marketing tone and context. Per-page tagged batches preserve context but require reliable parsing of `[N]` markers in model output.

**deep-translator** (`GoogleTranslator`) wraps Google’s public translate endpoint (unofficial — not Cloud Translation API): no API key, but rate limits, occasional blocks, and weaker marketing tone than an LLM. It can break when Google changes the site. Batch per page with backoff/retries; cache in `text.{lang}.json` so reconstruction does not re-translate. (`googletrans` does not run on Python 3.13+.)

### 6. Stateless reconstruction contract

The reconstructor must depend only on `layout.json`, `text.{lang}.json`, and extracted image assets — not the source PDF — so builds are reproducible, testable, and cacheable per language.

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│  Input: PDF brochure                        │
└────────────────┬────────────────────────────┘
                 │
         [1] TextExtractor
             pymupdf.get_text("dict")
             → layout.json + text.en.json per page
                 │
         [2] TranslationBatcher
             Per-page context blocks → API
             → text.{target}.json per page
                 │
         [3] PDFReconstructor
             Sample bg → whiteout → fit font → insert text
                 │
┌────────────────▼────────────────────────────┐
│  Output: translated.pdf                     │
│  (images/vectors unchanged where possible)  │
└─────────────────────────────────────────────┘
```

---

## Gradual Rollout

Build **extract → (optional translate) → rebuild** as three separate CLIs early. Introduce `layout.json` + `text.{lang}.json` in phase 1 so work is never thrown away. Defer async, Noto, pixmap whiteout, and multi-language until a **single page with solid white background** looks acceptable.

Use one **golden page** (a hard representative page from a real brochure) as the test fixture from phase 0 through phase 6.

```
Phase 0: see text on page
    → Phase 1: JSON on disk
    → Phase 2: ugly PDF rebuild
    → Phase 3: translate
    → Phase 4: fit font
    → Phase 5: better whiteout
    → Phase 6: full document + QA
    → Phase 7: pure stateless rebuild (later)
```

### Phase 0 — See the problem (~½ day)

**Build:** Open PDF with PyMuPDF; dump `get_text("dict")` for one page; print span `bbox`, `text`, `size`, `color`.

**Skip:** Translation, reconstruction, JSON schema, fonts.

**Done when:** Spans on screen match printed bboxes. Golden page chosen.

### Phase 1 — Stateless files, no PDF output (~1 day)

**Build:** Extractor writes per page:

- `layout.json` — ids, bboxes, font metadata, `bg_color` (default white for now)
- `text.en.json` — `span_id → string` from extraction

**Skip:** translation, reconstructor, image extraction.

**Done when:** Re-running extract produces identical JSON. Hand-edits to `text.en.json` are diffable in git.

### Phase 2 — Dumb reconstructor, one page (~1–2 days)

**Build:** `rebuild.py layout.json text.en.json → out.pdf` using the **same source PDF** as canvas:

- White rectangles over each bbox (fixed white, no sampling)
- `insert_text` with original font size, built-in `helv`
- Open source PDF, draw on top, save

**Skip:** Translation, font scaling, Noto, batching, pixmap backgrounds.

**Done when:** Re-inserted English looks acceptable on white areas. Overlap is expected until phase 4.

### Phase 3 — Translation as a separate step (~1 day)

**Build:** `translate.py text.en.json → text.ru.json`

- Start span-by-span with `GoogleTranslator` (easiest to debug)
- Simple sleep/retry on errors
- Rebuild with `text.ru.json`

**Skip:** Per-page `[N]` batching, LLM, async.

**Done when:** One page is target-language end-to-end.

**Upgrade (same phase):** Per-page tagged batch `[0] … [N]` for better marketing copy; parse markers back into `text.{lang}.json`.

### Phase 4 — Fit text in the box (~1 day)

**Build:** `fit_text_to_bbox` — shrink font until width ≤ bbox; respect `min_size`.

**Skip:** Noto (until glyphs break), gradient backgrounds.

**Done when:** Longer translations do not spill badly on headlines and buttons.

**Then:** Add Noto Sans fallback only for spans that show missing glyphs.

### Phase 5 — Smarter whiteout (~1–2 days)

**Build in order:**

1. Sample solid `bg_color` per bbox
2. If still ugly: `get_pixmap(clip=bbox)` patch for text-on-image / gradients

**Skip:** Full campaign folder layout, HTML preview.

**Done when:** Golden page has no obvious white boxes on colored panels.

### Phase 6 — Full document + ergonomics (~2–3 days)

**Build:**

- All pages: `extract → translate → rebuild`
- CLI: `extract | translate | rebuild` with `--page` and `--lang`
- Cache: skip translation if `text.{lang}.json` exists unless `--force`
- Optional: Jinja2 HTML preview from `layout + text.{lang}` (QA only)

**Skip:** Reconstructing without source PDF; asyncio until batching many pages hurts.

**Done when:** Full brochure PDF in one workflow; review HTML or PDF before shipping.

### Phase 7 — Pure stateless rebuild (later)

**Build:** Extract images to `images/xref_N.png`; reconstructor uses only JSON + images (no source PDF).

**Why last:** Highest integration cost; little value until phases 2–6 work on real brochures.

### What not to do early

| Temptation | Wait until |
|------------|------------|
| Per-page LLM batching | Phase 3 span-by-span works |
| Noto / CJK | Phase 4 + missing glyphs visible |
| Pixmap whiteout | Phase 5 + solid color fails visually |
| `asyncio` | Phase 6 + many pages / rate limits |
| pikepdf stream edit | Only if overlay path fails entirely |
| OCR / raster rebuild | Scanned PDFs only |

### Suggested repo shape

```
pdf_translator/
  extract.py      # phase 1+
  translate.py    # phase 3+
  rebuild.py      # phase 2+
  models.py       # TextSpan, layout schema
tests/
  fixtures/
    golden_page.pdf
    page_0/         # layout.json + text.en.json after phase 1
```

### MVP exit criteria

One golden page, source → target language (e.g. EN→RU), whiteout good on solid backgrounds, font scaled, pipeline cached in JSON. Later phases polish the same three scripts — not a rewrite.

### Python dependencies

Install from project root (see `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Package | Used from |
|---------|-----------|
| `pymupdf` | Phase 0+ |
| `deep-translator` | Phase 3+ |
| `Pillow` | Phase 5+ (background sampling) |
| `aiofiles` | Phase 6+ (optional async I/O) |
| `Jinja2` | Phase 6+ (optional HTML QA) |

Noto Sans is a font bundle on disk, not a pip package — add when phase 4 needs non-Latin scripts.
