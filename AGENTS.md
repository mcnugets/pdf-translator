# pdf-translator

Pipeline: **Extract → Translate → Rebuild**. Preserves PDF layout (images, vectors, colors) while replacing text with translations.

## CLI entrypoints

```bash
# Phase 0 — inspect spans on a page
python -m src.inspect_spans input.pdf --page 0
python -m src.inspect_spans input.pdf --page 0 --ocr              # include OCR
python -m src.inspect_spans input.pdf --page 0 --ocr --ocr-backend paddle

# Phase 1 — extract layout.json + text.en.json
python -m src.extract input.pdf --page 0 --out pages/page_0
python -m src.extract input.pdf --all-pages --out pages/          # batch
python -m src.extract input.pdf --page 0 --out pages/page_0 --no-ocr
python -m src.extract input.pdf --page 0 --out pages/page_0 --ocr-backend paddle

# Phase 2 — rebuild PDF from JSON
python -m src.rebuild input.pdf pages/page_0 -o out.pdf
python -m src.rebuild input.pdf pages/page_0 -o out.pdf --lang ru
python -m src.rebuild input.pdf pages/ -o out.pdf --lang ru --all-pages  # batch
python -m src.rebuild -o out.pdf pages/page_0 --stateless             # no source PDF needed
python -m src.rebuild -o out.pdf pages/ --all-pages --stateless       # batch stateless
python -m src.rebuild input.pdf pages/ -o out.pdf --lang ru --all-pages --exclude 1    # skip page 1

# Phase 3 — translate text files
python -m src.translate pages/page_0 --from en --to ru
python -m src.translate pages/page_0 --from en --to ru --force  # re-translate
python -m src.translate pages/ --all-pages --from en --to ru     # batch all page_N/
```

**Gotcha:** On Windows, prefix with `PYTHONIOENCODING=utf-8` to avoid `UnicodeEncodeError` from the `→` character in print statements.

## Data model

- `layout.json` — immutable geometry (bbox, font, size, color, flags, bg_color, source)
- `text.{lang}.json` — swappable per language, maps `span_id → string`

Both written by `spans.py:write_page_bundle()`. Core type is `SpanRecord` dataclass.

## OCR backends

`detect_ocr_backend()` in `spans.py` auto-selects: GPU-available → PaddleOCR, else → Tesseract.

| Backend | Install | Override |
|---------|---------|----------|
| **Tesseract** (default, recommended for weak systems) | Download binary from [GitHub](https://github.com/tesseract-ocr/tesseract), then `pip install pytesseract` | `--ocr-backend tesseract` or `TESSERACT_CMD` env var |
| **PaddleOCR** (GPU, more accurate) | `pip install paddleocr paddlepaddle` (needs Python 3.9–3.12) | `--ocr-backend paddle` |

Tesseract path is hardcoded to `C:\Program Files\Tesseract-OCR\tesseract.exe` in `spans.py:19-21` — override via `TESSERACT_CMD` env var.

## Development

```bash
pip install -r requirements.txt
python -c "from src.spans import detect_ocr_backend; print(detect_ocr_backend())"  # verify setup
```

Phases follow `docs/pdf-translation-architecture.md` (0–7). Currently at Phase 7. No test framework set up yet.

## Font fallback

`fonts/NotoSans-Regular.ttf` is bundled for non-Latin scripts (Cyrillic, CJK, etc.). `rebuild.py` auto-detects glyph coverage: text outside Latin-1 uses Noto Sans, otherwise uses built-in `helv`. Font size is binary-searched to fit within the original bbox width (`_fit_font_size()` with 4pt minimum).

## Architecture doc

`docs/pdf-translation-architecture.md` covers the full plan — strategies (overlay/whiteout vs stream edit vs OCR rebuild), stateless data model, phased rollout roadmap, and challenges.
