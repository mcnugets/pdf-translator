"""Create a minimal sample PDF for trying phase 0 without a real brochure."""

import pymupdf as fitz
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_page.pdf"


def main() -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Buy now", fontsize=24, color=(0, 0, 0))
    page.insert_text((72, 140), "Limited offer — act today", fontsize=12, color=(0.2, 0.2, 0.2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    doc.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
