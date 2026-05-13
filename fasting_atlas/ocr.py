"""OCR for scanned or low-text PDF pages using PyMuPDF render + Tesseract."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fasting_atlas.pdf_ingest import IngestedPaper


def ocr_lang_tesseract(lang: str) -> str:
    """Map CLI `eng+deu+rus` style to Tesseract language string."""
    return lang.strip().replace(",", "+").replace(" ", "")


def should_ocr_page(text: str, word_count: int, mode: str) -> bool:
    if mode == "force":
        return True
    if mode == "off":
        return False
    stripped = text.strip()
    if len(stripped) < 50 and word_count < 12:
        return True
    return False


def apply_ocr_to_ingest(
    ingested: "IngestedPaper",
    *,
    mode: str,
    ocr_lang: str,
    dpi: float = 200.0,
) -> "IngestedPaper":
    """Augment/replace per-page text and words when OCR policy applies."""
    from fasting_atlas.pdf_ingest import IngestedPaper, PageText, WordBox

    words_by_page: dict[int, list] = defaultdict(list)
    for w in ingested.words:
        words_by_page[w.page_number].append(w)

    if mode == "off":
        pages = [
            PageText(page_number=p.page_number, text=p.text, text_source=getattr(p, "text_source", "digital"))
            for p in ingested.pages
        ]
        words = [
            WordBox(
                page_number=w.page_number,
                text=w.text,
                x0=w.x0,
                top=w.top,
                x1=w.x1,
                bottom=w.bottom,
                source=getattr(w, "source", "digital"),
            )
            for w in ingested.words
        ]
        return IngestedPaper(
            source_file=ingested.source_file,
            pages=pages,
            words=words,
            tables=ingested.tables,
        )

    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires optional dependencies: pip install pymupdf pytesseract pillow. "
            "Also install system tesseract-ocr and language packs."
        ) from exc

    lang = ocr_lang_tesseract(ocr_lang)
    doc = fitz.open(ingested.source_file)

    new_pages: list[PageText] = []
    ocr_words_by_page: dict[int, list[WordBox]] = defaultdict(list)

    for page in ingested.pages:
        pnum = page.page_number
        wc = len(words_by_page.get(pnum, []))
        use_ocr = should_ocr_page(page.text, wc, mode)

        if not use_ocr:
            new_pages.append(
                PageText(
                    page_number=pnum,
                    text=page.text,
                    text_source=getattr(page, "text_source", "digital"),
                )
            )
            continue

        fitz_page = doc[pnum - 1]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        tess = pytesseract.image_to_data(img, lang=lang or "eng", output_type=pytesseract.Output.DICT)

        rect = fitz_page.rect
        sx = rect.width / float(pix.width)
        sy = rect.height / float(pix.height)

        ocr_text_parts: list[str] = []
        for i in range(len(tess["text"])):
            word = str(tess["text"][i] or "").strip()
            if not word:
                continue
            try:
                level = int(tess["level"][i])
            except (TypeError, ValueError, KeyError):
                level = 0
            if level < 5:
                continue
            try:
                conf = float(tess["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf != -1.0 and conf < 30.0:
                continue
            left = float(tess["left"][i])
            top = float(tess["top"][i])
            w = float(tess["width"][i])
            h = float(tess["height"][i])
            x0 = left * sx + rect.x0
            y0 = top * sy + rect.y0
            x1 = (left + w) * sx + rect.x0
            y1 = (top + h) * sy + rect.y0
            ocr_words_by_page[pnum].append(
                WordBox(page_number=pnum, text=word, x0=x0, top=y0, x1=x1, bottom=y1, source="ocr")
            )
            ocr_text_parts.append(word)

        merged_text = " ".join(ocr_text_parts).strip()
        if merged_text:
            new_pages.append(PageText(page_number=pnum, text=merged_text, text_source="ocr"))
        else:
            new_pages.append(
                PageText(
                    page_number=pnum,
                    text=page.text,
                    text_source="mixed",
                )
            )

    doc.close()

    ocr_page_nums = {p.page_number for p in new_pages if p.text_source == "ocr"}
    final_words: list[WordBox] = []
    for w in ingested.words:
        if w.page_number in ocr_page_nums:
            continue
        final_words.append(
            WordBox(
                page_number=w.page_number,
                text=w.text,
                x0=w.x0,
                top=w.top,
                x1=w.x1,
                bottom=w.bottom,
                source=getattr(w, "source", "digital"),
            )
        )
    for pnum in sorted(ocr_words_by_page.keys()):
        final_words.extend(ocr_words_by_page[pnum])
    final_words.sort(key=lambda x: (x.page_number, x.top, x.x0))

    return IngestedPaper(
        source_file=ingested.source_file,
        pages=new_pages,
        words=final_words,
        tables=ingested.tables,
    )
