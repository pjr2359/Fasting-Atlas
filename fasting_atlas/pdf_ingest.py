from __future__ import annotations

from dataclasses import dataclass

import pdfplumber


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class WordBox:
    page_number: int
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass
class IngestedTable:
    table_id: str
    page_number: int
    rows: list[list[str]]


@dataclass
class IngestedPaper:
    source_file: str
    pages: list[PageText]
    words: list[WordBox]
    tables: list[IngestedTable]


def ingest_pdf(pdf_path: str) -> IngestedPaper:
    pages: list[PageText] = []
    words: list[WordBox] = []
    tables: list[IngestedTable] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=page_idx, text=text))

            for word in page.extract_words() or []:
                words.append(
                    WordBox(
                        page_number=page_idx,
                        text=str(word.get("text", "")),
                        x0=float(word.get("x0", 0.0)),
                        top=float(word.get("top", 0.0)),
                        x1=float(word.get("x1", 0.0)),
                        bottom=float(word.get("bottom", 0.0)),
                    )
                )

            raw_tables = page.extract_tables() or []
            for table_idx, raw_table in enumerate(raw_tables, start=1):
                clean_rows: list[list[str]] = []
                for row in raw_table or []:
                    clean_rows.append([("" if cell is None else str(cell).strip()) for cell in row])
                tables.append(
                    IngestedTable(
                        table_id=f"p{page_idx}_t{table_idx}",
                        page_number=page_idx,
                        rows=clean_rows,
                    )
                )

    return IngestedPaper(source_file=pdf_path, pages=pages, words=words, tables=tables)
