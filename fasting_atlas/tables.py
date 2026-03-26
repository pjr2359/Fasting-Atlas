from __future__ import annotations

import re
from typing import Any

from fasting_atlas.llm_client import JsonLLM
from fasting_atlas.pdf_ingest import IngestedTable
from fasting_atlas.schemas import EvidenceCell, TableExtracted

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def extract_tables(
    source_file: str,
    ingested_tables: list[IngestedTable],
    client: JsonLLM,
) -> list[TableExtracted]:
    extracted: list[TableExtracted] = []
    for table in ingested_tables:
        headers = table.rows[0] if table.rows else []
        data_rows = table.rows[1:] if len(table.rows) > 1 else []
        cells: list[EvidenceCell] = []
        for row_idx, row in enumerate(table.rows):
            for col_idx, value in enumerate(row):
                numeric = _parse_numeric(value)
                cells.append(
                    EvidenceCell(
                        source_file=source_file,
                        table_id=table.table_id,
                        page=table.page_number,
                        row=row_idx,
                        col=col_idx,
                        text=value,
                        numeric_value=numeric,
                    )
                )

        variable_hints: list[str] = []
        unit_hints: list[str] = []
        confidence = 0.55
        method = "pdfplumber"

        if headers:
            hints = _infer_table_hints(client, headers, table.page_number)
            variable_hints = [str(x) for x in hints.get("variable_hints", []) if str(x).strip()]
            unit_hints = [str(x) for x in hints.get("unit_hints", []) if str(x).strip()]
            confidence = 0.75
            method = "pdfplumber+llm_hints"

        extracted.append(
            TableExtracted(
                table_id=table.table_id,
                page=table.page_number,
                headers=headers,
                rows=data_rows,
                cells=cells,
                variable_hints=variable_hints,
                unit_hints=unit_hints,
                confidence=confidence,
                extraction_method=method,
            )
        )
    return extracted


def _infer_table_hints(client: JsonLLM, headers: list[str], page_number: int) -> dict[str, Any]:
    schema_hint = '{"variable_hints":["string"],"unit_hints":["string"]}'
    return client.extract_json(
        prompt=(
            f"Given these table headers from page {page_number}, infer likely variables and units.\n"
            f"Headers: {headers}"
        ),
        schema_hint=schema_hint,
        temperature=0.2,
    )


def _parse_numeric(value: str) -> float | None:
    compact = value.replace(",", "").strip()
    if NUM_RE.match(compact):
        try:
            return float(compact)
        except ValueError:
            return None
    return None
