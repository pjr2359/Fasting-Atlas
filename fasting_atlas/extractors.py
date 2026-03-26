from __future__ import annotations

from typing import Any

from fasting_atlas.llm_client import JsonLLM
from fasting_atlas.schemas import EvidenceText, MethodsParticipantsItem, Metadata, NarrativeResultsItem
from fasting_atlas.sections import SectionBlock


def extract_metadata_llm(source_file: str, pages_preview: str, client: JsonLLM) -> Metadata:
    """Structured metadata via LLM only. Empty preview returns empty metadata."""
    preview = pages_preview.strip()
    if not preview:
        return Metadata()

    schema_hint = (
        '{"title":"string|null","authors":["string"],"year":number|null,"journal":"string|null",'
        '"study_design":"string|null","demographics":"string|null",'
        '"evidence":[{"page":1,"quote":"verbatim from the paper"}]}'
    )
    parsed = client.extract_json(
        prompt=(
            "From this scientific paper front matter (title page / first pages), extract metadata. "
            "Use null when unknown. Every non-null field should have at least one evidence quote from the text.\n\n"
            f"{preview[:12000]}"
        ),
        schema_hint=schema_hint,
        temperature=0.1,
    )

    evidence_list = [
        EvidenceText(
            source_file=source_file,
            page=int(ev.get("page", 1)),
            quote=str(ev.get("quote", "")),
        )
        for ev in parsed.get("evidence", [])
        if str(ev.get("quote", "")).strip()
    ]

    year_raw = parsed.get("year")
    year: int | None
    if year_raw is None:
        year = None
    else:
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            year = None

    return Metadata(
        title=parsed.get("title"),
        authors=[str(a) for a in parsed.get("authors", []) if str(a).strip()],
        year=year,
        journal=parsed.get("journal"),
        study_design=parsed.get("study_design"),
        demographics=parsed.get("demographics"),
        evidence=evidence_list,
    )


def extract_methods_items(
    source_file: str,
    section: SectionBlock,
    client: JsonLLM,
    temperature: float = 0.1,
) -> list[MethodsParticipantsItem]:
    if not section.text.strip():
        return []

    schema_hint = (
        '{"items":[{"label":"string","value":"string","method":"string|null","unit":"string|null",'
        '"confidence":0.0,"evidence":[{"page":1,"quote":"exact snippet"}]}]}'
    )
    payload = section.text[:16000]
    parsed = client.extract_json(
        prompt=(
            "Extract discrete methods and participant facts (assays, timing, dosing, N, inclusion). "
            "Each item must include evidence quotes copied verbatim from the text below.\n\n"
            f"{payload}"
        ),
        schema_hint=schema_hint,
        temperature=temperature,
    )
    return _parse_methods_items(source_file, parsed)


def extract_narrative_results_items(
    source_file: str,
    section: SectionBlock,
    client: JsonLLM,
    temperature: float = 0.1,
) -> list[NarrativeResultsItem]:
    if not section.text.strip():
        return []

    schema_hint = (
        '{"items":[{"statement":"string","relationship":"string|null","confidence":0.0,'
        '"evidence":[{"page":1,"quote":"exact snippet"}]}]}'
    )
    payload = section.text[:16000]
    parsed = client.extract_json(
        prompt=(
            "Extract narrative result statements (findings, comparisons, directions of effect). "
            "Each item must include evidence quotes verbatim from the text below.\n\n"
            f"{payload}"
        ),
        schema_hint=schema_hint,
        temperature=temperature,
    )
    return _parse_results_items(source_file, parsed)


def _parse_methods_items(source_file: str, parsed: dict[str, Any]) -> list[MethodsParticipantsItem]:
    items: list[MethodsParticipantsItem] = []
    for item in parsed.get("items", []):
        evidence = [
            EvidenceText(
                source_file=source_file,
                page=int(ev.get("page", 1)),
                quote=str(ev.get("quote", "")),
            )
            for ev in item.get("evidence", [])
            if str(ev.get("quote", "")).strip()
        ]
        items.append(
            MethodsParticipantsItem(
                label=str(item.get("label", "method_detail")),
                value=str(item.get("value", "")),
                method=item.get("method"),
                unit=item.get("unit"),
                confidence=float(item.get("confidence", 0.6)),
                extraction_method="llm",
                evidence=evidence,
            )
        )
    return items


def _parse_results_items(source_file: str, parsed: dict[str, Any]) -> list[NarrativeResultsItem]:
    items: list[NarrativeResultsItem] = []
    for item in parsed.get("items", []):
        evidence = [
            EvidenceText(
                source_file=source_file,
                page=int(ev.get("page", 1)),
                quote=str(ev.get("quote", "")),
            )
            for ev in item.get("evidence", [])
            if str(ev.get("quote", "")).strip()
        ]
        items.append(
            NarrativeResultsItem(
                statement=str(item.get("statement", "")),
                relationship=item.get("relationship"),
                confidence=float(item.get("confidence", 0.6)),
                extraction_method="llm",
                evidence=evidence,
            )
        )
    return items
