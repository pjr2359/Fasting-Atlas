from __future__ import annotations

from dataclasses import dataclass

from fasting_atlas.pdf_ingest import PageText

METHODS_KEYWORDS = (
    "methods",
    "materials and methods",
    "participants",
    "subjects",
    "study design",
)

RESULTS_KEYWORDS = (
    "results",
    "findings",
    "outcomes",
)


@dataclass
class SectionBlock:
    section_name: str
    pages: list[int]
    text: str


def _collect_by_keywords(pages: list[PageText], section_name: str, keywords: tuple[str, ...]) -> SectionBlock:
    matched: list[PageText] = []
    for page in pages:
        lowered = page.text.lower()
        if any(keyword in lowered for keyword in keywords):
            matched.append(page)

    # Fallback: early paper pages often contain methods/results in short papers.
    if not matched and pages:
        matched = pages[: min(4, len(pages))]

    return SectionBlock(
        section_name=section_name,
        pages=[p.page_number for p in matched],
        text="\n\n".join(p.text for p in matched).strip(),
    )


def find_methods_and_results_sections(pages: list[PageText]) -> tuple[SectionBlock, SectionBlock]:
    methods = _collect_by_keywords(pages, "methods_participants", METHODS_KEYWORDS)
    results = _collect_by_keywords(pages, "narrative_results", RESULTS_KEYWORDS)
    return methods, results
