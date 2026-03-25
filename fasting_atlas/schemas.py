from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceText(BaseModel):
    source_file: str
    page: int
    quote: str
    bbox: list[float] | None = None


class EvidenceCell(BaseModel):
    source_file: str
    table_id: str
    page: int
    row: int
    col: int
    text: str
    numeric_value: float | None = None
    bbox: list[float] | None = None


class Metadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    study_design: str | None = None
    demographics: str | None = None
    evidence: list[EvidenceText] = Field(default_factory=list)


class MethodsParticipantsItem(BaseModel):
    label: str
    value: str
    method: str | None = None
    unit: str | None = None
    confidence: float = 0.0
    extraction_method: str = "auto"
    evidence: list[EvidenceText] = Field(default_factory=list)


class NarrativeResultsItem(BaseModel):
    statement: str
    relationship: str | None = None
    confidence: float = 0.0
    extraction_method: str = "auto"
    evidence: list[EvidenceText] = Field(default_factory=list)


class TableExtracted(BaseModel):
    table_id: str
    page: int
    caption: str | None = None
    variable_hints: list[str] = Field(default_factory=list)
    unit_hints: list[str] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[EvidenceCell] = Field(default_factory=list)
    extraction_method: str = "auto"
    confidence: float = 0.0


class CouncilDiscrepancy(BaseModel):
    path: str
    pass_1_value: Any
    pass_2_value: Any
    severity: str = "medium"


class CouncilResult(BaseModel):
    pass_1_model: str
    pass_2_model: str
    discrepancies: list[CouncilDiscrepancy] = Field(default_factory=list)
    needs_human_review: bool = False


class QAResult(BaseModel):
    council: CouncilResult


class PaperExtraction(BaseModel):
    paper_id: str
    source_file: str
    metadata: Metadata
    methods_participants: list[MethodsParticipantsItem] = Field(default_factory=list)
    narrative_results: list[NarrativeResultsItem] = Field(default_factory=list)
    tables: list[TableExtracted] = Field(default_factory=list)
    figures_graphs: list[dict[str, Any]] = Field(default_factory=list)
    qa: QAResult
