from __future__ import annotations

from typing import Any, Literal

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


class CouncilAlignedPair(BaseModel):
    """One alignment between pass-1 and pass-2 list items (or unpaired orphan)."""

    domain: Literal["methods_participants", "narrative_results"]
    pass_1_index: int | None = None
    pass_2_index: int | None = None
    alignment_score: float = 0.0
    pass_1_summary: str | None = None
    pass_2_summary: str | None = None


class SemanticCouncilGroup(BaseModel):
    """LLM adjudication for one aligned pair or unpaired item."""

    domain: Literal["methods_participants", "narrative_results"]
    aligned_pair_index: int
    verdict: Literal["equivalent", "minor_wording", "true_conflict", "unpaired"]
    severity: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""
    recommended_action: Literal[
        "accept_pass_1", "accept_pass_2", "merge", "human_review", "no_action"
    ] = "no_action"


class CouncilResult(BaseModel):
    pass_1_model: str
    pass_2_model: str
    # Legacy / human-facing: structural diffs in structural mode; semantic conflicts in full mode.
    discrepancies: list[CouncilDiscrepancy] = Field(default_factory=list)
    needs_human_review: bool = False
    # Full council: raw structural JSON diff (debug telemetry).
    structural_discrepancies: list[CouncilDiscrepancy] | None = None
    aligned_method_pairs: list[CouncilAlignedPair] = Field(default_factory=list)
    aligned_narrative_pairs: list[CouncilAlignedPair] = Field(default_factory=list)
    semantic_groups: list[SemanticCouncilGroup] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    samples: int = 0
    method_clusters: int = 0
    narrative_clusters: int = 0
    average_method_support: float = 0.0
    average_narrative_support: float = 0.0


class QAResult(BaseModel):
    council: CouncilResult
    consensus: ConsensusResult | None = None


class FigureEvidence(BaseModel):
    source_file: str
    page: int
    quote: str | None = None
    bbox: list[float] | None = None


class FigureExtracted(BaseModel):
    figure_id: str
    page: int
    bbox: list[float] = Field(default_factory=list)
    caption: str | None = None
    image_sha256: str | None = None
    image_path: str | None = None
    extraction_method: str = "embedded_image"
    confidence: float = 0.0
    evidence: list[FigureEvidence] = Field(default_factory=list)
    # Tier B: optional digitized series in data space (after calibration).
    digitized_series: list[dict[str, Any]] = Field(default_factory=list)
    digitization_metadata: dict[str, Any] = Field(default_factory=dict)


class PaperExtraction(BaseModel):
    paper_id: str
    source_file: str
    metadata: Metadata
    methods_participants: list[MethodsParticipantsItem] = Field(default_factory=list)
    narrative_results: list[NarrativeResultsItem] = Field(default_factory=list)
    tables: list[TableExtracted] = Field(default_factory=list)
    figures_graphs: list[FigureExtracted] = Field(default_factory=list)
    qa: QAResult
