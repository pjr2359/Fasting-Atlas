from __future__ import annotations

from collections import Counter
from typing import Any

from fasting_atlas.council_align import overlap_score
from fasting_atlas.schemas import (
    EvidenceText,
    MethodsParticipantsItem,
    NarrativeResultsItem,
)


def _normalize_value(value: Any) -> str:
    return str(value or "").strip()


def _most_common_candidate(values: list[str]) -> tuple[str, int]:
    counter = Counter(values)
    if not counter:
        return "", 0
    return max(counter.items(), key=lambda item: (item[1], len(item[0])))


def _cluster_items(all_items: list[dict[str, Any]], narrative: bool, min_score: float = 0.12) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for item in all_items:
        best_cluster: list[dict[str, Any]] | None = None
        best_score = 0.0
        for cluster in clusters:
            score = overlap_score(item, cluster[0], narrative=narrative)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is not None and best_score >= min_score:
            best_cluster.append(item)
        else:
            clusters.append([item])
    return clusters


def _choose_evidence(cluster: list[dict[str, Any]], consensus_fields: dict[str, str]) -> list[dict[str, Any]]:
    if not cluster:
        return []

    for item in cluster:
        if all(_normalize_value(item.get(key)) == _normalize_value(value) for key, value in consensus_fields.items()):
            return item.get("evidence", []) or []

    return cluster[0].get("evidence", []) or []


def _build_consensus_cluster(cluster: list[dict[str, Any]], narrative: bool) -> tuple[MethodsParticipantsItem | NarrativeResultsItem, float]:
    if narrative:
        fields = ["statement", "relationship"]
        item_type = NarrativeResultsItem
    else:
        fields = ["label", "value", "method", "unit"]
        item_type = MethodsParticipantsItem

    consensus_values: dict[str, str | None] = {}
    support_ratios: list[float] = []
    for field in fields:
        values = [_normalize_value(item.get(field)) for item in cluster]
        best_value, support = _most_common_candidate(values)
        consensus_values[field] = best_value or None
        support_ratios.append(float(support) / len(cluster) if cluster else 0.0)

    confidence = sum(support_ratios) / len(support_ratios) if support_ratios else 0.0
    evidence_raw = _choose_evidence(cluster, {field: consensus_values[field] or "" for field in fields})
    evidence = [EvidenceText(**ev) for ev in evidence_raw if isinstance(ev, dict)]

    if narrative:
        return (
            NarrativeResultsItem(
                statement=consensus_values["statement"] or "",
                relationship=consensus_values["relationship"],
                confidence=confidence,
                extraction_method="llm-consensus",
                evidence=evidence,
            ),
            confidence,
        )

    return (
        MethodsParticipantsItem(
            label=consensus_values["label"] or "",
            value=consensus_values["value"] or "",
            method=consensus_values["method"],
            unit=consensus_values["unit"],
            confidence=confidence,
            extraction_method="llm-consensus",
            evidence=evidence,
        ),
        confidence,
    )


def _build_consensus_items(
    all_passes: list[list[MethodsParticipantsItem] | list[NarrativeResultsItem]],
    narrative: bool,
    min_score: float = 0.12,
) -> tuple[list[MethodsParticipantsItem] | list[NarrativeResultsItem], int, float]:
    flat_items: list[dict[str, Any]] = []
    for sample in all_passes:
        for item in sample:
            flat_items.append(item.model_dump())

    clusters = _cluster_items(flat_items, narrative=narrative, min_score=min_score)
    consensus_items = []
    confidences: list[float] = []
    for cluster in clusters:
        item, confidence = _build_consensus_cluster(cluster, narrative=narrative)
        consensus_items.append(item)
        confidences.append(confidence)

    average_support = sum(confidences) / len(confidences) if confidences else 0.0
    return consensus_items, len(clusters), average_support


def build_consensus_methods(
    all_passes: list[list[MethodsParticipantsItem]],
    min_score: float = 0.12,
) -> tuple[list[MethodsParticipantsItem], int, float]:
    return _build_consensus_items(all_passes, narrative=False, min_score=min_score)


def build_consensus_narrative(
    all_passes: list[list[NarrativeResultsItem]],
    min_score: float = 0.12,
) -> tuple[list[NarrativeResultsItem], int, float]:
    result = _build_consensus_items(all_passes, narrative=True, min_score=min_score)
    return result  # type: ignore[return-value]
