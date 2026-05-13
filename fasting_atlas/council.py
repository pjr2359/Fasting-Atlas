from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fasting_atlas.council_align import align_item_lists
from fasting_atlas.council_semantic import adjudicate_domain
from fasting_atlas.llm_client import JsonLLM
from fasting_atlas.schemas import CouncilAlignedPair, CouncilDiscrepancy, CouncilResult, SemanticCouncilGroup


def run_council(
    pass_1_model: str,
    pass_2_model: str,
    pass_1_payload: dict[str, Any],
    pass_2_payload: dict[str, Any],
    *,
    mode: str = "full",
    adjudication_llm: JsonLLM | None = None,
) -> CouncilResult:
    structural = diff_payloads(pass_1_payload, pass_2_payload)
    if mode not in ("structural", "full"):
        mode = "full"

    if mode == "structural":
        return CouncilResult(
            pass_1_model=pass_1_model,
            pass_2_model=pass_2_model,
            discrepancies=structural,
            needs_human_review=len(structural) > 0,
            structural_discrepancies=None,
            aligned_method_pairs=[],
            aligned_narrative_pairs=[],
            semantic_groups=[],
        )

    m_left = list(pass_1_payload.get("methods_participants") or [])
    m_right = list(pass_2_payload.get("methods_participants") or [])
    n_left = list(pass_1_payload.get("narrative_results") or [])
    n_right = list(pass_2_payload.get("narrative_results") or [])

    method_aligned = align_item_lists(m_left, m_right, narrative=False)
    narr_aligned = align_item_lists(n_left, n_right, narrative=True)

    aligned_method_pairs: list[CouncilAlignedPair] = []
    for rec in method_aligned:
        s1 = _brief_methods(rec.pass_1_item)
        s2 = _brief_methods(rec.pass_2_item)
        aligned_method_pairs.append(
            CouncilAlignedPair(
                domain="methods_participants",
                pass_1_index=rec.pass_1_index,
                pass_2_index=rec.pass_2_index,
                alignment_score=rec.score,
                pass_1_summary=s1,
                pass_2_summary=s2,
            )
        )

    aligned_narrative_pairs: list[CouncilAlignedPair] = []
    for rec in narr_aligned:
        aligned_narrative_pairs.append(
            CouncilAlignedPair(
                domain="narrative_results",
                pass_1_index=rec.pass_1_index,
                pass_2_index=rec.pass_2_index,
                alignment_score=rec.score,
                pass_1_summary=_brief_narrative(rec.pass_1_item),
                pass_2_summary=_brief_narrative(rec.pass_2_item),
            )
        )

    if adjudication_llm is None:
        return CouncilResult(
            pass_1_model=pass_1_model,
            pass_2_model=pass_2_model,
            discrepancies=structural,
            needs_human_review=len(structural) > 0,
            structural_discrepancies=structural,
            aligned_method_pairs=aligned_method_pairs,
            aligned_narrative_pairs=aligned_narrative_pairs,
            semantic_groups=[],
        )

    semantic_groups: list[SemanticCouncilGroup] = []
    semantic_groups.extend(
        adjudicate_domain(
            "methods_participants",
            aligned_method_pairs,
            m_left,
            m_right,
            adjudication_llm,
        )
    )
    semantic_groups.extend(
        adjudicate_domain(
            "narrative_results",
            aligned_narrative_pairs,
            n_left,
            n_right,
            adjudication_llm,
        )
    )

    discrepancies = _semantic_to_discrepancies(semantic_groups)
    needs_human_review = any(g.verdict == "true_conflict" for g in semantic_groups)

    return CouncilResult(
        pass_1_model=pass_1_model,
        pass_2_model=pass_2_model,
        discrepancies=discrepancies,
        needs_human_review=needs_human_review,
        structural_discrepancies=structural,
        aligned_method_pairs=aligned_method_pairs,
        aligned_narrative_pairs=aligned_narrative_pairs,
        semantic_groups=semantic_groups,
    )


def _brief_methods(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None
    parts = [
        str(item.get("label", "")),
        str(item.get("value", "")),
        str(item.get("method") or ""),
        str(item.get("unit") or ""),
    ]
    return " | ".join(p for p in parts if p)[:800] or None


def _brief_narrative(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None
    s = str(item.get("statement", ""))
    r = item.get("relationship")
    if r:
        s += f" [{r}]"
    return s[:800] or None


def _semantic_to_discrepancies(groups: list[SemanticCouncilGroup]) -> list[CouncilDiscrepancy]:
    out: list[CouncilDiscrepancy] = []
    for g in groups:
        if g.verdict != "true_conflict":
            continue
        out.append(
            CouncilDiscrepancy(
                path=f"semantic.{g.domain}[pair={g.aligned_pair_index}]",
                pass_1_value=g.verdict,
                pass_2_value=g.rationale or g.recommended_action,
                severity="high",
            )
        )
    return out


def diff_payloads(left: dict[str, Any], right: dict[str, Any]) -> list[CouncilDiscrepancy]:
    diffs: list[CouncilDiscrepancy] = []
    _walk("", left, right, diffs.append)
    return diffs


def _walk(path: str, left: object, right: object, emit: Callable[[CouncilDiscrepancy], None]) -> None:
    if type(left) is not type(right):
        emit(CouncilDiscrepancy(path=path or "root", pass_1_value=left, pass_2_value=right, severity="high"))
        return

    if isinstance(left, dict):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        for key in sorted(left_keys | right_keys):
            next_path = f"{path}.{key}" if path else key
            if key not in left:
                emit(CouncilDiscrepancy(path=next_path, pass_1_value=None, pass_2_value=right.get(key), severity="medium"))
                continue
            if key not in right:
                emit(CouncilDiscrepancy(path=next_path, pass_1_value=left.get(key), pass_2_value=None, severity="medium"))
                continue
            _walk(next_path, left.get(key), right.get(key), emit)
        return

    if isinstance(left, list):
        if len(left) != len(right):
            emit(CouncilDiscrepancy(path=path, pass_1_value=f"len={len(left)}", pass_2_value=f"len={len(right)}", severity="medium"))
        for idx, (lv, rv) in enumerate(zip(left, right)):
            _walk(f"{path}[{idx}]", lv, rv, emit)
        return

    if left != right:
        emit(CouncilDiscrepancy(path=path, pass_1_value=left, pass_2_value=right, severity="low"))
