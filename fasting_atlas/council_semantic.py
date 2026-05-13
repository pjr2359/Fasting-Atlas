"""Batched LLM semantic adjudication for Council aligned pairs."""

from __future__ import annotations

import json
from typing import Any, Literal

from fasting_atlas.llm_client import JsonLLM
from fasting_atlas.schemas import CouncilAlignedPair, SemanticCouncilGroup

CouncilDomain = Literal["methods_participants", "narrative_results"]

_ADJ_SCHEMA = """{
  "judgments": [
    {
      "aligned_pair_index": 0,
      "verdict": "equivalent|minor_wording|true_conflict|unpaired",
      "severity": "low|medium|high",
      "rationale": "short string",
      "recommended_action": "accept_pass_1|accept_pass_2|merge|human_review|no_action"
    }
  ]
}"""


def _summarize_methods(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    parts = [
        str(item.get("label", "")),
        str(item.get("value", "")),
        str(item.get("method") or ""),
        str(item.get("unit") or ""),
    ]
    return " | ".join(p for p in parts if p)


def _summarize_narrative(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    return str(item.get("statement", "")) + (" ; " + str(item.get("relationship")) if item.get("relationship") else "")


def adjudicate_domain(
    domain: CouncilDomain,
    aligned_pairs: list[CouncilAlignedPair],
    pass_1_items: list[dict[str, Any]],
    pass_2_items: list[dict[str, Any]],
    llm: JsonLLM,
    *,
    batch_size: int = 8,
    temperature: float = 0.0,
) -> list[SemanticCouncilGroup]:
    """Run LLM on batches of aligned pairs; returns one SemanticCouncilGroup per aligned row."""
    if not aligned_pairs:
        return []

    out: list[SemanticCouncilGroup] = []
    narrative = domain == "narrative_results"

    for start in range(0, len(aligned_pairs), batch_size):
        batch = aligned_pairs[start : start + batch_size]
        lines: list[str] = []
        for bi, ap in enumerate(batch):
            global_idx = start + bi
            i1, i2 = ap.pass_1_index, ap.pass_2_index
            d1 = pass_1_items[i1] if i1 is not None and i1 < len(pass_1_items) else None
            d2 = pass_2_items[i2] if i2 is not None and i2 < len(pass_2_items) else None
            if narrative:
                s1, s2 = _summarize_narrative(d1), _summarize_narrative(d2)
            else:
                s1, s2 = _summarize_methods(d1), _summarize_methods(d2)
            lines.append(
                json.dumps(
                    {
                        "aligned_pair_index": global_idx,
                        "alignment_score": ap.alignment_score,
                        "pass_1_index": i1,
                        "pass_2_index": i2,
                        "pass_1": s1[:1200],
                        "pass_2": s2[:1200],
                    },
                    ensure_ascii=False,
                )
            )

        prompt = (
            "You compare two independent extractions from the same scientific paper.\n"
            "For each object, decide if they describe the same underlying fact.\n"
            "Verdict meanings:\n"
            "- equivalent: same fact, negligible wording differences\n"
            "- minor_wording: same fact, different phrasing or rounding\n"
            "- true_conflict: incompatible facts or values\n"
            "- unpaired: only one side present (orphan)\n"
            "Rules: be conservative; use true_conflict only for real contradictions.\n\n"
            "Batch (JSON lines):\n"
            + "\n".join(lines)
        )

        raw = llm.extract_json(prompt, _ADJ_SCHEMA, temperature=temperature, retries=2, num_predict=2048)
        judgments = raw.get("judgments") if isinstance(raw, dict) else None
        if not isinstance(judgments, list):
            judgments = []

        by_idx: dict[int, dict[str, Any]] = {}
        for j in judgments:
            if not isinstance(j, dict):
                continue
            idx = j.get("aligned_pair_index")
            if isinstance(idx, int):
                by_idx[idx] = j

        for bi, ap in enumerate(batch):
            global_idx = start + bi
            j = by_idx.get(global_idx, {})
            verdict = str(j.get("verdict", "minor_wording")).lower()
            if verdict not in ("equivalent", "minor_wording", "true_conflict", "unpaired"):
                verdict = "minor_wording"
            sev = str(j.get("severity", "medium")).lower()
            if sev not in ("low", "medium", "high"):
                sev = "medium"
            action = str(j.get("recommended_action", "no_action")).lower()
            if action not in ("accept_pass_1", "accept_pass_2", "merge", "human_review", "no_action"):
                action = "no_action"
            if ap.pass_1_index is None or ap.pass_2_index is None:
                verdict = "unpaired"
                if action == "no_action":
                    action = "human_review"
            out.append(
                SemanticCouncilGroup(
                    domain=domain,
                    aligned_pair_index=global_idx,
                    verdict=verdict,  # type: ignore[arg-type]
                    severity=sev,  # type: ignore[arg-type]
                    rationale=str(j.get("rationale", ""))[:2000],
                    recommended_action=action,  # type: ignore[arg-type]
                )
            )

    return out
