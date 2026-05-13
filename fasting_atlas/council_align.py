"""Align extraction items between two passes using normalized keys and token overlap."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", s).lower()
    t = re.sub(r"[_\-\s]+", " ", t)
    t = re.sub(r"r\s*a|ra\b|rate of appearance", " ra ", t)
    return " ".join(t.split())


def token_set(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_text(s)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _item_text(item: dict[str, Any], *, narrative: bool) -> str:
    parts: list[str] = []
    if narrative:
        parts.append(str(item.get("statement", "")))
        parts.append(str(item.get("relationship") or ""))
    else:
        parts.append(str(item.get("label", "")))
        parts.append(str(item.get("value", "")))
        parts.append(str(item.get("method") or ""))
        parts.append(str(item.get("unit") or ""))
    ev = item.get("evidence") or []
    if isinstance(ev, list) and ev:
        q = ev[0].get("quote") if isinstance(ev[0], dict) else None
        if q:
            parts.append(str(q))
    return " ".join(parts)


def overlap_score(a: dict[str, Any], b: dict[str, Any], *, narrative: bool) -> float:
    ta = token_set(_item_text(a, narrative=narrative))
    tb = token_set(_item_text(b, narrative=narrative))
    return jaccard(ta, tb)


@dataclass
class AlignedPairRecord:
    pass_1_index: int | None
    pass_2_index: int | None
    score: float
    pass_1_item: dict[str, Any] | None
    pass_2_item: dict[str, Any] | None


def align_item_lists(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    narrative: bool,
    min_score: float = 0.12,
) -> list[AlignedPairRecord]:
    """Greedy matching: each right used at most once; best score wins per left."""
    used_r: set[int] = set()
    pairs: list[AlignedPairRecord] = []

    for i, a in enumerate(left):
        best_j = -1
        best_s = -1.0
        for j, b in enumerate(right):
            if j in used_r:
                continue
            s = overlap_score(a, b, narrative=narrative)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j >= 0 and best_s >= min_score:
            used_r.add(best_j)
            pairs.append(
                AlignedPairRecord(
                    pass_1_index=i,
                    pass_2_index=best_j,
                    score=best_s,
                    pass_1_item=a,
                    pass_2_item=right[best_j],
                )
            )
        else:
            pairs.append(
                AlignedPairRecord(
                    pass_1_index=i,
                    pass_2_index=None,
                    score=0.0,
                    pass_1_item=a,
                    pass_2_item=None,
                )
            )

    matched_r = {p.pass_2_index for p in pairs if p.pass_2_index is not None}
    for j, b in enumerate(right):
        if j not in matched_r:
            pairs.append(
                AlignedPairRecord(
                    pass_1_index=None,
                    pass_2_index=j,
                    score=0.0,
                    pass_1_item=None,
                    pass_2_item=b,
                )
            )

    pairs.sort(key=lambda p: (p.pass_1_index is None, p.pass_1_index or 9999, p.pass_2_index or 9999))
    return pairs
