from __future__ import annotations

from collections.abc import Callable

from fasting_atlas.schemas import CouncilDiscrepancy, CouncilResult


def run_council(
    pass_1_model: str,
    pass_2_model: str,
    pass_1_payload: dict,
    pass_2_payload: dict,
) -> CouncilResult:
    discrepancies = diff_payloads(pass_1_payload, pass_2_payload)
    needs_review = len(discrepancies) > 0
    return CouncilResult(
        pass_1_model=pass_1_model,
        pass_2_model=pass_2_model,
        discrepancies=discrepancies,
        needs_human_review=needs_review,
    )


def diff_payloads(left: dict, right: dict) -> list[CouncilDiscrepancy]:
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
