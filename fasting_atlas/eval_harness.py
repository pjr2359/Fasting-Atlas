"""Repeatable evaluation against a labeled CSV (course / QA harness)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def run_eval(gold_csv: Path, parsed_dir: Path) -> dict[str, Any]:
    """
    CSV columns: ``paper_id``, ``check``, ``arg1``, ``arg2`` (arg2 optional).

    Supported ``check`` values:
    - ``metadata_title_contains`` — arg1 substring in ``metadata.title``
    - ``methods_count_ge`` — int(arg1) <= len(methods_participants)
    - ``narrative_count_ge`` — int(arg1) <= len(narrative_results)
    - ``tables_count_ge`` — int(arg1) <= len(tables)
    - ``json_path_equals`` — arg1 dotted path (e.g. qa.council.needs_human_review), arg2 expected JSON value
    """
    rows: list[dict[str, str]]
    with gold_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for row in rows:
        paper_id = (row.get("paper_id") or "").strip()
        check = (row.get("check") or "").strip()
        arg1 = (row.get("arg1") or "").strip()
        arg2 = (row.get("arg2") or "").strip()

        if not paper_id or not check:
            continue

        json_path = parsed_dir / f"{paper_id}.json"
        if not json_path.is_file():
            results.append({"paper_id": paper_id, "check": check, "ok": False, "detail": f"missing {json_path}"})
            failed += 1
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        ok, detail = _run_one_check(data, check, arg1, arg2)
        results.append({"paper_id": paper_id, "check": check, "ok": ok, "detail": detail})
        if ok:
            passed += 1
        else:
            failed += 1

    return {
        "gold_csv": str(gold_csv),
        "parsed_dir": str(parsed_dir),
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "results": results,
    }


def _run_one_check(data: dict[str, Any], check: str, arg1: str, arg2: str) -> tuple[bool, str]:
    if check == "metadata_title_contains":
        title = (data.get("metadata") or {}).get("title") or ""
        ok = arg1.lower() in str(title).lower()
        return ok, f"title={title!r}"

    if check == "methods_count_ge":
        n = len(data.get("methods_participants") or [])
        need = int(arg1)
        ok = n >= need
        return ok, f"count={n} need>={need}"

    if check == "narrative_count_ge":
        n = len(data.get("narrative_results") or [])
        need = int(arg1)
        ok = n >= need
        return ok, f"count={n} need>={need}"

    if check == "tables_count_ge":
        n = len(data.get("tables") or [])
        need = int(arg1)
        ok = n >= need
        return ok, f"count={n} need>={need}"

    if check == "json_path_equals":
        cur: Any = data
        for part in arg1.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and part.isdigit():
                cur = cur[int(part)]
            else:
                return False, f"path missing at {part!r}"
        expected = _parse_arg2_value(arg2)
        ok = cur == expected
        return ok, f"got={cur!r} expected={expected!r}"

    return False, f"unknown check {check!r}"


def _parse_arg2_value(arg2: str) -> Any:
    lowered = arg2.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(arg2)
    except ValueError:
        pass
    try:
        return float(arg2)
    except ValueError:
        pass
    if (arg2.startswith('"') and arg2.endswith('"')) or (arg2.startswith("'") and arg2.endswith("'")):
        return arg2[1:-1]
    return arg2
