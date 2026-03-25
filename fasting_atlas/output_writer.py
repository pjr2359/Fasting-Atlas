from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fasting_atlas.schemas import PaperExtraction


def build_paper_id(source_file: str) -> str:
    digest = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:12]
    stem = Path(source_file).stem.lower().replace(" ", "-")
    return f"{stem}-{digest}"


def write_paper_output(output_dir: str, extraction: PaperExtraction) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / f"{extraction.paper_id}.json"
    target.write_text(json.dumps(extraction.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)
