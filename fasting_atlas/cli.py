from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fasting_atlas.council import run_council
from fasting_atlas.extractors import (
    extract_metadata_llm,
    extract_methods_items,
    extract_narrative_results_items,
)
from fasting_atlas.llm_client import LLMError, OllamaClient
from fasting_atlas.output_writer import build_paper_id, write_paper_output
from fasting_atlas.pdf_ingest import PageText, ingest_pdf
from fasting_atlas.schemas import PaperExtraction, QAResult
from fasting_atlas.sections import find_methods_and_results_sections
from fasting_atlas.tables import extract_tables


def main() -> None:
    parser = argparse.ArgumentParser(prog="fasting-atlas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse PDF papers into traceable JSON (requires Ollama).")
    parse_parser.add_argument("--input", default="papers", help="Input folder with PDFs.")
    parse_parser.add_argument("--output", default="parsed", help="Output folder for JSON files.")
    parse_parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parse_parser.add_argument("--model", default="mistral:7b", help="Ollama model name.")
    parse_parser.add_argument("--llm-timeout", type=int, default=120, help="Seconds to wait per LLM request.")
    parse_parser.add_argument("--debug", action="store_true", help="Print per-stage progress and timings.")

    args = parser.parse_args()
    if args.command == "parse":
        parse_command(
            args.input,
            args.output,
            args.ollama_url,
            args.model,
            args.llm_timeout,
            args.debug,
        )


def _pages_preview(pages: list[PageText], max_pages: int = 2) -> str:
    parts = [p.text for p in pages[:max_pages]]
    return "\n\n".join(parts)


def parse_command(
    input_dir: str,
    output_dir: str,
    ollama_url: str,
    model: str,
    llm_timeout: int,
    debug: bool,
) -> None:
    def log(message: str) -> None:
        print(message, flush=True)

    input_path = Path(input_dir)
    pdf_files = sorted(input_path.rglob("*.pdf"))
    if not pdf_files:
        log(f"No PDF files found in {input_path}.")
        return

    llm_client = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=llm_timeout, debug=debug)
    pass_2_client = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=llm_timeout, debug=debug)

    log(f"Found {len(pdf_files)} PDFs under {input_path}")

    for index, pdf_file in enumerate(pdf_files, start=1):
        paper_started = time.perf_counter()
        log(f"[{index}/{len(pdf_files)}] Start {pdf_file.name}")

        try:
            t0 = time.perf_counter()
            ingested = ingest_pdf(str(pdf_file))
            if debug:
                log(
                    f"[{pdf_file.name}] ingest done pages={len(ingested.pages)} tables={len(ingested.tables)} "
                    f"elapsed={time.perf_counter() - t0:.2f}s"
                )

            t0 = time.perf_counter()
            methods_section, results_section = find_methods_and_results_sections(ingested.pages)
            if debug:
                log(
                    f"[{pdf_file.name}] sectioning done methods_pages={methods_section.pages} "
                    f"results_pages={results_section.pages} elapsed={time.perf_counter() - t0:.2f}s"
                )

            t0 = time.perf_counter()
            preview = _pages_preview(ingested.pages)
            metadata = extract_metadata_llm(ingested.source_file, preview, llm_client)
            if debug:
                log(f"[{pdf_file.name}] metadata (LLM) done elapsed={time.perf_counter() - t0:.2f}s")

            log(f"[{pdf_file.name}] methods/results pass1 start")
            t0 = time.perf_counter()
            methods_pass_1 = extract_methods_items(ingested.source_file, methods_section, llm_client, temperature=0.1)
            results_pass_1 = extract_narrative_results_items(
                ingested.source_file, results_section, llm_client, temperature=0.1
            )
            if debug:
                log(
                    f"[{pdf_file.name}] methods/results pass1 done items=({len(methods_pass_1)}, {len(results_pass_1)}) "
                    f"elapsed={time.perf_counter() - t0:.2f}s"
                )

            log(f"[{pdf_file.name}] methods/results pass2 start")
            t0 = time.perf_counter()
            methods_pass_2 = extract_methods_items(ingested.source_file, methods_section, pass_2_client, temperature=0.6)
            results_pass_2 = extract_narrative_results_items(
                ingested.source_file, results_section, pass_2_client, temperature=0.6
            )
            if debug:
                log(
                    f"[{pdf_file.name}] methods/results pass2 done items=({len(methods_pass_2)}, {len(results_pass_2)}) "
                    f"elapsed={time.perf_counter() - t0:.2f}s"
                )

            log(f"[{pdf_file.name}] table extraction start")
            t0 = time.perf_counter()
            tables = extract_tables(ingested.source_file, ingested.tables, llm_client)
            if debug:
                log(f"[{pdf_file.name}] table extraction done tables={len(tables)} elapsed={time.perf_counter() - t0:.2f}s")

            payload_1 = {
                "methods_participants": [item.model_dump() for item in methods_pass_1],
                "narrative_results": [item.model_dump() for item in results_pass_1],
            }
            payload_2 = {
                "methods_participants": [item.model_dump() for item in methods_pass_2],
                "narrative_results": [item.model_dump() for item in results_pass_2],
            }

            council = run_council(
                pass_1_model=model,
                pass_2_model=f"{model}-alt",
                pass_1_payload=payload_1,
                pass_2_payload=payload_2,
            )
            if debug:
                log(
                    f"[{pdf_file.name}] council done discrepancies={len(council.discrepancies)} "
                    f"needs_review={council.needs_human_review}"
                )

            paper_id = build_paper_id(ingested.source_file)
            extraction = PaperExtraction(
                paper_id=paper_id,
                source_file=ingested.source_file,
                metadata=metadata,
                methods_participants=methods_pass_1,
                narrative_results=results_pass_1,
                tables=tables,
                figures_graphs=[],
                qa=QAResult(council=council),
            )
            out_file = write_paper_output(output_dir, extraction)
            log(f"[{index}/{len(pdf_files)}] Wrote {out_file} elapsed_total={time.perf_counter() - paper_started:.2f}s")

        except LLMError as exc:
            log(f"ERROR [{pdf_file.name}] LLM failed: {exc}")
            sys.exit(1)
