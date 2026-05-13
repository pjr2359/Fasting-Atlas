from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from fasting_atlas.council import run_council
from fasting_atlas.consensus import build_consensus_methods, build_consensus_narrative
from fasting_atlas.eval_harness import run_eval
from fasting_atlas.extractors import (
    extract_metadata_llm,
    extract_methods_items,
    extract_narrative_results_items,
)
from fasting_atlas.figures import extract_figures
from fasting_atlas.llm_client import DEFAULT_CLAUDE_MODEL, ClaudeClient, JsonLLM, LLMError, OllamaClient
from fasting_atlas.output_writer import build_paper_id, write_paper_output
from fasting_atlas.pdf_ingest import PageText, ingest_pdf
from fasting_atlas.schemas import ConsensusResult, FigureExtracted, PaperExtraction, QAResult
from fasting_atlas.sections import find_methods_and_results_sections
from fasting_atlas.tables import extract_tables

# Project root: .../fasting_atlas/cli.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Load ANTHROPIC_API_KEY (etc.) from .env in the project root; does not override already-set env vars.
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(prog="fasting-atlas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse PDF papers into traceable JSON (Ollama or Anthropic Claude).",
    )
    parse_parser.add_argument("--input", default="papers", help="Input folder with PDFs.")
    parse_parser.add_argument("--output", default="parsed", help="Output folder for JSON files.")
    parse_parser.add_argument(
        "--llm-backend",
        choices=("ollama", "claude"),
        default="ollama",
        help="LLM provider. Claude uses ANTHROPIC_API_KEY from the environment or project .env file.",
    )
    parse_parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parse_parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id for the active backend. Defaults: llama3.2:1b (Ollama), "
            f"{DEFAULT_CLAUDE_MODEL} (Claude)."
        ),
    )
    parse_parser.add_argument(
        "--llm-timeout",
        type=int,
        default=300,
        help="HTTP read timeout seconds per LLM request (default 300).",
    )
    parse_parser.add_argument(
        "--ocr",
        choices=("auto", "off", "force"),
        default="off",
        help="OCR merge for low-text pages (requires Tesseract). Default off until environment is verified.",
    )
    parse_parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract languages, e.g. eng+deu+rus",
    )
    parse_parser.add_argument("--ocr-dpi", type=float, default=200.0, help="Render DPI for OCR (default 200).")
    parse_parser.add_argument(
        "--council-mode",
        choices=("structural", "full"),
        default="full",
        help="Council: structural JSON diff only, or alignment + semantic adjudication.",
    )
    parse_parser.add_argument(
        "--figures",
        choices=("off", "a", "b"),
        default="off",
        help="Figure extraction: off | tier-a (inventory+PNG) | tier-b (+ optional digitization).",
    )
    parse_parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root folder for figure PNGs and calibration sidecars.",
    )
    parse_parser.add_argument(
        "--figures-digitize",
        action="store_true",
        help="With --figures b, run calibration mapping and bounded CV sampling.",
    )
    parse_parser.add_argument(
        "--figures-calibration-dir",
        default=None,
        help="Folder containing <figure_id>_calibration.json (defaults to per-paper artifact dir).",
    )
    parse_parser.add_argument(
        "--consensus-samples",
        type=int,
        default=2,
        help="Number of independent extraction samples used for consensus-style output.",
    )
    parse_parser.add_argument(
        "--consensus-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for consensus extraction passes (higher = more diverse outputs).",
    )
    parse_parser.add_argument("--debug", action="store_true", help="Print per-stage progress and timings.")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run labeled checks from a gold CSV against parsed JSON outputs.",
    )
    eval_parser.add_argument("--gold", required=True, help="Path to gold CSV (see scripts/eval/gold_template.csv).")
    eval_parser.add_argument("--parsed", required=True, help="Directory containing <paper_id>.json files.")
    eval_parser.add_argument("--out", default=None, help="Optional path to write eval JSON report.")

    args = parser.parse_args()
    if args.command == "parse":
        model = args.model or (
            DEFAULT_CLAUDE_MODEL if args.llm_backend == "claude" else "llama3.2:1b"
        )
        parse_command(
            args.input,
            args.output,
            llm_backend=args.llm_backend,
            ollama_url=args.ollama_url,
            model=model,
            llm_timeout=args.llm_timeout,
            ocr_mode=args.ocr,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
            council_mode=args.council_mode,
            figures_tier=args.figures,
            artifacts_dir=args.artifacts_dir,
            figures_digitize=args.figures_digitize,
            figures_calibration_dir=args.figures_calibration_dir,
            consensus_samples=args.consensus_samples,
            consensus_temperature=args.consensus_temperature,
            debug=args.debug,
        )
    elif args.command == "eval":
        gold = Path(args.gold)
        parsed = Path(args.parsed)
        if not gold.is_file():
            print(f"Gold CSV not found: {gold}", flush=True)
            sys.exit(1)
        if not parsed.is_dir():
            print(f"Parsed directory not found: {parsed}", flush=True)
            sys.exit(1)
        report = run_eval(gold, parsed)
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}", flush=True)
        else:
            print(text, flush=True)
        sys.exit(0 if report.get("failed", 0) == 0 else 2)


def _pages_preview(pages: list[PageText], max_pages: int = 2) -> str:
    parts = [p.text for p in pages[:max_pages]]
    return "\n\n".join(parts)


def _make_llm_clients(
    llm_backend: str,
    ollama_url: str,
    model: str,
    llm_timeout: int,
    debug: bool,
) -> tuple[JsonLLM, JsonLLM, str]:
    """Returns (client_pass1, client_pass2, backend_label_for_qa)."""
    if llm_backend == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMError(
                "Claude backend requires ANTHROPIC_API_KEY (export it or put it in a .env file in the project root)."
            )
        c1 = ClaudeClient(api_key=key, model=model, timeout_seconds=llm_timeout, debug=debug)
        c2 = ClaudeClient(api_key=key, model=model, timeout_seconds=llm_timeout, debug=debug)
        return c1, c2, f"anthropic:{model}"

    c1 = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=llm_timeout, debug=debug)
    c2 = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=llm_timeout, debug=debug)
    return c1, c2, f"ollama:{model}"


def parse_command(
    input_dir: str,
    output_dir: str,
    llm_backend: str,
    ollama_url: str,
    model: str,
    llm_timeout: int,
    ocr_mode: str,
    ocr_lang: str,
    ocr_dpi: float,
    council_mode: str,
    figures_tier: str,
    artifacts_dir: str,
    figures_digitize: bool,
    figures_calibration_dir: str | None,
    consensus_samples: int,
    consensus_temperature: float,
    debug: bool,
) -> None:
    def log(message: str) -> None:
        print(message, flush=True)

    try:
        llm_client, pass_2_client, backend_label = _make_llm_clients(
            llm_backend, ollama_url, model, llm_timeout, debug
        )
    except LLMError as exc:
        log(f"ERROR configuring LLM: {exc}")
        sys.exit(1)

    input_path = Path(input_dir)
    pdf_files = sorted(input_path.rglob("*.pdf"))
    if consensus_samples < 1:
        log("ERROR consensus_samples must be at least 1.")
        sys.exit(1)
    if not pdf_files:
        log(f"No PDF files found in {input_path}.")
        return

    log(f"Found {len(pdf_files)} PDFs under {input_path} (backend={backend_label})")

    for index, pdf_file in enumerate(pdf_files, start=1):
        paper_started = time.perf_counter()
        log(f"[{index}/{len(pdf_files)}] Start {pdf_file.name}")

        try:
            t0 = time.perf_counter()
            ingested = ingest_pdf(
                str(pdf_file),
                ocr_mode=ocr_mode,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
            )
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

            log(f"[{pdf_file.name}] methods/results consensus sampling start")
            t0 = time.perf_counter()
            methods_passes = [
                extract_methods_items(
                    ingested.source_file,
                    methods_section,
                    llm_client,
                    temperature=consensus_temperature,
                )
                for _ in range(consensus_samples)
            ]
            results_passes = [
                extract_narrative_results_items(
                    ingested.source_file,
                    results_section,
                    llm_client,
                    temperature=consensus_temperature,
                )
                for _ in range(consensus_samples)
            ]
            if debug:
                log(
                    f"[{pdf_file.name}] consensus sampling done passes={consensus_samples} "
                    f"elapsed={time.perf_counter() - t0:.2f}s"
                )

            methods_consensus, method_clusters, avg_method_support = build_consensus_methods(
                methods_passes
            )
            narrative_consensus, narrative_clusters, avg_narrative_support = build_consensus_narrative(
                results_passes
            )
            if debug:
                log(
                    f"[{pdf_file.name}] consensus built methods={len(methods_consensus)} "
                    f"clusters={method_clusters} avg_support={avg_method_support:.2f}; "
                    f"narrative={len(narrative_consensus)} clusters={narrative_clusters} "
                    f"avg_support={avg_narrative_support:.2f}"
                )

            methods_pass_1 = methods_passes[0] if methods_passes else []
            results_pass_1 = results_passes[0] if results_passes else []
            methods_pass_2 = methods_passes[1] if len(methods_passes) > 1 else methods_pass_1
            results_pass_2 = results_passes[1] if len(results_passes) > 1 else results_pass_1

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

            adjudication_llm: JsonLLM | None = llm_client if council_mode == "full" else None
            council = run_council(
                pass_1_model=backend_label,
                pass_2_model=f"{backend_label}-alt",
                pass_1_payload=payload_1,
                pass_2_payload=payload_2,
                mode=council_mode,
                adjudication_llm=adjudication_llm,
            )
            if debug:
                log(
                    f"[{pdf_file.name}] council done discrepancies={len(council.discrepancies)} "
                    f"needs_review={council.needs_human_review} "
                    f"aligned_methods={len(council.aligned_method_pairs)} "
                    f"semantic={len(council.semantic_groups)}"
                )

            paper_id = build_paper_id(ingested.source_file)
            figures_list: list[FigureExtracted] = []
            if figures_tier != "off":
                cal_dir = Path(figures_calibration_dir) if figures_calibration_dir else None
                figures_list = extract_figures(
                    str(pdf_file),
                    paper_id,
                    artifacts_dir,
                    tier=figures_tier,
                    calibration_dir=cal_dir,
                    digitize_flag=figures_digitize,
                )

            consensus_summary = ConsensusResult(
                samples=consensus_samples,
                method_clusters=method_clusters,
                narrative_clusters=narrative_clusters,
                average_method_support=avg_method_support,
                average_narrative_support=avg_narrative_support,
            )
            extraction = PaperExtraction(
                paper_id=paper_id,
                source_file=ingested.source_file,
                metadata=metadata,
                methods_participants=methods_consensus,
                narrative_results=narrative_consensus,
                tables=tables,
                figures_graphs=figures_list,
                qa=QAResult(council=council, consensus=consensus_summary),
            )
            out_file = write_paper_output(output_dir, extraction)
            log(f"[{index}/{len(pdf_files)}] Wrote {out_file} elapsed_total={time.perf_counter() - paper_started:.2f}s")

        except LLMError as exc:
            log(f"ERROR [{pdf_file.name}] LLM failed: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
