# Fasting-Atlas

A PDF-to-JSON extraction pipeline for human experiment papers with traceable structured output.

## Research Basis

This project now incorporates a self-consistency-style consensus extraction approach inspired by:

- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (ICLR 2023)
- https://arxiv.org/abs/2203.11171

## Usage

Run the parser with a supported backend:

```bash
.venv/bin/python3 -m fasting_atlas.cli parse --input papers --output parsed --llm-backend claude --consensus-samples 2 --consensus-temperature 0.6
```

The output includes structured `methods_participants`, `narrative_results`, and QA telemetry in `qa.council` and `qa.consensus`.
