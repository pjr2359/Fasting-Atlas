## Part 2: Evaluation

We evaluated Fasting-Atlas as a PDF-to-JSON extraction system for human fasting studies. Our evaluation asked four questions:

1. **Reliability:** Can the system successfully parse the full corpus without crashing or producing invalid JSON?
2. **Extraction yield:** How much structured information does the system recover from each paper?
3. **Grounding:** Are extracted facts traceable back to quoted source evidence in the PDFs?
4. **Consensus design:** How do different self-consistency settings affect agreement, conflict rate, and runtime?

### Evaluation Setup

We ran the parser on the nine-paper fasting corpus in `papers/`. Each run used the Claude backend and produced structured JSON containing metadata, methods/participant facts, narrative results, extracted tables, and QA telemetry. We compared three consensus configurations:

| Configuration | Samples | Temperature | Completed papers |
|---|---:|---:|---:|
| Conservative | 3 | 0.4 | 9/9 |
| Moderate | 3 | 0.6 | 9/9 |
| Diverse | 5 | 0.8 | 9/9 |

All three configurations completed the corpus. The 5-sample diverse setting required resuming after API credit and malformed-response failures, so we treat those retries as part of the operational cost of using a higher-temperature, higher-sample configuration.

We assessed each output with automated checks for schema validity, title extraction, extraction counts, consensus support, semantic council conflicts, and evidence grounding. For grounding, we used two metrics. Exact quote match is a strict lower bound because PDF text extraction changes spacing and symbols. Token-overlap grounding is the more realistic automated check: an evidence quote is counted as grounded if at least 80% of its content tokens occur on the cited page, falling back to the full document when article page numbers differ from PDF page indices.

### Results

The conservative 3-sample setting was the best overall operating point. It completed all nine papers, produced schema-valid JSON for all outputs, extracted titles for all papers, and had the highest average consensus support.

| Metric | Conservative, 3 samples T=0.4 | Moderate, 3 samples T=0.6 | Diverse, 5 samples T=0.8 |
|---|---:|---:|---:|
| Successful parses | 9/9 | 9/9 | 9/9 |
| Schema-valid JSON | 9/9 | 9/9 | 9/9 |
| Titles extracted | 9/9 | 9/9 | 9/9 |
| Mean methods facts per paper | 12.1 | 11.6 | 13.0 |
| Mean narrative results per paper | 3.9 | 3.2 | 3.6 |
| Tables extracted | 10 total | 10 total | 10 total |
| Mean method consensus support | 0.638 | 0.608 | 0.558 |
| Mean narrative consensus support | 0.589 | 0.551 | 0.469 |
| Mean semantic conflicts per paper | 3.1 | 4.7 | 3.9 |
| Evidence exact-match rate | 20.1% | 20.7% | 23.1% |
| Evidence token-grounding rate | 65.0% | 65.9% | 64.5% |
| Mean evidence token overlap | 0.802 | 0.809 | 0.782 |
| Mean runtime per paper | 127s | 130s | about 195s |

The conservative configuration recovered 109 methods/participant facts and 35 narrative result facts across the nine papers, while preserving traceability through evidence quotes. It also had the highest method and narrative consensus support and the fewest semantic council conflicts. This matters because council conflicts correspond to cases where independent extraction passes disagreed enough to require human review. A lower conflict rate at similar yield and lower runtime indicates a more stable extraction configuration.

The evidence-grounding results show that exact quote matching is too strict for this corpus because older PDFs often mutate whitespace, special characters, and page numbering during digital text extraction. However, the token-overlap grounding metric showed that about two-thirds of cited evidence quotes were strongly supported by the source text, with mean token overlap near or above 0.80 in the 3-sample configurations.

### Answer to the Evaluation Questions

**Reliability:** The system was reliable across the completed evaluation: all three configurations parsed all 9 papers and produced 100% schema-valid JSON. The 5-sample condition did require retries due to API credit limits and malformed high-temperature responses, which shows that larger consensus settings increase operational cost even when final outputs are valid.

**Extraction yield:** The conservative setting extracted an average of 12.1 methods/participant facts and 3.9 narrative result facts per paper, plus 10 total tables across the corpus. This shows that the system does more than metadata extraction; it produces a usable structured summary of experimental design and reported outcomes.

**Grounding:** Every evaluated output included evidence quotes, and the conservative setting achieved a 65.0% token-grounding rate with mean token overlap of 0.802. We interpret this as partial success: the system usually attaches relevant source evidence, but source grounding remains an area for improvement because exact quote recovery is brittle on scanned or older PDFs.

**Consensus design:** Increasing temperature from 0.4 to 0.6 did not improve performance. The moderate setting had lower method and narrative support and more council conflicts. Increasing to 5 samples at temperature 0.8 extracted slightly more methods facts but lowered consensus support and increased runtime substantially. Therefore, the conservative 3-sample configuration is the best setting for this project.

Overall, we consider the system successful as a prototype extraction pipeline. It handled the full nine-paper corpus, produced valid structured JSON for every paper in the complete runs, extracted substantive methods and results facts, and attached source evidence. The main limitation is not basic parsing reliability but precision of evidence grounding and the need for human review on conflicts identified by the council QA stage.
