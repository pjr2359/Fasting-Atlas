# Milestone 3: Status Report (OLD vs NEW)

## OLD (verbatim project proposal; copy/paste exactly)

Title: Modular NLP Framework for Domain-Specific Information Extraction: Application to Famine Studies  
Team Members: Leane Ying (ly374), Patrick Reilly (pjr99), Gregory Marzano (gam268)  
AI Keywords: Natural Language Processing (NLP), Named Entity Recognition (NER), Knowledge Representation (KR), Information Extraction  
Application Setting: This project is intended to be used on academic research papers in famine studies (including historical, economic, public health analyses of famine events, etc.) in order to help famine researchers. Our system will process the papers and extract structured metadata related to the specific famine events, then normalize and aggregate the results.  
Description:

This project will create a domain-specific NLP pipeline that transforms unstructured academic research papers on historical famines into both machine-readable structured data and human-readable output. Famine research papers contain valuable information that is often inconsistently formatted and difficult to parse. Using NLP techniques such as named entity recognition and relation extraction, the system will identify and extract key event-level attributes that will be mapped into a standardized schema.
As previously stated, the output will be delivered in two formats which work together: a machine-readable representation which includes structured JSON and graph-based data formats will allow computers to compare different famine cases and a human-readable summary to enable researchers to understand and verify the information. The system will handle research on famine studies specifically through its modular architecture which will divide research-paper parsing work into general components and domain-specific schemas. The system design will allow scalability to other academic disciplines without needing to change the core system design.



Our project is a schema based information retrieval system that will be built on top of transformer based language models. Rather than relying on simple ‘summarization,’ we plan to define a structured famine-event schema (such as with event name, location, time span, mortality estimates, reported causes, policy responses, etc.) and use pretrained models with targeted prompting or light fine tuning to extract entities and relations that fill this schema. This ensures consistent, comparable outputs of the free-form text.

The pipeline will include document preprocessing, entity recognition, relation extraction and post processing normalization. Since famine studies give inconsistent metrics (like raw counts vs percentages), we will implement rule-based normalization to standardize units and their representations before exporting this data to structured json files. The architecture would also theoretically separate NLP components from famine specific schema logic, meaning the framework would be adaptable to other domains.
We will evaluate our system using quantitative and comparative models. First we can manually annotate a subset of the research papers for a “gold standard,” or verified dataset. Using this dataset, we can measure precision, recall, and F1 score for entity and relation extraction, as well as accuracy of normalized outputs to ensure that the structured data reliably represents source text.
Also, we will compare our schema driven method to a baseline such as unconstrained LLM summarization. We will test whether structured extraction improves our consistency and enables reliable cross paper queries (such as retrieving famines by region or time period). Our success would not only be defined by both strong extraction metrics as well as the system’s ability to support aggregating results across studies.
Planning and Design:
Week 1-2:
We would create our subset of research papers from a variety of different ranges of sources, such as historical, economic, and public health perspectives to create our corpus. We’ll design a famine-event schema to capture the attributes like name, geographic location, time span, mortality estimates, causes, and resolution, as well as developing our document preprocessing and NLP components. Document preprocessing is key for the first couple of weeks.
Week 3-6:
After finalizing the schema, we can begin the NLP implementation. We can use NER components using pretrained transformer models that have domain-specific prompting or light fine-tuning to identify relevant entities. We can define relationships between locations, dates, and causes to find relation extraction. We can also work on normalization rules for quantities and dates at the same time. A checkpoint at the end of these weeks would run the pipeline on a small sample of papers with our schema outputs.
Week 7-8
After we have a working implementation we can begin to focus on making sure outputs are consistent, usable, and comparable. We can solidify our export into standardized JSON to support cross-paper comparison. We can generate readable summaries for revision as well. We can revise our model and make any chances necessary.
Week 9-11
Once we have usable output, we can evaluate it and run a comparative analysis among a larger set of data. We can measure precision and F1 for the relation extraction, and the accuracy of normalized values. We can also identify common errors and add fixes.
Week 11-13
The final part of our project will consist of adding minor refinements to our model and mostly of writing up our technical results and report and preparing for our final presentation. We can synthesize our results and determine our findings. We will ensure our code is cleaned and ready for submission.
We will mostly rely on established NLP software and curated academic data resources. We’ll implement the system in Python using pretrained transformer-based language models through the Hugging Face Transformer library, along with supporting NLP tools like spaCy for text preprocessing and Pytorch for model fine-tuning and evaluation. The primary data sources will be peer-reviewed academic journals about famine studies that span wide historical, economic, and public health perspectives.

\---

## NEW text since Milestone 2 (paste/format as different-colored font)

\[NEW] **Project scope / domain shift (what changed vs proposal):**  
The original proposal targeted famine-event extraction specifically. The work now focuses on the Fasting-Atlas MVP: extracting method-aware, traceable experimental data from heterogeneous scientific papers in a **domain-agnostic** way (aligned with the project’s current goal of converting publications into a structured database of human experimental data). The key architectural idea remains “schema-guided structured extraction,” but the concrete schema and extraction targets are now aligned to human experimental methods/results and tables.

\[NEW] **MVP implemented (what works now):**

1. A runnable Python pipeline (`fasting\_atlas`) that ingests the PDFs in `papers/` and writes one JSON file per PDF under `parsed/`.
2. PDF ingest uses `pdfplumber` to extract per-page text and table grids.
3. Section heuristics identify “Methods/Participants” and “Results/Narrative results” candidate blocks.
4. LLM structured extraction (schema-guided JSON) extracts:

   * `metadata` (authors/year/journal/study design/demographics when detectable)
   * `methods\_participants` (assays, instruments, timing, dosing, inclusion/exclusion details)
   * `narrative\_results` (qualitative findings and comparisons)
   * `tables` (structured numeric cells from pdfplumber; plus LLM-based header/unit hints when headers are detected)
5. **Traceability is enforced in the output schema**: extracted items include evidence fields with `source\_file`, `page`, and a verbatim `quote` (and table cells retain numeric parsing when possible).
6. **Quality control (CouncilLLM):** for methods/results, the pipeline runs two independent extraction passes (temperature differs) and stores differences in `qa.council.discrepancies\[]`, plus `needs\_human\_review`.

\[NEW] **Backend options (local + hosted LLM):**  
The CLI supports two LLM backends:

* Local: Ollama (model tag configured via CLI; optional environment configuration; used for structured extraction)
* Hosted: Anthropic Claude via the Messages API (API key read from environment / project `.env` as `ANTHROPIC\_API\_KEY`)

\[NEW] **No heuristic “fallback padding” for extraction:**  
The MVP avoids inserting heuristic “fake” methods/results content when extraction fails. If an LLM call fails (timeout/model error/JSON parse error), the run errors so debugging is explicit. (Empty extractions are allowed when the section text is empty.)

\[NEW] **Current status by numbers (smoke run evidence):**  
For the current corpus under `papers/` (9 PDFs), the pipeline produced corresponding JSON outputs:

* `parsed/`: 9 JSON files created
* `parsed\_smoke/`: 1 smaller smoke run JSON created for fast iteration

\[NEW] **What’s imperfect / known issues (for reviewer context):**

1. The current CouncilLLM discrepancy diff is structural and index-based for list outputs. This can produce many “low severity” discrepancy entries that are actually semantically similar (e.g., paraphrase differences or differing list ordering).
2. Figures/graphs datapoint extraction and OCR for scanned PDFs are still out of scope for the MVP milestone.
3. Table evidence currently records cell coordinates/values, but deeper bounding-box evidence wiring can be improved next.

\[NEW] **Planned improvements before the final milestone:**

1. Add a “semantic council” step to deduplicate near-equivalent extracted items and summarize true conflicts, rather than only listing structural diffs.
2. Improve item alignment between pass 1 and pass 2 outputs (e.g., match items by normalized keys such as label/method/unit plus evidence similarity) before diffing.
3. Improve traceability for tables/plots (evidence location granularity) and expand beyond digital PDFs if needed.

\[NEW] **Updated timeline (Milestone 3 → end of semester):**

* Completed:

  * Week 1–2 equivalent: corpus selection + schema/pipeline scaffolding
  * Week 3–6 equivalent: pdf ingest (pages + tables), section finding heuristics, LLM structured extraction, dual-pass CouncilLLM diff, CLI/debug logging
* Remaining (through the milestone due date and next weeks):

  * Improve council discrepancy “noise” via semantic dedup/alignment
  * Strengthen evaluation procedure (manual checks on a subset; define success criteria)
  * Add/iterate figures/OCR extraction only if required by the course scope
  * Finalize report writing and presentation materials

\[NEW] **Division of labor (current):**

* Programming lead (Patrick Reilly): implemented the pipeline modules, schema, CLI, and CouncilLLM diff; wired Ollama/Claude backends; added debug logging and removed extraction fallbacks.
* Other team members: help by running the pipeline on the corpus with different backends, reviewing output quality on a small labeled subset (“gold standard” style), and contributing to the report narrative + updated timeline.

\[NEW] **References (technical):**

1. `pdfplumber`: https://github.com/jsvine/pdfplumber
2. Ollama: https://ollama.com/
3. Anthropic Claude API (Messages): https://docs.anthropic.com/en/docs/about-claude/models

