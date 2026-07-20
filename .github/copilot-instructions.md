# AI Agent Instructions — Paper Extraction Pipeline

> Project context for AI coding assistants. Richer working memory (people, terms,
> current work) lives in local, uncommitted files (`CLAUDE.md`, `HANDOFF.md`) on the
> maintainer's machine.

## Objective

Extract polymer **LCCC** (Liquid Chromatography at Critical Conditions) experimental
conditions from scientific PDFs into a structured database. Target: *Digital Discovery*
submission. Emphasis on reliability — deduplication, hallucination rejection, and
per-record provenance.

## Architecture (3 phases)

```
PDF ──pymupdf4llm──► Markdown ──chunker──► TextChunk(s)
                                              │
                    Phase 1: run_local.py     ▼  vLLM + grammar-constrained JSON
        ┌─────────────────────────────────────────────────┐
        │  Qwen 3.5-27B (primary)   Mistral-Small-24B (2nd) │  → results/<model>/<sub>/*_latest.json
        └─────────────────────────────────────────────────┘
                                              │
                    Phase 2: run_consensus.py ▼
        pre_consensus_dedup (per model)  →  DeepSeek-R1-32B bidirectional judge
                                              │  → results/consensus/<sub>/*_consensus.json
                    Phase 3: csv_exporter.py  ▼
        one summary CSV per subfolder (+ CSV-level dedup safety net)
```

Extraction and consensus run on the iTiger cluster via SLURM (see `CLUSTER_GUIDE.md`).
Only CSV export runs comfortably on a laptop.

## Current status

- ✅ Phase 1 extraction (vLLM, model-agnostic via `config/model_registry.py`)
- ✅ Phase 2 bidirectional consensus with chromatographic fingerprint matching
- ✅ Per-model pre-consensus dedup (`pipeline/pre_consensus_dedup.py`)
- ✅ Phase 3 CSV export with dedup safety net
- ✅ Telemetry logging (`pipeline/telemetry.py`)
- ☐ Whole-sentence evidence columns (Dillon's request — needs `llm_extractor.py` prompt change)
- ☐ Digital Discovery reproducibility logging (input/output logs, model ids + dates)

## Team

- **Dr. Xiaofei Zhang** — main supervisor (Computer Science, Wesley's dept).
- **Dr. Yongmei Wang** — supervisor (Chemistry); sets extraction/data requirements.
- **Dillon** — RA in Dr. Wang's group; reviews CSV outputs.
- **Wesley** — builds the pipeline (CS side).

## Conventions & guardrails

- **Parser is pymupdf4llm**, not GROBID. `pdf_parser.parse_pdf_to_markdown` has OCR-off
  and plain-text fallbacks.
- **Whole paper as one chunk** by default (25k-token cap); avoids context fragmentation.
- **Grammar-constrained output** (`EXTRACTION_SCHEMA` / `CONSENSUS_SCHEMA`) — invalid
  JSON and hallucinated field names are impossible by construction.
- **Matching helpers live in `consensus_judge.py`** (`_canonicalize_polymer`,
  `_is_abbreviation`, `_word_jaccard`, `_norm_solvents`, `_analyte_base_family_match`).
  Reuse them — do not duplicate.
- **Ziebarth guard**: different raw analyte names AND different temperatures must never
  merge. Preserve this behavior in any matching change.
- **Do not modify `results/`** by hand.
- **Model short names** (`qwen3.5-27b`, `mistral-small-24b`, `deepseek-r1-32b`) map to HF
  ids in `config/model_registry.py`.

## Tests

```bash
python scripts/test_dedup_fix.py            # consensus matcher (22 cases)
python scripts/test_pre_consensus_dedup.py  # pre-consensus dedup (17 cases)
```

Both mock `vllm` internally, so they run without GPUs.
