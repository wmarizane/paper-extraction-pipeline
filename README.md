# paper-extraction-pipeline

LLM-based pipeline that extracts polymer **LCCC** (Liquid Chromatography at Critical
Conditions) experimental data from scientific PDFs into a structured database.
Target venue: *Digital Discovery*.

## Overview

Three phases:

1. **Extraction** (`run_local.py`) — PDF → Markdown → grammar-constrained JSON, one
   record per critical condition. Run independently for each extractor model.
2. **Consensus** (`run_consensus.py`) — a reasoning "judge" (DeepSeek-R1-32B) reconciles
   the two extractions bidirectionally, deduplicates, and traces per-model confidence.
3. **CSV export** (`pipeline/csv_exporter.py`) — consensus JSONs → one flat summary CSV
   per subfolder.

Models:

| Role | Model | Notes |
|------|-------|-------|
| Primary extractor | Qwen 3.5-27B | Higher precision; preferred in conflicts |
| Secondary extractor | Mistral-Small-24B | Higher recall, more hallucination-prone |
| Consensus judge | DeepSeek-R1-32B | Chain-of-thought reconciliation |

Inference is **vLLM** with grammar-constrained (structured) JSON output. PDF parsing is
**pymupdf4llm** (native, no server — GROBID is no longer used).

## Repository layout

```
run_local.py                    Phase 1 driver (one PDF at a time)
run_consensus.py                Phase 2 driver (scans results/, runs the judge)
pipeline/
  pdf_parser.py                 PDF -> Markdown (pymupdf4llm, with fallbacks)
  chunker.py                    Markdown -> TextChunk(s); whole-paper by default
  llm_extractor.py              Extraction prompt + schema + vLLM inference
  consensus_judge.py            Judge prompt + chromatographic matching/merging
  pre_consensus_dedup.py        Per-model dedup BEFORE consensus (Dr. Wang 7-7)
  csv_exporter.py               Consensus/extraction JSON -> summary CSV
config/
  settings.py                   .env-backed settings (pydantic)
  model_registry.py             Short name -> HF id + vLLM kwargs
results/
  qwen3.5-27b/ mistral-small-24b/   Raw per-model extractions (by subfolder/paper)
  consensus/                    Final consensus JSONs
Inputs/                         Input PDFs, organized by polymer subfolder
scripts/
  test_dedup_fix.py             Consensus dedup regression tests (22 cases)
  test_pre_consensus_dedup.py   Pre-consensus dedup tests (17 cases)
  dedup_existing_consensus.py   Post-process existing consensus JSONs (no LLM)
submit_all.sh                   Submit Phase 1 (both models, all subfolders) + Phase 3
submit_phase2.sh                Submit multi-model extraction jobs
submit_phase3.sh                SLURM batch: consensus + CSV export
run_extraction.slurm            Phase 1 cluster job (vLLM + pymupdf4llm)
```

## Running locally

Extraction and consensus need GPUs and are normally run on the cluster (see
`CLUSTER_GUIDE.md`). Two things run fine on a laptop:

```bash
# CSV export from existing consensus JSONs (no GPU / no vllm needed)
python pipeline/csv_exporter.py results/consensus/PEG results/PEG_consensus_summary.csv

# Dedup tests (mock vllm internally)
python scripts/test_pre_consensus_dedup.py
python scripts/test_dedup_fix.py
```

## Deduplication

Duplicate critical conditions are removed at three points:

- **Pre-consensus, per model** (`pre_consensus_dedup.py`): a 6-field chromatographic
  fingerprint (stationary phase chemistry, column name, mobile-phase solvents, ratio,
  temperature, critical component), case-insensitive and polymer-chemistry-aware, with
  an analyte guard that keeps genuinely distinct records apart (end-group variants, MW
  series, architecture prefixes, homo- vs block-copolymers).
- **Inside consensus** (`consensus_judge._dedup_conditions`): fuzzy fingerprint over the
  merged set.
- **CSV safety net** (`csv_exporter._dedup_csv_rows`): final pass by normalized row
  fingerprint.

## Intermediate files

- `parsed_md/<pdf_name>.md` — parsed Markdown for inspection.
- `results/<model>/<subfolder>/<paper>_latest.json` — canonical per-paper extraction
  (always overwritten; what consensus reads).
