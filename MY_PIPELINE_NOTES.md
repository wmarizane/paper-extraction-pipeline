# My Pipeline Notes — Personal Reference
> NOT for commits. Written 2026-06-22 before meeting with Dr. Wang.
> This is a low-level walkthrough of every file, every function, every design decision.

---

## Big Picture

The pipeline converts PDFs of polymer chromatography papers into a structured database of LCCC (Liquid Chromatography at Critical Conditions) experimental conditions. It uses three AI models:
- **Qwen 3.5-27B** — primary extractor, higher reliability
- **Mistral-Small-24B** — secondary extractor, more prone to hallucinations but catches things Qwen misses
- **DeepSeek-R1-32B** — "judge" model, uses chain-of-thought reasoning to reconcile the two

Everything runs on a compute cluster (SLURM scripts). Locally you only run the CSV export.

---

## File Structure

```
paper-extraction-pipeline/
├── run_local.py            ← Phase 1 driver: PDF → raw extraction JSON (one paper at a time)
├── run_consensus.py        ← Phase 2 driver: Qwen JSON + Mistral JSON → consensus JSON
├── pipeline/
│   ├── pdf_parser.py       ← PDF → Markdown conversion
│   ├── chunker.py          ← Markdown → TextChunk(s) for the LLM
│   ├── llm_extractor.py    ← The extraction prompt + vLLM inference
│   ├── consensus_judge.py  ← DeepSeek judge prompt + all matching/merging logic
│   └── csv_exporter.py     ← Consensus JSONs → flat CSV
├── results/
│   ├── qwen3.5-27b/        ← Raw Qwen extraction JSONs, by subfolder/paper
│   ├── mistral-small-24b/  ← Raw Mistral extraction JSONs
│   └── consensus/          ← Final consensus JSONs (one per paper)
├── Inputs/                 ← Input PDFs, organized in subfolders by polymer type
│   ├── PEG/
│   ├── PLA/
│   ├── PPO/
│   ├── pdf_files_1stbatch/
│   ├── pdf_files_2ndbatch/
│   └── pdf_files_3rdbatch/
├── config/
│   ├── settings.py         ← Reads .env variables (model name, GPU settings, output dir)
│   └── model_registry.py   ← Maps model short names → HuggingFace IDs + vLLM settings
└── run_extraction.slurm    ← SLURM script for cluster Phase 1
    run_consensus.slurm     ← not shown but exists for Phase 2
```

---

## Phase 1 — Extraction (`run_local.py`)

### Entry point
`python run_local.py Inputs/PEG/[112] Pasch1995.pdf --model qwen3.5-27b --subfolder PEG`

CLI args parsed by `argparse`:
- `pdf_path` — the PDF file
- `--model` — short name looked up in model_registry (defaults to `LLM_MODEL` in `.env`)
- `--feedback` — optional string from the judge for a retry run (Phase 2 can trigger this)
- `--subfolder` — e.g. "PEG" — keeps results organized in subfolders matching Inputs/

Creates a `PipelineRunner` object, calls `.run()`.

### Stage 1 — Prerequisites check
Just calls `check_parser_ready()` from `pdf_parser.py` which always returns `True` now (was checking a REST API before, now using local PyMuPDF). Basically a no-op sanity check.

### Stage 2 — PDF parsing (`pdf_parser.py → parse_pdf_to_markdown`)
Uses `pymupdf4llm.to_markdown(pdf_path)`. This library reads the PDF's internal structure and outputs Markdown that:
- Preserves column layout as best it can
- Turns tables into Markdown table syntax (pipes)
- Preserves headers as `# / ## / ###`

If that crashes (OCR-related bug), it retries with `use_ocr=False`.  
If that also crashes, it falls back to raw `fitz.open()` page-by-page text extraction (no structure, just text).

Result: a `str` of Markdown for the whole paper.

### Stage 3 — Chunking (`chunker.py → chunk_pdf → TextChunker.process_markdown`)
**Key design decision**: we DON'T chunk by default. The whole paper goes as one `TextChunk`.

Why? Modern LLMs have large context windows (32k+ tokens). Chunking means the model sees only part of the paper per call — it might see "ACN/water 70:30" in the methods section but miss the temperature "25°C" buried in the results. The full-paper approach lets the model cross-reference everything.

Token counting uses `tiktoken` with the `cl100k_base` encoding (OpenAI's encoding — it's a reasonable approximation for any model).

Hard limit: `MAX_SAFE_TOKENS = 25000`. If the paper is larger, it recursively splits at:
1. `\n## ` (section level)
2. `\n### ` (subsection level)
3. `\n\n` (paragraph level)
4. Hard character split (last resort, never really happens)

Output: a `List[TextChunk]`. Usually length 1. Each `TextChunk` has:
```python
text: str          # the actual text
section: str       # "Full Paper" or "Part 1/2/3"
chunk_index: int   # 0, 1, 2...
token_count: int   # for logging
source_pdf: str    # filename
```

### Stage 4 — LLM Extraction (`llm_extractor.py → LLMExtractor.extract_from_chunks`)

#### Setup
`LLMExtractor.__init__` loads the model into GPU RAM via vLLM:
```python
self.llm = LLM(
    model=model_config.hf_id,          # e.g. "Qwen/Qwen2.5-7B-Instruct"
    gpu_memory_utilization=0.9,
    max_model_len=32768,
    trust_remote_code=True,
    ...model_specific_kwargs...
)
```

`SamplingParams` are set with:
```python
temperature=0.0  # deterministic (from settings)
max_tokens=8192  # max output length
top_p=0.9
structured_outputs=StructuredOutputsParams(json=EXTRACTION_SCHEMA)
```

The `structured_outputs=EXTRACTION_SCHEMA` is critical — it tells vLLM to use **grammar-constrained sampling** (also called structured generation). This means the model CANNOT produce invalid JSON. The schema enforces every field's type and required fields. Hallucinated field names are impossible.

#### The Prompt (`_build_extraction_prompt`)
The prompt is one big f-string. Structure:
1. **Role**: "You are a scientific information extraction assistant specialized in polymer liquid chromatography."
2. **Text to analyze**: the full Markdown text
3. **Task**: extract all LCCC conditions
4. **Definition of what to extract**: only conditions explicitly mentioning "LCCC", "critical condition", "critical adsorption point", or mass-independence of elution
5. **DO NOT EXTRACT**: SEC, regular HPLC, theory-only, simulations
6. **Extraction unit**: one record per distinct condition setup
7. **Interpretation rules** (15 bullet points):
   - DEDUPLICATION: merge if same experiment mentioned in abstract/methods/results
   - LITERATURE IGNORE: skip conditions cited from other papers
   - MULTIPLE ANALYTES: split into separate rows, never comma-separate
   - CRITICAL COMPONENT & ARCHITECTURE: reflect the polymer that establishes the condition (e.g., linear PS, not ring PS, if you're using it as a reference)
   - RANGES vs specific: prefer the exact critical composition over a range
   - TEMPERATURES: only the LCCC column temperature, not fractionation/synthesis temps
   - FRACTIONATION REJECTION: prep columns (ID >8mm), fractionation PURPOSE = reject
   - END-GROUPS: different end-groups = separate rows
   - POLYMER SPECIFICITY: use "Ls-PS" not "polystyrene", "it-PP" not "polypropylene"
   - COLUMN MODE: classify as "Reversed Phase", "Normal Phase", etc.
   - MULTIPLE AUTHORS: join with "; "
8. **JSON schema**: 20+ fields listed with types
9. **Output instruction**: "Return ONLY valid JSON. No markdown. No explanations. Start with {."

If `self.feedback` is set (retry after judge complaint), it's inserted as:
```
SUPERVISOR FEEDBACK FROM PREVIOUS RUN:
<feedback text>
PLEASE CORRECT YOUR MISTAKES BASED ON THIS FEEDBACK.
```

#### Chat template (`_format_prompt`)
Wraps the prompt in the model's expected chat format using the tokenizer's `apply_chat_template`. Two messages:
- System: "You are a JSON extraction assistant. Output ONLY valid JSON..."
- User: the full prompt

For Qwen, thinking mode is explicitly disabled via `enable_thinking=False` in the template kwargs — otherwise Qwen outputs a huge `<think>...</think>` block before the JSON.

#### Inference loop (`extract_from_chunks`)
Batches all chunks (usually just 1) into a single `llm.generate()` call. vLLM handles batching internally. Returns `outputs[i].outputs[0].text` for each.

Retry logic: if JSON parsing fails, it retries up to `settings.llm_retry_attempts` times using `_build_retry_prompt` ("Your previous output was invalid JSON...").

#### Response parsing (`_parse_llm_response`)
1. Strip `<think>...</think>` tags (regex, case-insensitive) — these leak from reasoning models despite being disabled
2. Strip markdown code fences (` ```json ... ``` `)
3. Try `json.loads(text)` directly
4. If that fails: find the first `{` and last `}` and try again (handles "Here is the JSON: {...}" type outputs)
5. Verify it has `extracted_conditions` key

#### Output saving (`_save_output`)
Two files written:
- `results/<model>/<subfolder>/<paper>_extracted_<timestamp>.json` — archived version
- `results/<model>/<subfolder>/<paper>_latest.json` — always overwritten, this is what consensus reads

The JSON structure:
```json
{
  "metadata": { "source_pdf": "...", "extraction_date": "...", "model": "..." },
  "summary": { "total_conditions": N },
  "extracted_data": { "conditions": [...] },
  "chunk_details": [...]
}
```

---

## Phase 2 — Consensus (`run_consensus.py`)

### Entry point
`python run_consensus.py`  
No CLI args — it scans `results/qwen3.5-27b/` for `*_latest.json` files, finds the matching Mistral files, and runs consensus for each paper.

### Loading raw data
`load_json(path)` reads `extracted_data.conditions` from a `_latest.json` file.

Tags each condition with `source_model = "qwen"` or `"mistral"` — used later in merge conflict resolution.

### The `ConsensusJudge` class (`consensus_judge.py`)

#### Initialization
Loads DeepSeek-R1-32B via vLLM with `gpu_memory_utilization=0.90` (maximized because DeepSeek is a large model). Temperature=0.6 — slightly warm because we want the reasoning to explore, not be completely deterministic.

#### `run_bidirectional_consensus(qwen_data, llama_data)`
This is the main public method. Called once per paper.

**Why bidirectional?**  
If you just run "Qwen as ground truth, validate Mistral against it", the judge's attention is biased toward Qwen's framing. Flipping the order (Mistral as primary, Qwen as validator) catches conditions that Qwen's framing might have suppressed. The intersection of both runs is more reliable than either alone.

**Step-by-step:**

##### Run A: `run_consensus(qwen_data, llama_data)` → `conds_a`
1. Builds the judge prompt with Qwen listed as "HIGH RELIABILITY" and Mistral as "LOWER RELIABILITY, PRONE TO HALLUCINATION"
2. Applies chat template (no system role for DeepSeek — it uses user message only)
3. Calls `llm.generate()` — DeepSeek thinks (internally, invisible to us since we strip `<think>` tags) and produces a JSON with `final_consensus.extracted_conditions`
4. Strips `<think>...</think>` blocks from output
5. Parses JSON, returns the full response including `requires_retry` and `feedback_for_models`

##### Run B: `run_consensus(llama_data, qwen_data)` → `conds_b`
Same but with Mistral listed first as "EXTRACTION 1" and Qwen as "EXTRACTION 2". Different framing = different attention = potentially catches different conditions.

##### Fuzzy intersection
Now we have `conds_a` and `conds_b`. We need to find the overlap.

**Phase 1: Matched pairs**  
For each condition in A, find its match in B using `_chromatographic_match()` (explained below). If found, merge them with `_merge_conditions()`.

**Phase 2: Unmatched**  
Conditions in A that had no match in B, and conditions in B that had no match in A.

**Phase 3: Validate unmatched**  
Each unmatched condition goes through `_validate_unmatched()` — a tiny LLM call:
```
"An AI extracted the following condition, but another AI completely missed it.
Is this a valid LCCC condition based on its own evidence_text, or a hallucination?"
→ {"is_valid": true/false}
```
Temperature=0.0, max_tokens=50. Quick binary decision. If valid, include it. This catches real conditions that one run missed.

**Phase 4: Final dedup**  
`_dedup_conditions()` runs the same fingerprint matcher over the combined list to remove any remaining identical conditions.

**Phase 5: Confidence tracing**  
For each final condition, trace back which original model found it:
- Compare the final condition against every Qwen input condition via `_chromatographic_match()`
- If found → `qwen_conf = qc.get("critical_condition_confidence")` (e.g., "explicit")
- If not found → `qwen_conf = "missed"`
- Same for Mistral
- Stored as `model_confidences: {"qwen": "explicit", "mistral": "missed"}`

This is the provenance tracking. You can see exactly which model contributed each row.

#### The Judge Prompt (`_build_prompt`)
15 instructions to DeepSeek. Key ones:
1. **MODEL RELIABILITY PRIORS**: Qwen = trust more. Mistral = hallucination-prone. If evidence_text supports Mistral's claim strongly, still accept it.
2. Think using `<think>` tags, debate discrepancies
3. Merge duplicates into one comprehensive record
4. Reject hallucinations (no evidence_text support)
5. If one model captured pore_size and the other didn't — merge them
6. **LITERATURE IGNORE**: column_name + flow_rate + detector ALL null = almost always a cited reference, reject
7. **SIMULATION REJECTION**: Monte Carlo, lattice models, etc.
8. **MULTIPLE ANALYTES**: split into separate rows, no comma-separation
9. **CRITICAL COMPONENT**: reflect the polymer used to ESTABLISH the condition
10. **FRACTIONATION REJECTION**: preparative purpose = reject, even if at critical conditions
11. **RANGES**: prefer specific point over range
12. **QUALITY FEEDBACK**: `requires_retry=true` if extraction is severely corrupted
13. **POLYMER SPECIFICITY**: Ls-PS > polystyrene, it-PP > polypropylene
14. **MULTIPLE AUTHORS**: join with "; "
15. Output ONLY valid JSON

#### `_chromatographic_match(ca, cb)` — THE KEY ALGORITHM
This function decides whether two condition records describe the same experiment. Called many times during intersection and dedup.

**6 signals computed:**

| Signal | How computed | Missing behavior |
|--------|-------------|-----------------|
| Column | substring containment OR word Jaccard ≥ 0.4 | True (no penalty) |
| Solvents | set overlap > 0 (after synonym normalization) | True |
| Ratio | exact after normalizing separators (`/` `\` `-` → `:`, strip spaces) | True |
| Temperature | exact string match after stripping `°C` | True |
| Critical component | canonical match OR abbreviation OR Jaccard ≥ 0.6 | True |
| Analyte polymer | same | True |

"Missing" means if either side has no value for that field, the signal defaults to True (not a contradiction). This is intentional — we don't penalize for one model not capturing pore_size.

**Chromatographic contradictions**: count of False signals among column/solvents/ratio/temp  
**Polymer contradictions**: count of False signals among component/analyte

**OVERRIDE rule**: If ≥ 2 chromatographic signals are available AND all are True (no chrom contradictions), AND canonical analyte AND canonical component both match (strict equality, no fuzzy) → return True immediately. This lets you merge "PEG" and "poly(ethylene oxide)" on the same column/ratio.

**GUARD rule** (added by us): If raw analyte names differ AND temperatures differ → return False. Prevents merging Ls-PS@14.8°C with Ring-PS@17.3°C even though both canonicalize to "polystyrene" and share the same column/ratio. This was the Ziebarth bug.

**Standard rule**: 
- Any polymer contradiction → False
- Chromatographic contradictions > 1 → False
- Otherwise → True

#### `_canonicalize_polymer(val)` — Polymer Name Normalization
Three steps:
1. Strip descriptor suffixes: "BO block" → "BO", "EO repeat unit" → "EO", "PS backbone" → "PS"
2. Look up in `CANONICAL_POLYMERS` dict (70+ entries: "ps" → "polystyrene", "peg" → "poly(ethylene glycol)", etc.) — alphanumeric-only comparison
3. Strip architecture prefix with `ARCH_PREFIX_RE` regex (`ring-`, `cyclic-`, `linear-`, `star-`, `ls-`, `lu-`, `it-`, etc.) and retry lookup

This is for MATCHING ONLY. The original specific name is preserved in the output (from Qwen if available, since Qwen has priority).

#### `_norm_solvents(solv)`
Handles both string and list inputs. Splits by `/` or `,`. Looks up in `CANONICAL_SOLVENTS` dict (e.g., "ACN" → "acetonitrile", "THF" → "tetrahydrofuran", "CH2Cl2" → "dichloromethane"). Returns a Python set for set overlap comparison.

#### `_norm_ratio(ratio)`
"58/42", "58:42", "58-42" all become "58:42". Whitespace stripped.

#### `_word_jaccard(a, b)`
Extracts all alphanumeric word tokens, computes |A∩B| / |A∪B|. Used for fuzzy column name matching and fuzzy polymer name matching when canonical matching fails.

#### `_is_abbreviation(a, b)`
Checks if the shorter string is an acronym of the longer:
- "PS" vs "Polystyrene" → checks if 'p' starts 'polystyrene' (prefix)
- "PMMA" vs "poly(methyl methacrylate)" → checks initials of words/parts

#### `_merge_conditions(ca, cb)`
For each field:
- If both are the same → use that value
- If one is None/empty → use the other
- If both are set and different → conflict

Conflict resolution:
- If one is `source_model="qwen"` and other is `"mistral"` → prefer Qwen
- Otherwise → call `_resolve_dispute()` (another tiny LLM call): "Based on evidence A vs evidence B, which value is correct?" Returns `{"resolved_value": "..."}`. Temperature=0.0, max_tokens=100.

#### Post-filters in `run_consensus.py` (lines 116-168)

**Bug note**: The sanitizer runs AFTER the null-heavy filter, so `"null"` strings (truthy!) pass the filter before being converted to None. Should be fixed by moving sanitizer before the filter.

1. **Null-heavy filter** (lines 116-133): Rejects conditions where:
   - Rule 1: column_name + flow_rate + detector ALL absent → literature reference signal
   - Rule 2: 4 or more of [column_name, mobile_phase_ratio, temperature_celsius, flow_rate, detector, pore_size] absent
   
2. **String "null" sanitizer** (lines 135-145): Converts `"null"` strings to Python `None`. Also handles nested dicts (aqueous_parameters).

3. **Comma-split** (lines 147-168): If `analyte_polymer` has ", " in it:
   - Strip content in parentheses first (to avoid splitting chemical names like "H(EO),(PO),(EO),OH")
   - Split by ", "
   - If 2+ parts all length > 1 → create separate conditions (dict copy with different analyte_polymer)
   - Heuristic — can get it wrong on complex chemical names

4. **Retry loop** (lines 75-112): If `requires_retry=True` in judge output:
   - Re-run extraction for the model that had bad quality (via `subprocess.run(["python", "run_local.py", pdf_path, "--model", model_name, "--feedback", feedback_str])`)
   - Reload JSONs, run consensus again (one more time, no further retries)

#### Output JSON format
```json
{
  "metadata": {
    "source_pdf": "Pasch1995.pdf",
    "model": "deepseek-r1-32b-consensus",
    "inputs": ["qwen3.5-27b", "mistral-small-24b"]
  },
  "summary": { "total_conditions": 3 },
  "extracted_data": {
    "conditions": [
      {
        "analyte_polymer": "triblock copolymer H(EO)x(PO)y(EO)xOH",
        "critical_component": "ethylene oxide block",
        "column_name": "Nucleosil 100 RP-18",
        ...
        "model_confidences": { "qwen": "explicit", "mistral": "missed" }
      }
    ]
  }
}
```

---

## Phase 3 — CSV Export (`csv_exporter.py`)

`export_folder_to_csv(folder_path, output_csv)` — reads all `*.json` in a folder (the consensus folder for one subfolder), flattens, writes CSV.

**The `_clean(val)` helper**: converts `None` → `""` and string `"null"` → `""`. Added to prevent literal "null" in the CSV output.

**Field mapping** (from JSON field → CSV column name):
- `paper_doi` → "DOI"
- `analyte_polymer` → "Analyte Polymer"
- `mobile_phase_solvents` (list) → joined with ", " → "Mobile Phase Solvents"
- `aqueous_parameters.pH` → "Aqueous pH"
- `model_confidences.qwen` → "Qwen Confidence"
- `model_confidences.mistral` → "Mistral Confidence"
- etc.

One CSV per subfolder (PEG, PLA, PPO, etc.).

---

## Configuration

### `.env` file (not committed)
Key variables:
- `LLM_MODEL=qwen3.5-27b` — default extraction model
- `VLLM_GPU_MEMORY=0.9` — fraction of GPU VRAM to use
- `VLLM_MAX_MODEL_LEN=32768` — max context window
- `VLLM_MAX_TOKENS=8192` — max output tokens
- `LLM_TEMPERATURE=0.0` — deterministic extraction
- `LLM_RETRY_ATTEMPTS=3`
- `OUTPUT_DIR=results`

### `config/model_registry.py`
Maps short names to HuggingFace model IDs and model-specific vLLM kwargs:
- `"qwen3.5-27b"` → `"Qwen/Qwen3-27B"`, `has_thinking_mode=True` (disable with `enable_thinking=False` in chat template)
- `"mistral-small-24b"` → `"mistralai/Mistral-Small-24B-Instruct-2501"`, standard chat
- `"deepseek-r1-32b"` → `"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"`, standard

---

## Cluster Execution

### SLURM scripts
`run_extraction.slurm` — Phase 1 on cluster:
- Requests GPU nodes
- Loops through all PDFs in a subfolder
- Runs `python run_local.py <pdf> --model <model> --subfolder <sf>` for each
- Both models run in separate SLURM jobs (or sequentially on same node)

`submit_all.sh` — submits Phase 1 for all subfolders and both models.

`submit_phase3.sh` — submits Phase 2 (consensus) after Phase 1 is done.

### Why not run locally?
Qwen 3.5-27B needs ~16GB VRAM at 4-bit quantization. DeepSeek-R1-32B needs ~20GB. A typical laptop has 8–16GB GPU. The cluster has A100s (80GB) or similar.

---

## Data Flow Summary (Numbers)

For a typical large paper (e.g. [346] Abrar2011):
- Qwen extracts: ~12 conditions (some dupes, some near-misses)
- Mistral extracts: ~9 conditions
- Run A (Qwen→Mistral judge): ~8 conditions
- Run B (Mistral→Qwen judge): ~8 conditions
- Intersection (matched pairs): ~6
- Validated unmatched: ~2
- After dedup + filters: **~7 final conditions**

The total across all 284 conditions came from ~90 papers across 6 subfolders.

---

## Key Design Decisions & Why

| Decision | Reason |
|---|---|
| Full paper as single chunk (no sliding window) | Prevents context fragmentation; temp in results won't be missed because methods are in a different chunk |
| Structured output (grammar-constrained JSON) | Eliminates JSON parsing failures; no invalid schema |
| Two models instead of one | Recall: Qwen misses some things Mistral catches and vice versa |
| DeepSeek as judge (not GPT-4) | Runs on-cluster without API costs; chain-of-thought reasoning is good for debate tasks |
| Bidirectional consensus (run A + run B) | Removes prompt-order bias |
| Fuzzy fingerprint matching (not LLM-only) | LLM merging is expensive and non-deterministic; the programmatic matcher is fast and auditable |
| Qwen preferred over Mistral in conflicts | Empirically, Qwen has better precision in this domain |
| Architecture prefix stripping for matching (Ring-PS → polystyrene) | Allows merging of condition records where one model used the specific name and other used the base name — BUT the specific name is preserved in the final output |
| Ziebarth guard (diff name AND diff temp → no merge) | Without it, Ls-PS@14.8°C and Ring-PS@17.3°C merge incorrectly because they share column/ratio and both canonicalize to "polystyrene" |

---

## Known Issues / Bugs

1. **Sanitizer ordering bug**: In `run_consensus.py`, the `"null"` string sanitizer runs AFTER the null-heavy filter. `flow_rate="null"` is truthy → passes the filter → gets sanitized to `None` too late. **Fix**: move sanitizer block above the filter block.

2. **Ziebarth prep condition**: Cond 4 (T=19°C, 250×9.8mm, "for fractionation of ring PS") still present. DeepSeek validates Mistral's extraction despite evidence text clearly saying "fractionation". New fractionation rejection instruction should catch this on next full re-run.

3. **Pasch1995 triblock duplicate**: Cond 2 and 3 are OCR variants of the same molecule (`H(EO)x(PO)y(EO)xOH` vs `H(EO),(PO),(EO),OH`). Can't match because the fuzzy matching sees different strings. Low impact (3 rows when should be 2).

4. **Comma-split heuristic**: The regex-based comma splitter can fail on complex chemical names. E.g. `"H(EO),(PO),(EO),OH"` — parentheses protection works but is fragile.

---

## How to Verify Results Quickly

```bash
# Count conditions per subfolder
python -c "
import json; from pathlib import Path
for sf in ['PEG','PLA','PPO','pdf_files_1stbatch','pdf_files_2ndbatch','pdf_files_3rdbatch']:
    n = sum(len(json.load(open(f))['extracted_data']['conditions'])
            for f in Path(f'results/consensus/{sf}').glob('*_consensus.json'))
    print(f'{sf}: {n}')
"

# Check a specific paper
python -c "import json; d=json.load(open('results/consensus/PEG/[112] Pasch1995_consensus.json')); [print(c['analyte_polymer'], c['temperature_celsius']) for c in d['extracted_data']['conditions']]"
```

---

## Manuscript Language Notes

- "We employed a three-stage automated pipeline"
- "Two independent large language models (Qwen 3.5-27B and Mistral-Small-24B) performed parallel extraction..."
- "A reasoning model (DeepSeek-R1-32B) served as a consensus judge, evaluating the independent extractions and resolving discrepancies through chain-of-thought reasoning"
- "Bidirectional consensus was employed to mitigate prompt-order bias"
- "Conditions were matched using a multi-signal chromatographic fingerprint (column identity, mobile phase solvents, composition ratio, and column temperature)"
- "The reduction in record count between individual model outputs and the consensus table reflects the pipeline's precision enforcement: individual models are tuned for high recall, while the consensus stage applies hallucination rejection, deduplication, and literature-reference filtering"
- "Model confidence provenance is tracked per condition, indicating whether each extraction was confirmed by both models (explicit/explicit), captured by one only, or missed entirely"
