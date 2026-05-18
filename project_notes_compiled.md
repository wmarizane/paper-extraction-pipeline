# Project Journal

---

## Phase 1: Project Inception & Technical Foundation (Early-Mid March 2026)

**Objective:** Build a production-grade LLM-based pipeline to automatically extract structured scientific data from research PDFs with high reliability.

**Roadmap Established:**
- **Phase 1:** Infrastructure Pipeline (PDF parsing → text chunking → LLM extraction → JSON generation)
- **Phase 2:** Verification Layer (Schema validation, consistency checking, LLM-based verification)
- **Phase 3:** Optimization (Prompt refinement, error correction loops)

### Foundation & Initial Chunking Design (March 22-23, 2026)
- **Design:** Implemented chunking strategies (`section` vs `paragraph`) using a GPT-4 standard tokenizer (`cl100k_base`) with a greedy bin-packing algorithm to fit within LLM context windows.
- **JSON Parsing Logic:** Developed robust JSON parsing fallback mechanics for stripping markdown code blocks and extracting raw JSON objects from conversational LLM outputs.

---

## Phase 2: Department Synchronization & Schema Definition (Early-Mid April 2026)

### Chem Department Synchronization (April 6, 2026)
- **Ground Truth Provided:** The Chemistry department provided manual, human-expert extractions for 3 test papers in Excel format to serve as our evaluation baseline.
- **Initial Wishlist & Scope:** Reviewed the massive initial field wishlist ("PolyCrit"). Decided to defer some fields and focus on a condensed, critical draft for the initial test set.
- **Architectural Shift:** To ensure zero hallucinations and maximum extraction reliability, we agreed the pipeline must implement a multi-model "debate" layer (similar to Med-PaLM/BlueMed) where LLMs audit each other's extracted information.

### Prompt Engineering & Feedback (April 10 - April 17, 2026)
- **Feedback from Dr. Wang:** "ChatGPT identified critical conditions very well, even better than human expert... Wow! you got all of the conditions correctly. this is a great news."
- **Issue: Broad Extractions.** The initial prompt "Identify all polymers" was too broad and resulted in extracting irrelevant background chemistry.
- **Pivot:** Narrowed the prompt to focus strictly on LCCC definitions: *"Critical conditions are reached when polymers of the same microstructure elute independently of molar mass..."*
- **Pivot:** Integrated a strict JSON schema prompt with explicit negative constraints to ignore ordinary SEC/HPLC and general discussions of LCCC theory.

---

## Phase 3: Scaling, Multi-Model Consensus, & Overcoming Challenges (Late April/May 2026)

As the pipeline scaled to run autonomously on the GPU cluster, we encountered several significant technical and LLM behavioral hurdles that forced major architectural pivots.

### 1. The Inference Engine Pivot (Ollama to vLLM & Singularity)
- **Issue:** `ollama` struggled with CUDA utilization and concurrency on the SLURM cluster. Furthermore, the cluster security policies prohibited the use of Docker (due to root privilege requirements).
- **Pivot:** Abandoned Docker and `ollama`. We migrated the entire inference backend to `vLLM` to natively handle high-throughput batching, and packaged the environment into a `Singularity` container which runs safely in unprivileged user space.

### 2. The Model Scale Pivot (Abandoning Small LLMs)
- **Issue:** We initially tested `phi:latest` (small models) for speed, but manually verified that they suffered from severe domain-specific hallucinations. For example, `poly(butylene oxide)` was extracted as `C4H6O` (missing 2 hydrogens) and `polyisoprene` was `C8H8` (wrong carbon count).
- **Pivot:** Scrapped small quantized models entirely. Switched to large, unquantized models utilizing Multi-GPU setups, settling on `Qwen3.5-27B` for the primary extraction layer due to its exceptional structural adherence and specialized domain knowledge.

### 3. The Extraction Granularity Pivot (Schema & Rule Upgrades)
- **Issue:** Early extractions suffered from rigid schema limitations and naive interpretation.
  1. `LLaMA` was extracting solvent ratios as broad ranges (e.g., `90-95%`) instead of the specific optimal LCCC point (e.g., `92%`).
  2. Identical experimental conditions were being extracted multiple times as duplicate records simply because the analyte name changed slightly (e.g., PS/PMMA blends vs PS-b-PMMA).
  3. The rigid `solvent_1`/`solvent_2` schema broke down on complex chromatography systems requiring ternary mixtures or pH/salt modifiers.
- **Pivot:** Expanded the schema into a flexible `mobile_phase_solvents` array and added an `aqueous_parameters` object. We explicitly injected negative constraints into the prompt instructing the models to prioritize specific optimal percentages over ranges and to merge identical setups across multiple analytes into a single comma-separated record.

### 4. Air-Gapped Cluster Issues (Offline Tokenization)
- **Issue:** The SLURM compute nodes are air-gapped without internet access. When the chunker layer initialized, the `tiktoken` library attempted to silently download encoding files from OpenAI Azure blobs, causing `NameResolutionError` timeouts that crashed the pipeline.
- **Pivot:** Pre-downloaded the tokenizer files to a local cache directory on the login node and explicitly injected `TIKTOKEN_CACHE_DIR` into the SLURM environment variables, allowing 100% offline execution.

### 5. Finalizing the Map-Reduce Verification Architecture
- **Issue:** How do we guarantee the extraction of *novel* experiments and catch theoretical hallucinations? During testing, `Mistral-Small` hallucinated a full LCCC experiment from `Zhu2015` simply because the paper discussed LCCC theory and simulations in the text.
- **Pivot:** We deployed a final **Phase 3 Consensus Judge** (`DeepSeek-R1-32B`). DeepSeek acts as the Verification Layer. It reads the separate JSON outputs from both Qwen and Mistral, uses its `<think>` reasoning tokens to debate discrepancies, and applies strict rules. In the `Zhu2015` case, DeepSeek successfully identified that Mistral's extraction was based on theoretical simulations, rejected it, and output a perfectly clean `0 conditions` JSON.
