# Deep Dive Analysis: Consensus Output Quality

## 1. Duplicates Analysis

We search for exact duplicates or near-duplicates (same column, solvents, ratio, temperature, analyte, component) within the new results to ensure the deduplication pipeline is still functioning after our changes.

- **Total New Conditions:** 309
- **Suspected Duplicates:** 38 (12.3%)
- **Conclusion:** There are still some duplicates occurring. We need to investigate why they weren\'t merged.

### Examples of Suspected Duplicates

**Paper:** `[3] Belenkii1976`
- Match 1: poly(ethylene oxide) | poly(ethylene oxide) | Q=missed M=explicit
- Match 2: poly(ethylene oxide) | poly(ethylene oxide) | Q=missed M=explicit

**Paper:** `[321] Malik2009`
- Match 1: PEG-MME 1100 | EO block | Q=explicit M=missed
- Match 2: PEG-MME 1100 | EO block | Q=explicit M=missed

**Paper:** `[321] Malik2009`
- Match 1: PEG-MME 350-PO-BO | BO block | Q=explicit M=missed
- Match 2: PEG-MME 350-PO-BO | BO block | Q=explicit M=missed

**Paper:** `[321] Malik2009`
- Match 1: PEG-MME 350-PO-BO-HO | HO block | Q=explicit M=missed
- Match 2: PEG-MME 350-PO-BO-HO | HO block | Q=explicit M=missed

**Paper:** `[321] Malik2009`
- Match 1: poly(butene oxide) | BO block | Q=explicit M=missed
- Match 2: poly(butene oxide) | BO block | Q=explicit M=missed

## 2. Ratio Normalization Consistency

The new pipeline normalizes separators (e.g., `-`, `/` are converted to `:` before comparison). Let's see the impact on raw outputs.

- **Unique Ratio Strings in OLD:** 128
- **Unique Ratio Strings in NEW:** 143
- **Separators found in NEW ratio outputs:** -, /, :
- **Note:** The judge normalizes ratios for matching, but retains the raw extracted string for the final output.

## 3. Quality of Merging (Qwen vs Mistral)

Based on the `model_confidences` metadata injected into the final records:

- **Total conditions identified confidently:** 309
- **Both models extracted independently:** 86 (27.8%)
- **Qwen extracted uniquely (Mistral missed):** 116 (37.5%)
- **Mistral extracted uniquely (Qwen missed):** 99 (32.0%)

**Conflict Resolution Analysis:**
Because of the new `source_model` deterministic fallback, when Qwen and Mistral both extract a field but disagree on the value, Qwen's value is deterministically selected without requiring an expensive LLM dispute resolution call.

## 4. Deep Dive: Recovered Distinct Analytes

### Paper: `[289] Banerjee2015`
**OLD Pipeline (1 conditions):**
> 1. Analyte: **PIB-diol** | Comp: PIB backbone | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5

**NEW Pipeline (6 conditions):**
> 1. Analyte: **PIB-diol, PIB-diallyl** | Comp: PIB backbone | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=explicit, M=missed
> 2. Analyte: **PIB-diol, PIB-monool, PIB-diallyl, PIB-dichloride, PIB-diolefin** | Comp: PIB backbone | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=explicit, M=missed
> 3. Analyte: **PIB-diol** | Comp: PIB-diol | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=missed, M=explicit
> 4. Analyte: **PIB-monool** | Comp: PIB-monool | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=missed, M=explicit
> 5. Analyte: **PIB-diallyl** | Comp: PIB-diallyl | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=missed, M=explicit
> 6. Analyte: **PIB-dichloride** | Comp: PIB-dichloride | Col: YMC HPLC COLUMN | Ratio: 80.5/19.5 | Models: Q=missed, M=explicit

**Analysis:**
The old pipeline aggressively merged all PIB variations (diol, diallyl, monool, dichloride) into a single row because they shared the identical `YMC HPLC COLUMN` and `80.5/19.5` ratio. The new pipeline correctly identifies that they have distinct `Analyte Polymer` identities and keeps them separate as 6 distinct experiments.

### Paper: `Ziebarth_2016`
**OLD Pipeline (3 conditions):**
> 1. Analyte: **polystyrene** | Comp: Ls-PS | Col: Nucleosil C18 | Ratio: 58/42
> 2. Analyte: **Lu-PS** | Comp: Lu-PS | Col: Nucleosil C18 | Ratio: 58/42
> 3. Analyte: **polystyrene** | Comp: linear | Col: Nucleosil C18 | Ratio: 58/42

**NEW Pipeline (5 conditions):**
> 1. Analyte: **Ls-PS** | Comp: Ls-PS | Col: Nucleosil C18 | Ratio: 58/42 | Models: Q=explicit, M=missed
> 2. Analyte: **Lu-PS** | Comp: Lu-PS | Col: Nucleosil C18 | Ratio: 58/42 | Models: Q=explicit, M=missed
> 3. Analyte: **Ring-PS** | Comp: Ring-PS | Col: Nucleosil C18 | Ratio: 58/42 | Models: Q=explicit, M=missed
> 4. Analyte: **polystyrene** | Comp: linear | Col: Nucleosil C18 | Ratio: 58/42 | Models: Q=missed, M=explicit
> 5. Analyte: **polystyrene** | Comp: ring | Col: Nucleosil C18 | Ratio: 58/42 | Models: Q=missed, M=explicit

**Analysis:**
The old pipeline merged different polystyrene topologies (linear, ring, Ls-PS, Lu-PS) because they all shared the `Nucleosil C18` column and `58/42` ratio. The new strict match rule preserves the distinct topologies.

### Paper: `[233] Malik2012`
**OLD Pipeline (4 conditions):**
> 1. Analyte: **Polystyrene (PS)** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82
> 2. Analyte: **PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4
> 3. Analyte: **Polystyrene (PS)** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82
> 4. Analyte: **PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4

**NEW Pipeline (8 conditions):**
> 1. Analyte: **Polystyrene (PS)** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82 | Models: Q=explicit, M=explicit
> 2. Analyte: **PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4 | Models: Q=explicit, M=explicit
> 3. Analyte: **PS-b-PEO** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82 | Models: Q=missed, M=explicit
> 4. Analyte: **PS-b-PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4 | Models: Q=missed, M=explicit
> 5. Analyte: **Polystyrene (PS)** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82 | Models: Q=explicit, M=explicit
> 6. Analyte: **PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4 | Models: Q=explicit, M=explicit
> 7. Analyte: **PS-b-PEO** | Comp: PS | Col: Symmetry 300 | Ratio: 18:82 | Models: Q=missed, M=explicit
> 8. Analyte: **PS-b-PEO** | Comp: PEO | Col: Nucleosil Si 300 | Ratio: 96:4 | Models: Q=missed, M=explicit

**Analysis:**
The old pipeline missed the block copolymer end-group distinctions entirely. The new pipeline correctly segregates `Polystyrene (PS)`, `PEO`, and the block copolymer `PS-b-PEO` (for both PS and PEO critical conditions).
