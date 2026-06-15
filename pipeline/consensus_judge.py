"""Consensus Map-Reduce Judge using DeepSeek-R1-32B via vLLM."""

import json
import logging
import re
import time
from typing import Any, Dict, List

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from config.settings import settings
from config.model_registry import get_model_config
from pipeline.llm_extractor import EXTRACTION_SCHEMA

logger = logging.getLogger(__name__)

CANONICAL_POLYMERS = {
    "ps": "polystyrene",
    "polystyrene": "polystyrene",
    "pmma": "poly(methyl methacrylate)",
    "poly(methyl methacrylate)": "poly(methyl methacrylate)",
    "polymethylmethacrylate": "poly(methyl methacrylate)",
    "polymethyl methacrylate": "poly(methyl methacrylate)",
    "pnbma": "poly(n-butyl methacrylate)",
    "poly(n-butyl methacrylate)": "poly(n-butyl methacrylate)",
    "pe": "polyethylene",
    "polyethylene": "polyethylene",
    "pp": "polypropylene",
    "polypropylene": "polypropylene",
    "it-pp": "polypropylene",
    "itpp": "polypropylene",
    "isotactic polypropylene": "polypropylene",
    "peg": "poly(ethylene glycol)",
    "peo": "poly(ethylene glycol)",
    "poly(ethylene glycol)": "poly(ethylene glycol)",
    "poly(ethylene oxide)": "poly(ethylene glycol)",
    "polyethylene glycol": "poly(ethylene glycol)",
    "polyethylene oxide": "poly(ethylene glycol)",
    "eo": "poly(ethylene glycol)",
    "ethylene oxide": "poly(ethylene glycol)",
    "ppo": "poly(propylene glycol)",
    "ppg": "poly(propylene glycol)",
    "poly(propylene glycol)": "poly(propylene glycol)",
    "poly(propylene oxide)": "poly(propylene glycol)",
    "polypropylene glycol": "poly(propylene glycol)",
    "polypropylene oxide": "poly(propylene glycol)",
    "propylene oxide": "poly(propylene glycol)",
    "po": "poly(propylene glycol)",
    "pbo": "poly(butene oxide)",
    "poly(butene oxide)": "poly(butene oxide)",
    "bo": "poly(butene oxide)",
    "butene oxide": "poly(butene oxide)",
    "pho": "poly(hexene oxide)",
    "poly(hexene oxide)": "poly(hexene oxide)",
    "ho": "poly(hexene oxide)",
    "hexene oxide": "poly(hexene oxide)",
    "pib": "polyisobutylene",
    "polyisobutylene": "polyisobutylene",
    "pla": "poly(lactic acid)",
    "plla": "poly(lactic acid)",
    "poly(lactic acid)": "poly(lactic acid)",
    "poly(l-lactic acid)": "poly(lactic acid)",
    "poly(lactide)": "poly(lactic acid)",
    "polylactide": "poly(lactic acid)",
    "pi": "polyisoprene",
    "polyisoprene": "polyisoprene",
    "1,4-pi": "polyisoprene",
    "polyisoprene (1,4-pi)": "polyisoprene",
    "polyisoprene (1,4-isoprene)": "polyisoprene",
}

CANONICAL_SOLVENTS = {
    "odcb": "1,2-dichlorobenzene",
    "ortho-dichlorobenzene": "1,2-dichlorobenzene",
    "1,2-dichlorobenzene": "1,2-dichlorobenzene",
    "tcb": "1,2,4-trichlorobenzene",
    "1,2,4-trichlorobenzene": "1,2,4-trichlorobenzene",
    "acn": "acetonitrile",
    "acetonitrile": "acetonitrile",
    "ch3cn": "acetonitrile",
    "thf": "tetrahydrofuran",
    "tetrahydrofuran": "tetrahydrofuran",
    "mek": "butanone",
    "methyl ethyl ketone": "butanone",
    "butanone": "butanone",
    "dmf": "dimethylformamide",
    "dimethylformamide": "dimethylformamide",
    "n,n-dimethylformamide": "dimethylformamide",
    "ipa": "isopropyl alcohol",
    "isopropyl alcohol": "isopropyl alcohol",
    "isopropanol": "isopropyl alcohol",
    "2-propanol": "isopropyl alcohol",
    "dcm": "dichloromethane",
    "dichloromethane": "dichloromethane",
    "ch2cl2": "dichloromethane",
}

# Architecture/topology prefixes stripped before canonical polymer lookup (for MATCHING only).
# The original specific name (e.g. "Ring-PS") is preserved in the final output via source_model priority.
ARCH_PREFIX_RE = re.compile(
    r'^(ring|cyclic|linear|star|comb|ls|lu|it|at|st|dendri|hyper|branched)[_\-\s]+',
    re.IGNORECASE
)

CONSENSUS_SCHEMA = {
    "type": "object",
    "properties": {
        "requires_retry": {"type": "boolean"},
        "feedback_for_models": {
            "type": "object",
            "properties": {
                "mistral-small-24b": {"type": ["string", "null"]},
                "qwen3.5-27b": {"type": ["string", "null"]}
            }
        },
        "final_consensus": EXTRACTION_SCHEMA
    },
    "required": ["requires_retry", "feedback_for_models", "final_consensus"]
}

class ConsensusJudge:
    """
    Takes JSON arrays from two conflicting models and uses DeepSeek-R1 
    to debate and merge them into a single Ground Truth JSON.
    """
    
    def __init__(self, model_name: str = "deepseek-r1-32b", init_llm: bool = True):
        self.model_name = model_name
        self.model_config = get_model_config(self.model_name)
        
        if init_llm:
            logger.info(f"Initializing Consensus Judge ({self.model_name})")
            vllm_kwargs = {
                "model": self.model_config.hf_id,
                "gpu_memory_utilization": 0.90, # maximize for deepseek
                "max_model_len": self.model_config.max_model_len,
                "trust_remote_code": True,
            }
            vllm_kwargs.update(self.model_config.vllm_kwargs)
            self.llm = LLM(**vllm_kwargs)
        else:
            self.llm = None

        self.sampling_params = SamplingParams(
            temperature=0.6, # Slight temperature for reasoning variation
            max_tokens=8192,
            top_p=0.9,
            structured_outputs=StructuredOutputsParams(json=CONSENSUS_SCHEMA)
        )
        
    def _build_prompt(self, qwen_data: List[Dict], llama_data: List[Dict]) -> str:
        prompt = f"""You are an expert scientific consensus judge for polymer liquid chromatography.
Two independent AI agents have extracted liquid chromatography critical conditions (LCCC) from the same paper.

Your task is to review both extractions, rigorously debate their discrepancies, and output a single GROUND TRUTH list of conditions.

EXTRACTION 1 (Model: Qwen3.5-27B - HIGH RELIABILITY PRIORS):
```json
{json.dumps(qwen_data, indent=2)}
```

EXTRACTION 2 (Model: Mistral-Small-24B - LOWER RELIABILITY PRIORS, PRONE TO HALLUCINATION):
```json
{json.dumps(llama_data, indent=2)}
```

INSTRUCTIONS:
1. **MODEL RELIABILITY PRIORS**: Qwen3.5-27B has higher extraction fidelity and precision. Mistral-small-24B is more prone to hallucinations and merging unrelated conditions. When the two extractions disagree:
   a. Prefer the extraction whose evidence_text directly and specifically supports the claim.
   b. If evidence quality is ambiguous or comparable, prefer Qwen's extraction.
   c. Only override Qwen when Mistral provides clearly stronger evidence_text that directly quotes or closely paraphrases the source paper.
   d. If Mistral contains a record that Qwen completely missed, validate it carefully against its own evidence_text before including it.
2. Use your <think> tags to debate discrepancies. Look closely at the "evidence_text" provided by each extraction to deduce which agent correctly interpreted the paper.
3. Identify and merge duplicates into a single comprehensive record.
4. Reject any hallucinated conditions that lack explicit or strong inference in the evidence text.
5. If one extraction captured valid secondary fields (like pore size or detector) that the other missed, merge them together.
6. **LITERATURE IGNORE**: Reject any conditions that are merely referenced as background literature or previous studies. ONLY output novel experiments performed by the authors. A strong signal of a literature reference is when column_name, flow_rate, AND detector are ALL null — this almost always means the condition was extracted from a cited table, not the authors' own work. Reject it unless the evidence_text explicitly confirms it as a novel author experiment.
7. **SIMULATION REJECTION**: Reject any conditions that are based on computer simulations, Monte Carlo modeling, theoretical lattices, or numerical calculations.
8. **MULTIPLE ANALYTES**: If a critical condition has comma-separated analyte polymers, you MUST split them into SEPARATE records — one per polymer — duplicating all other fields. NEVER output comma-separated polymer names in analyte_polymer. Exception: commas that are part of the chemical name itself (e.g. "1,4-polyisoprene", "H(EO),(PO),(EO),OH").
9. **CRITICAL COMPONENT & ARCHITECTURE**: If the condition is established on a specific polymer (e.g., linear) but used to analyze a different polymer (e.g., cyclic), `critical_component` and `architecture` MUST reflect the polymer used to **establish** the condition.
10. **FRACTIONATION/PREPARATIVE REJECTION**: Reject any condition whose primary purpose is preparative fractionation or sample preparation — even if it operates at critical conditions. Only include ANALYTICAL LCCC measurements. Strong rejection signals: (a) semi-preparative/preparative column dimensions (inner diameter > 8mm), (b) evidence_text describes "fractionation" or "preparative" as the PURPOSE, (c) the setup is used to isolate fractions, not characterize the sample. A semi-preparative column used specifically for LCCC analysis (not fractionation) IS valid.
11. **RANGES**: If a range is extracted but a specific optimal percentage is also given, prioritize the specific percentage.
12. **QUALITY FEEDBACK**: If either extraction is severely corrupted, set "requires_retry" to true and provide string feedback. Otherwise false.
13. **POLYMER SPECIFICITY**: Always preserve the most specific polymer name from the source paper. Prefer topology-specific names (Ls-PS, Lu-PS, Ring-PS over generic "polystyrene"), stereospecific names (it-PP over "polypropylene"), and end-group-specific names over generic base polymer names. The more specific name is almost always the correct one.
14. **MULTIPLE AUTHORS**: If the paper has multiple corresponding authors, include all of them in `corresponding_author_name` separated by '; '. Same for `corresponding_email_address` and `physical_address`.
15. Output ONLY valid JSON matching the schema below.

JSON SCHEMA:
{{
  "requires_retry": "boolean",
  "feedback_for_models": {{
    "mistral-small-24b": "string or null",
    "qwen3.5-27b": "string or null"
  }},
  "final_consensus": {{
    "extracted_conditions": [
      {{
        "analyte_polymer": "string (comma-separated if multiple) or null",
        "critical_component": "string or null",
        "architecture": "string or null",
        "critical_condition_basis": "string or null",
        "critical_condition_confidence": "explicit | strong_inference | unclear",
        "column_name": "string or null",
        "stationary_phase_chemistry": "string or null",
        "column_mode": "string or null — e.g. 'Reversed Phase', 'Normal Phase', 'Size Exclusion', 'Hydrophilic Interaction'",
        "mobile_phase_solvents": "array of strings or null",
        "mobile_phase_ratio": "string or null",
        "mobile_phase_ratio_units": "string or null",
        "aqueous_parameters": {{
          "pH": "string or null",
          "salt_added": "boolean",
          "salt_type": "string or null",
          "salt_concentration": "string or null"
        }},
        "temperature_celsius": "string or null",
        "flow_rate": "string or null",
        "pore_size": "string or null",
        "column_dimensions": "string or null",
        "detector": "string or null",
        "evidence_text": "string or null",
        "notes": "string or null",
        "paper_doi": "string or null",
        "corresponding_author_name": "string or null — if multiple, join with '; '",
        "corresponding_email_address": "string or null — if multiple, join with '; '",
        "physical_address": "string or null — if multiple, join with '; '",
        "publication_year": "string or null"
      }}
    ]
  }}
}}

Output ONLY the final JSON starting with {{. Do not output anything after the JSON.
"""
        return prompt

    def _format_prompt(self, raw_prompt: str) -> str:
        try:
            tokenizer = self.llm.get_tokenizer()
            messages = [
                {"role": "user", "content": raw_prompt}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return raw_prompt

    def run_consensus(self, qwen_data: List[Dict], llama_data: List[Dict]) -> Dict[str, Any]:
        if not qwen_data and not llama_data:
            return {
                "requires_retry": False,
                "feedback_for_models": {"mistral-small-24b": None, "qwen3.5-27b": None},
                "final_consensus": {"extracted_conditions": []}
            }
            
        raw_prompt = self._build_prompt(qwen_data, llama_data)
        formatted_prompt = self._format_prompt(raw_prompt)
        
        outputs = self.llm.generate([formatted_prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text
        
        # Clean tags
        text = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", response_text)
        text = re.sub(r"```json\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()
        
        try:
            data = json.loads(text)
            if "extracted_conditions" not in data:
                data["extracted_conditions"] = []
            return data
        except json.JSONDecodeError:
            # Fallback
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                try:
                    data = json.loads(text[first_brace:last_brace + 1])
                    if "extracted_conditions" not in data:
                        data["extracted_conditions"] = []
                    return data
                except:
                    pass
        
        raise ValueError("DeepSeek failed to return valid JSON.")

    @staticmethod
    def _norm(val):
        """Normalize a string value for comparison: lowercase, strip, collapse whitespace."""
        if not val:
            return ""
        return re.sub(r'\s+', ' ', str(val).lower().strip())

    @staticmethod
    def _norm_solvents(solv):
        """Normalize solvent list to a sorted set of lowercase strings with synonym replacement."""
        if not solv:
            return set()
        if isinstance(solv, str):
            raw_set = {s.strip().lower() for s in solv.replace("/", ",").split(",") if s.strip()}
        else:
            raw_set = {str(s).strip().lower() for s in solv if s}
            
        norm_set = set()
        for s in raw_set:
            s_clean = re.sub(r'[^a-z0-9]', '', s)
            matched = False
            for key, canonical in CANONICAL_SOLVENTS.items():
                key_clean = re.sub(r'[^a-z0-9]', '', key)
                if s_clean == key_clean or s_clean == key or s == key:
                    norm_set.add(canonical)
                    matched = True
                    break
            if not matched:
                norm_set.add(s)
        return norm_set

    @staticmethod
    def _norm_ratio(ratio):
        """Normalize mobile phase ratio by stripping spaces, lowering, and unifying separators."""
        if not ratio:
            return ""
        ratio = str(ratio).lower().strip()
        ratio = re.sub(r'[/\\-]', ':', ratio)
        return re.sub(r'\s+', '', ratio)

    @staticmethod
    def _word_jaccard(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        if not a or not b:
            return 0.0
        words_a = set(re.findall(r'[a-z0-9]+', a.lower()))
        words_b = set(re.findall(r'[a-z0-9]+', b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _canonicalize_polymer(val: str) -> str:
        """Canonicalize polymer names using mapping, falling back to clean lowercase."""
        if not val:
            return ""
        val_norm = val.lower().strip()

        # Step 1: Strip descriptor suffixes commonly appended to critical_component values.
        # e.g. "BO block" -> "BO", "EO repeat unit" -> "EO", "PS backbone" -> "PS"
        val_norm = re.sub(
            r'\s+(block|repeat\s*unit|repeating\s*unit|unit|segment|chain|backbone)$',
            '', val_norm, flags=re.IGNORECASE
        ).strip()

        # Step 2: Try direct canonical lookup first (handles exact matches like "ps", "peo", etc.)
        val_clean = re.sub(r'[^a-z0-9]', '', val_norm)
        for key, canonical in CANONICAL_POLYMERS.items():
            key_clean = re.sub(r'[^a-z0-9]', '', key)
            if val_clean == key_clean or val_clean == key or val_norm == key:
                return canonical

        # Step 3: Strip architecture/topology prefixes and retry.
        # e.g. "Ring-PS" -> "PS" -> "polystyrene", "Ls-PS" -> "PS" -> "polystyrene"
        val_stripped = ARCH_PREFIX_RE.sub('', val_norm).strip()
        if val_stripped and val_stripped != val_norm:
            val_stripped_clean = re.sub(r'[^a-z0-9]', '', val_stripped)
            for key, canonical in CANONICAL_POLYMERS.items():
                key_clean = re.sub(r'[^a-z0-9]', '', key)
                if val_stripped_clean == key_clean or val_stripped_clean == key or val_stripped == key:
                    return canonical

        return val_norm

    @staticmethod
    def _is_abbreviation(a: str, b: str) -> bool:
        """
        Check if one string is a prefix or acronym/abbreviation of another.
        e.g., 'PS' and 'Polystyrene' -> True
              'PMMA' and 'Poly(methyl methacrylate)' -> True
        """
        if not a or not b:
            return False
        a = a.lower().strip()
        b = b.lower().strip()
        
        if a == b:
            return True
            
        # Ensure a is the shorter one
        if len(a) > len(b):
            a, b = b, a
            
        if len(a) < 2:
            return False
            
        # Clean special chars
        a_clean = re.sub(r'[^a-z0-9]', '', a)
        b_clean = re.sub(r'[^a-z0-9]', '', b)
        
        if not a_clean or not b_clean:
            return False
            
        # Prefix check (e.g., "polyisop" and "polyisoprene")
        if b_clean.startswith(a_clean) and len(a_clean) >= 4:
            return True
            
        # Acronym check:
        # e.g., 'pmma' and 'poly(methyl methacrylate)'
        # Split b into words/parts on non-alphanumeric
        parts = [p for p in re.split(r'[^a-z0-9]', b) if p]
        if len(parts) >= len(a_clean):
            # Check if initials match
            initials = "".join(p[0] for p in parts)
            if initials.startswith(a_clean) or a_clean == initials:
                return True
                
        # Handle 'p' prefix for 'poly'
        # e.g., 'pnbma' -> 'poly(n-butyl methacrylate)'
        if a.startswith('p') and b.startswith('poly'):
            a_sub = a[1:]
            b_sub = b[4:]
            a_sub_clean = re.sub(r'[^a-z0-9]', '', a_sub)
            b_sub_clean = re.sub(r'[^a-z0-9]', '', b_sub)
            if b_sub_clean.startswith(a_sub_clean):
                return True
            parts_sub = [p for p in re.split(r'[^a-z0-9]', b_sub) if p]
            if len(parts_sub) >= len(a_sub_clean):
                initials_sub = "".join(p[0] for p in parts_sub)
                if initials_sub.startswith(a_sub_clean) or a_sub_clean == initials_sub:
                    return True
                    
        return False

    @staticmethod
    def _chromatographic_match(ca: Dict, cb: Dict) -> bool:
        """
        Fuzzy chromatographic fingerprint matching with abbreviation and synonym support.
        
        Uses a multi-signal scoring approach across hard experimental parameters
        (column, solvents, ratio, temperature) and soft semantic fields 
        (critical_component, analyte_polymer). Two items describing the same experiment phrased
        differently should still match.
        
        Returns True if the conditions likely describe the same experimental setup.
        """
        norm = ConsensusJudge._norm
        norm_solvents = ConsensusJudge._norm_solvents
        norm_ratio = ConsensusJudge._norm_ratio
        word_jaccard = ConsensusJudge._word_jaccard
        canon_poly = ConsensusJudge._canonicalize_polymer
        is_abbrev = ConsensusJudge._is_abbreviation

        col_a = norm(ca.get("column_name"))
        col_b = norm(cb.get("column_name"))
        solv_a = norm_solvents(ca.get("mobile_phase_solvents"))
        solv_b = norm_solvents(cb.get("mobile_phase_solvents"))
        ratio_a = norm_ratio(ca.get("mobile_phase_ratio"))
        ratio_b = norm_ratio(cb.get("mobile_phase_ratio"))
        temp_a = norm(ca.get("temperature_celsius"))
        temp_b = norm(cb.get("temperature_celsius"))
        
        comp_a = norm(ca.get("critical_component"))
        comp_b = norm(cb.get("critical_component"))
        analyte_a = norm(ca.get("analyte_polymer"))
        analyte_b = norm(cb.get("analyte_polymer"))
        
        comp_a_canon = canon_poly(comp_a)
        comp_b_canon = canon_poly(comp_b)
        analyte_a_canon = canon_poly(analyte_a)
        analyte_b_canon = canon_poly(analyte_b)

        # --- Signal 1: Column name (substring containment or word overlap) ---
        if col_a and col_b:
            col_match = (col_a in col_b) or (col_b in col_a) or (word_jaccard(col_a, col_b) >= 0.4)
        else:
            col_match = True  # If either is missing, don't penalize

        # --- Signal 2: Solvents (set overlap) ---
        if solv_a and solv_b:
            solv_match = len(solv_a & solv_b) > 0
        else:
            solv_match = True

        # --- Signal 3: Mobile phase ratio (exact after normalization) ---
        if ratio_a and ratio_b:
            ratio_match = (ratio_a == ratio_b)
        else:
            ratio_match = True

        # --- Signal 4: Temperature (exact after normalization) ---
        if temp_a and temp_b:
            # Strip unit suffixes for comparison
            temp_a_clean = re.sub(r'[°c\s]', '', temp_a)
            temp_b_clean = re.sub(r'[°c\s]', '', temp_b)
            temp_match = (temp_a_clean == temp_b_clean)
        else:
            temp_match = True

        # --- Signal 5: Critical component (word-level Jaccard, lenient) ---
        if comp_a and comp_b:
            comp_match = (
                (comp_a_canon == comp_b_canon) or 
                (comp_a_canon in comp_b_canon) or 
                (comp_b_canon in comp_a_canon) or 
                is_abbrev(comp_a, comp_b) or 
                (word_jaccard(comp_a, comp_b) >= 0.6)
            )
        else:
            comp_match = True
            
        # --- Signal 6: Analyte polymer (word-level Jaccard, lenient) ---
        if analyte_a and analyte_b:
            analyte_match = (
                (analyte_a_canon == analyte_b_canon) or 
                (analyte_a_canon in analyte_b_canon) or 
                (analyte_b_canon in analyte_a_canon) or 
                is_abbrev(analyte_a, analyte_b) or 
                (word_jaccard(analyte_a, analyte_b) >= 0.6)
            )
        else:
            analyte_match = True

        # Split signals into chromatographic and polymer naming parts
        chrom_signals = []
        if col_a and col_b:
            chrom_signals.append(col_match)
        if solv_a and solv_b:
            chrom_signals.append(solv_match)
        if ratio_a and ratio_b:
            chrom_signals.append(ratio_match)
        if temp_a and temp_b:
            chrom_signals.append(temp_match)

        poly_signals = []
        if comp_a and comp_b:
            poly_signals.append(comp_match)
        if analyte_a and analyte_b:
            poly_signals.append(analyte_match)

        # If no hard signals are comparable, fall back to True (rare edge case)
        if not chrom_signals and not poly_signals:
            return True

        chrom_contradictions = sum(1 for s in chrom_signals if not s)
        poly_contradictions = sum(1 for s in poly_signals if not s)

        # OVERRIDE RULE: If chromatographic setup matches perfectly AND polymer identity
        # is compatible, allow soft naming mismatches. Requires analyte OR critical_component
        # to agree (or at least one to be missing) to prevent merging distinct analytes
        # that share the same column/solvent setup (e.g., PEG vs PEG-MME).
        if len(chrom_signals) >= 2 and chrom_contradictions == 0:
            # Use STRICT analyte identity for the override (no lenient Jaccard).
            # Two analytes must be canonically equal to merge via chromatographic override.
            strict_analyte_ok = (
                not analyte_a or not analyte_b or  # one side missing — OK
                analyte_a_canon == analyte_b_canon  # canonical match — OK
            )
            strict_comp_ok = (
                not comp_a or not comp_b or  # one side missing — OK
                comp_a_canon == comp_b_canon  # canonical match — OK
            )
            if strict_analyte_ok and strict_comp_ok:
                return True
            # Otherwise fall through to standard rule

        # GUARD: If raw analyte names differ AND temperature differs, these are
        # distinct experiments on the same column setup — do NOT merge.
        # Example: "Ls-PS" at 14.8°C vs "Ring-PS" at 17.3°C on the same Nucleosil C18
        # column with the same mobile phase. Both canonicalize to "polystyrene" but
        # they are separate critical conditions for different polymer topologies.
        if (analyte_a and analyte_b and analyte_a != analyte_b and
                temp_a and temp_b and not temp_match):
            return False

        # Standard rule: polymer identity mismatches are hard blockers.
        # Chromatographic contradictions are allowed (at most 1) only if polymers agree.
        if poly_contradictions > 0:
            return False
        return chrom_contradictions <= 1

    def _merge_conditions(self, ca: Dict, cb: Dict) -> Dict:
        """Merge two conditions, resolving conflicts via Dispute Resolution if necessary."""
        merged = {}
        disputes = []
        for k in set(ca.keys()) | set(cb.keys()):
            va = ca.get(k)
            vb = cb.get(k)
            
            # Simple cases
            if va == vb:
                merged[k] = va
            elif not va and vb:
                merged[k] = vb
            elif va and not vb:
                merged[k] = va
            else:
                # Conflict! va and vb are both present and different.
                source_a = ca.get("source_model")
                source_b = cb.get("source_model")
                
                if source_a == "qwen" and source_b == "mistral":
                    merged[k] = va
                elif source_b == "qwen" and source_a == "mistral":
                    merged[k] = vb
                else:
                    # Fallback to Dispute Resolution
                    disputes.append((k, va, vb))
                    merged[k] = va # Temporary fallback
                
        if disputes and self.llm:
            for k, va, vb in disputes:
                # Exclude dicts/lists from simple string dispute resolution for now
                if isinstance(va, str) and isinstance(vb, str) and len(va) < 200:
                    resolved = self._resolve_dispute(k, va, vb, ca.get("evidence_text", ""), cb.get("evidence_text", ""))
                    merged[k] = resolved
                    
        return merged

    def _resolve_dispute(self, field: str, val_a: str, val_b: str, ev_a: str, ev_b: str) -> str:
        """Runs a tiny focused prompt to resolve a field contradiction."""
        prompt = f"""You are resolving a conflict between two AI extractions for the field '{field}'.
Option A: "{val_a}"
Evidence A: "{ev_a}"

Option B: "{val_b}"
Evidence B: "{ev_b}"

Based on the evidence, which value is more scientifically accurate? 
Return ONLY a valid JSON object: {{"resolved_value": "chosen string or null"}}"""
        
        schema = {
            "type": "object",
            "properties": {"resolved_value": {"type": ["string", "null"]}},
            "required": ["resolved_value"]
        }
        
        try:
            formatted = self._format_prompt(prompt)
            params = SamplingParams(temperature=0.0, max_tokens=100, structured_outputs=StructuredOutputsParams(json=schema))
            out = self.llm.generate([formatted], params, use_tqdm=False)
            data = json.loads(re.sub(r"```json|```", "", out[0].outputs[0].text).strip())
            return data.get("resolved_value", val_a)
        except Exception as e:
            logger.warning(f"Dispute resolution failed for {field}: {e}")
            return val_a

    def _validate_unmatched(self, cond: Dict) -> bool:
        """Validates if an unmatched condition is a hallucination or real."""
        if not self.llm:
            return True
            
        prompt = f"""An AI extracted the following condition, but another AI completely missed it.
Condition:
```json
{json.dumps(cond, indent=2)}
```

Is this a valid, physically performed experimental LCCC condition based on its own evidence_text, or is it a hallucination/simulation/literature reference?
Return ONLY a valid JSON object: {{"is_valid": true/false}}"""
        
        schema = {
            "type": "object",
            "properties": {"is_valid": {"type": "boolean"}},
            "required": ["is_valid"]
        }
        
        try:
            formatted = self._format_prompt(prompt)
            params = SamplingParams(temperature=0.0, max_tokens=50, structured_outputs=StructuredOutputsParams(json=schema))
            out = self.llm.generate([formatted], params, use_tqdm=False)
            data = json.loads(re.sub(r"```json|```", "", out[0].outputs[0].text).strip())
            return data.get("is_valid", True)
        except Exception as e:
            logger.warning(f"Validation failed for unmatched condition: {e}")
            return True

    @staticmethod
    def _dedup_conditions(conds: List[Dict]) -> List[Dict]:
        """Remove duplicate conditions based on chromatographic fingerprint."""
        if not conds:
            return []
        deduped = []
        for c in conds:
            is_dup = False
            for existing in deduped:
                if ConsensusJudge._chromatographic_match(c, existing):
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(c)
        return deduped

    def run_bidirectional_consensus(self, qwen_data: List[Dict], llama_data: List[Dict]) -> Dict[str, Any]:
        logger.info("Running Bidirectional Consensus (Swap and Intersect)...")
        # Run A: Qwen -> Mistral
        logger.info("Run A (Qwen -> Mistral)")
        out_a = self.run_consensus(qwen_data, llama_data)
        
        # Run B: Mistral -> Qwen
        logger.info("Run B (Mistral -> Qwen)")
        out_b = self.run_consensus(llama_data, qwen_data)
        
        conds_a = out_a.get("final_consensus", {}).get("extracted_conditions", [])
        conds_b = out_b.get("final_consensus", {}).get("extracted_conditions", [])
        
        logger.info(f"Run A produced {len(conds_a)} conditions, Run B produced {len(conds_b)} conditions.")
        
        # --- Fuzzy Chromatographic Fingerprint Intersection ---
        # Phase 1: Find matched pairs (present in both runs)
        matched_conds = []
        used_b = set()
        
        for i, ca in enumerate(conds_a):
            best_j = -1
            for j, cb in enumerate(conds_b):
                if j in used_b:
                    continue
                if self._chromatographic_match(ca, cb):
                    best_j = j
                    break
            
            if best_j >= 0:
                merged = self._merge_conditions(ca, conds_b[best_j])
                matched_conds.append(merged)
                used_b.add(best_j)
                logger.info(f"  Matched A[{i}] <-> B[{best_j}]: {self._norm(ca.get('column_name', ''))} / {self._norm(ca.get('critical_component', ''))}")
        
        # Phase 2: Collect unmatched conditions from both runs
        unmatched_a = [ca for i, ca in enumerate(conds_a) if not any(
            self._chromatographic_match(ca, conds_b[j]) for j in range(len(conds_b)) if j not in used_b
        ) and i >= len(matched_conds)]  # already matched ones won't be here
        
        unmatched_b_indices = set(range(len(conds_b))) - used_b
        unmatched_b = [conds_b[j] for j in unmatched_b_indices]
        
        # Phase 3: Include and validate unmatched conditions
        all_conds = matched_conds.copy()
        
        for ca in conds_a:
            already_in = any(self._chromatographic_match(ca, m) for m in all_conds)
            if not already_in:
                if self._validate_unmatched(ca):
                    logger.info(f"  Including validated unmatched from Run A: {self._norm(ca.get('column_name', ''))} / {self._norm(ca.get('critical_component', ''))}")
                    all_conds.append(ca)
                else:
                    logger.info(f"  Rejected unmatched from Run A (Validation failed).")
        
        for cb in [conds_b[j] for j in unmatched_b_indices]:
            already_in = any(self._chromatographic_match(cb, m) for m in all_conds)
            if not already_in:
                if self._validate_unmatched(cb):
                    logger.info(f"  Including validated unmatched from Run B: {self._norm(cb.get('column_name', ''))} / {self._norm(cb.get('critical_component', ''))}")
                    all_conds.append(cb)
                else:
                    logger.info(f"  Rejected unmatched from Run B (Validation failed).")
        
        # Phase 4: Deduplicate the final set
        final_conds = self._dedup_conditions(all_conds)
        
        # Phase 5: Trace back confidences to Qwen and Mistral inputs
        for fc in final_conds:
            qwen_conf = "missed"
            mistral_conf = "missed"
            for qc in qwen_data:
                if self._chromatographic_match(fc, qc):
                    qwen_conf = qc.get("critical_condition_confidence", "unclear")
                    break
            for mc in llama_data:
                if self._chromatographic_match(fc, mc):
                    mistral_conf = mc.get("critical_condition_confidence", "unclear")
                    break
            fc["model_confidences"] = {"qwen": qwen_conf, "mistral": mistral_conf}
        
        requires_retry = out_a.get("requires_retry", False) or out_b.get("requires_retry", False)
        fb_mistral = out_a.get("feedback_for_models", {}).get("mistral-small-24b") or out_b.get("feedback_for_models", {}).get("mistral-small-24b")
        fb_qwen = out_a.get("feedback_for_models", {}).get("qwen3.5-27b") or out_b.get("feedback_for_models", {}).get("qwen3.5-27b")
        
        logger.info(f"Bidirectional consensus complete: {len(matched_conds)} matched + {len(final_conds) - len(matched_conds)} unmatched = {len(final_conds)} total (from {len(conds_a)} A, {len(conds_b)} B).")
        
        return {
            "requires_retry": requires_retry,
            "feedback_for_models": {
                "mistral-small-24b": fb_mistral,
                "qwen3.5-27b": fb_qwen
            },
            "final_consensus": {
                "extracted_conditions": final_conds
            }
        }
