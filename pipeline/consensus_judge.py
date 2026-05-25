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

EXTRACTION 1:
```json
{json.dumps(qwen_data, indent=2)}
```

EXTRACTION 2:
```json
{json.dumps(llama_data, indent=2)}
```

INSTRUCTIONS:
1. Review both extractions impartially. 
2. Use your <think> tags to debate discrepancies. Look closely at the "evidence_text" provided by each extraction to deduce which agent correctly interpreted the paper.
3. Identify and merge duplicates into a single comprehensive record.
4. Reject any hallucinated conditions that lack explicit or strong inference in the evidence text.
5. If one extraction captured valid secondary fields (like pore size or detector) that the other missed, merge them together.
6. **LITERATURE IGNORE**: Reject any conditions that are merely referenced as background literature or previous studies. ONLY output novel experiments performed by the authors.
7. **SIMULATION REJECTION**: Reject any conditions that are based on computer simulations, Monte Carlo modeling, theoretical lattices, or numerical calculations. The database MUST only contain real, physical, laboratory chromatography experiments. If the candidates extracted simulation parameters, reject them and set "final_consensus" to an empty list (with no conditions).
8. **MULTIPLE ANALYTES**: If the exact same critical condition is applied to multiple analyte polymers, merge them into ONE record and list all polymers in `analyte_polymer` separated by commas.
9. **RANGES**: If a range is extracted but a specific optimal percentage is also given, prioritize the specific percentage.
10. **QUALITY FEEDBACK**: If either extraction is severely corrupted, hallucinated, or completely failed due to a missing section, set "requires_retry" to true and provide explicit string feedback for that model in "feedback_for_models" so they can try again. If both are fine, set requires_retry to false.
11. Output ONLY valid JSON matching the schema below.

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
        "paper_doi": "string or null"
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
