"""Consensus Map-Reduce Judge using DeepSeek-R1-32B via vLLM."""

import json
import logging
import re
import time
from typing import Any, Dict, List

from vllm import LLM, SamplingParams
from config.settings import settings
from config.model_registry import get_model_config

logger = logging.getLogger(__name__)

class ConsensusJudge:
    """
    Takes JSON arrays from two conflicting models and uses DeepSeek-R1 
    to debate and merge them into a single Ground Truth JSON.
    """
    
    def __init__(self):
        self.model_name = "deepseek-r1-32b"
        self.model_config = get_model_config(self.model_name)
        
        logger.info("Initializing Consensus Judge (DeepSeek-R1-32B)")
        vllm_kwargs = {
            "model": self.model_config.hf_id,
            "gpu_memory_utilization": 0.90, # maximize for deepseek
            "max_model_len": self.model_config.max_model_len,
            "trust_remote_code": True,
        }
        vllm_kwargs.update(self.model_config.vllm_kwargs)
        
        self.llm = LLM(**vllm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=0.6, # Slight temperature for reasoning variation
            max_tokens=8192,
            top_p=0.9,
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
6. Output ONLY valid JSON matching the 19-field LCCC schema.

JSON SCHEMA:
{{
  "extracted_conditions": [
    {{
      "analyte_polymer": "string or null",
      "critical_component": "string or null",
      "architecture": "string or null",
      "critical_condition_basis": "string or null",
      "critical_condition_confidence": "explicit | strong_inference | unclear",
      "column_name": "string or null",
      "stationary_phase_chemistry": "string or null",
      "mobile_phase_solvent_1": "string or null",
      "mobile_phase_solvent_2": "string or null",
      "mobile_phase_other_components": "string or null",
      "mobile_phase_ratio": "string or null",
      "mobile_phase_ratio_units": "string or null",
      "temperature_celsius": "string or null",
      "flow_rate": "string or null",
      "pore_size": "string or null",
      "column_dimensions": "string or null",
      "detector": "string or null",
      "evidence_text": "string or null",
      "notes": "string or null"
    }}
  ]
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
            return {"extracted_conditions": []}
            
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
