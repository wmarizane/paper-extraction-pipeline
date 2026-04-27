"""LLM-based extraction from scientific papers using vLLM."""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

from config.settings import settings
from config.model_registry import get_model_config, ModelConfig
from pipeline.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from LLM extraction of a text chunk."""
    chunk_index: int
    section: str
    extracted_data: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    llm_calls: int = 0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "section": self.section,
            "extracted_data": self.extracted_data,
            "success": self.success,
            "error_message": self.error_message,
            "llm_calls": self.llm_calls,
            "processing_time": self.processing_time,
        }


class LLMExtractor:
    """
    Extracts structured data from text using vLLM.

    Model-agnostic: uses the model registry to apply model-specific
    configuration (GDN backend for Qwen3.5, thinking mode handling, etc).
    """
    
    def __init__(
        self,
        model_name: str = None,
        gpu_memory_utilization: float = None,
        max_model_len: int = None
    ):
        self.model_name = model_name or settings.llm_model
        self.model_config: ModelConfig = get_model_config(self.model_name)
        self.gpu_memory = gpu_memory_utilization or settings.vllm_gpu_memory
        self.max_model_len = max_model_len or settings.vllm_max_model_len or self.model_config.max_model_len
        self.max_retries = settings.llm_retry_attempts
        self.temperature = settings.llm_temperature

        logger.info(f"Initializing vLLM with model: {self.model_config.hf_id}")
        logger.info(f"  Quantization: {self.model_config.quantization}")
        logger.info(f"  VRAM: ~{self.model_config.vram_gb:.1f} GiB")
        logger.info(f"  Max model len: {self.max_model_len}")
        logger.info(f"  Model-specific kwargs: {self.model_config.vllm_kwargs}")
        
        # Build vLLM kwargs — model-specific settings from registry
        vllm_kwargs = {
            "model": self.model_config.hf_id,
            "gpu_memory_utilization": self.gpu_memory,
            "max_model_len": self.max_model_len,
            "trust_remote_code": True,
        }
        vllm_kwargs.update(self.model_config.vllm_kwargs)

        self.llm = LLM(**vllm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=settings.vllm_max_tokens,
            top_p=0.9,
        )
        logger.info("LLM extractor initialized.")

    def _build_extraction_prompt(self, chunk: TextChunk) -> str:
        """
        Dr. Wang's v2 extraction prompt (approved 2026-04-17).
        19-field schema with explicit LCCC definitions and interpretation rules.
        """
        prompt = f"""You are a scientific information extraction assistant specialized in polymer liquid chromatography.

TEXT TO ANALYZE:
{chunk.text}

TASK:
Extract all explicitly mentioned or strongly supported liquid chromatography critical conditions (LCCC) experimental setups from the text.

DEFINITION OF WHAT TO EXTRACT:
Extract a record only if the text indicates one of the following:
1. it explicitly mentions "critical condition", "liquid chromatography at critical condition", "LCCC", "critical adsorption point", or equivalent terminology; OR
2. it explicitly states that elution/retention is independent of molar mass for a defined polymer species/component; OR
3. it explicitly gives a chromatography setup identified as the critical point/critical composition for a polymer, block, backbone, or end-group-defined species.

DO NOT EXTRACT:
- ordinary SEC, RPC, HPLC, adsorption, or interaction chromatography conditions unless the text clearly identifies them as critical conditions
- general discussion of LCCC theory without an experimental setup
- inferred values not stated in the text

EXTRACTION UNIT:
Create one entry per distinct critical-condition setup.
If multiple polymer species or blocks each have different critical conditions, create separate entries.

IMPORTANT INTERPRETATION RULES:
- Distinguish between the full analyte polymer and the specific component/species at critical condition.
- If a block copolymer is analyzed under critical conditions for one block, record the analyte polymer and also the specific critical component.
- **DEDUPLICATION**: If the exact same experimental setup is mentioned multiple times in the text (e.g. in the abstract, methods, and conclusion), merge them into a single entry. Do not create duplicate records for identical setups.
- Use null for missing information.
- Preserve reported wording where exact normalization is not possible.
- Do not guess units or compositions.
- If evidence is suggestive but not explicit, mark confidence accordingly.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema. No markdown. No explanations. Start with {{{{}}.

JSON SCHEMA:
{{{{
  "extracted_conditions": [
    {{{{
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
    }}}}
  ]
}}}}
"""
        return prompt

    def _build_retry_prompt(self, chunk: TextChunk) -> str:
        return (
            "Your previous output was invalid JSON. You must return ONLY one valid JSON object and no text.\n\n"
            f"{self._build_extraction_prompt(chunk)}"
        )

    def _format_prompt(self, raw_prompt: str) -> str:
        """Applies chat template with model-specific settings."""
        try:
            tokenizer = self.llm.get_tokenizer()
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": "You are a JSON extraction assistant. Output ONLY valid JSON. Do not think, reason, or explain. Start your response with { and end with }."
                    },
                    {"role": "user", "content": raw_prompt}
                ]

                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }

                # Disable thinking mode for models that have it
                if self.model_config.has_thinking_mode:
                    template_kwargs["enable_thinking"] = False

                return tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
        except Exception as e:
            logger.warning(f"Could not apply chat template: {e}")
        return raw_prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        
        # Clean reasoning/thinking tags that might leak
        text = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)
        text = re.sub(r"```json\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        
        text = text.strip()
        
        # Try direct parse first
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Root must be a JSON object")
            if "extracted_conditions" not in data:
                data["extracted_conditions"] = []
            return data
        except json.JSONDecodeError:
            pass
        
        # Fallback: find the first JSON object in mixed output (reasoning + JSON)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            try:
                data = json.loads(json_candidate)
                if not isinstance(data, dict):
                    raise ValueError("Root must be a JSON object")
                if "extracted_conditions" not in data:
                    data["extracted_conditions"] = []
                logger.info("Extracted JSON from mixed LLM output")
                return data
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to decode JSON: no valid JSON found\nRaw Response: {text[:300]}")

    def extract_from_chunks(self, chunks: List[TextChunk]) -> List[ExtractionResult]:
        if not chunks:
            return []

        results: List[Optional[ExtractionResult]] = [None] * len(chunks)
        pending_indices = list(range(len(chunks)))
        llm_calls = [0] * len(chunks)

        for attempt in range(1, self.max_retries + 1):
            if not pending_indices:
                break

            raw_prompts = [
                self._build_extraction_prompt(chunks[idx]) if attempt == 1 else self._build_retry_prompt(chunks[idx])
                for idx in pending_indices
            ]
            
            # Use chat template
            formatted_prompts = [self._format_prompt(p) for p in raw_prompts]

            start_time = time.time()
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            time_taken = time.time() - start_time

            next_pending: List[int] = []
            for i, idx in enumerate(pending_indices):
                llm_calls[idx] += 1
                response_text = outputs[i].outputs[0].text
                
                try:
                    data = self._parse_llm_response(response_text)
                    results[idx] = ExtractionResult(
                        chunk_index=idx,
                        section=chunks[idx].section,
                        extracted_data=data,
                        success=True,
                        llm_calls=llm_calls[idx],
                        processing_time=time_taken
                    )
                except Exception as e:
                    if attempt < self.max_retries:
                        next_pending.append(idx)
                    else:
                        results[idx] = ExtractionResult(
                            chunk_index=idx,
                            section=chunks[idx].section,
                            extracted_data=None,
                            success=False,
                            error_message=str(e),
                            llm_calls=llm_calls[idx],
                            processing_time=time_taken
                        )
            
            pending_indices = next_pending

        return results


def extract_from_chunks(chunks: List[TextChunk], use_batch: bool = True) -> List[ExtractionResult]:
    extractor = LLMExtractor()
    return extractor.extract_from_chunks(chunks)

if __name__ == "__main__":
    print("Run via run_local.py or run_extraction.slurm")
