"""LLM-based extraction from scientific papers using vLLM."""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

from config.settings import settings
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
    """
    
    def __init__(
        self,
        model_name: str = None,
        gpu_memory_utilization: float = None,
        max_model_len: int = None
    ):
        self.model_name = model_name or settings.llm_model
        self.gpu_memory = gpu_memory_utilization or settings.vllm_gpu_memory
        self.max_model_len = max_model_len or settings.vllm_max_model_len
        self.max_retries = settings.llm_retry_attempts
        self.temperature = settings.llm_temperature

        logger.info(f"Initializing vLLM with model: {self.model_name}")
        
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory,
            max_model_len=self.max_model_len,
            gdn_prefill_backend="triton",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048, # increased max tokens mapping for potentially large tables
            top_p=0.9,
        )
        logger.info("LLM extractor initialized.")

    def _build_extraction_prompt(self, chunk: TextChunk) -> str:
        prompt = f"""You are a scientific data extraction assistant specialized in polymer chemistry.

**TEXT TO ANALYZE:**
{chunk.text}

**EXTRACTION TASK:**
Extract the liquid chromatography critical conditions (LCCC) experimental setups mentioned in the text.
A critical condition is reached when polymers of the same microstructure elute independently of molar mass.
Identify all polymer systems and their associated chromatography parameters.

**RULES:**
- Output ONLY valid JSON matching the schema below.
- Do not output any markdown formatting, reasoning, or thinking tags. Start directly with the {{ character.
- If no critical conditions are found, return an empty array for "extracted_conditions".
- Do not invent information. Leave fields as null if the information is missing.

**JSON SCHEMA REQUIRED:**
{{
  "extracted_conditions": [
    {{
      "polymer_system": "string (e.g., Polyisoprene, EO-PO Block Copolymer)",
      "target_species_at_critical_condition": "string",
      "architecture": "string (e.g., diblock, triblock, random)",
      "column_name": "string",
      "stationary_phase_chemistry": "string",
      "mobile_phase_solvents": "string (e.g., Butanone/Cyclohexane)",
      "mobile_phase_composition": "string (e.g., 92/8)",
      "temperature_celsius": "string",
      "pore_size": "string",
      "column_dimensions": "string",
      "notes": "string (any additional important details)"
    }}
  ]
}}
"""
        return prompt

    def _build_retry_prompt(self, chunk: TextChunk) -> str:
        return (
            "Your previous output was invalid JSON. You must return ONLY one valid JSON object and no text.\n\n"
            f"{self._build_extraction_prompt(chunk)}"
        )

    def _format_prompt(self, raw_prompt: str) -> str:
        """Applies chat template if the model supports it."""
        try:
            tokenizer = self.llm.get_tokenizer()
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.warning(f"Could not apply chat template: {e}")
        return raw_prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        
        # Clean reasoning tags that might leak
        text = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)
        text = re.sub(r"```json\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```\s*(.*?)\s*```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        
        text = text.strip()
        
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Root must be a JSON object")
            if "extracted_conditions" not in data:
                data["extracted_conditions"] = []
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {str(e)}\nRaw Response: {text[:200]}")

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
            
            # Use chat template!
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
