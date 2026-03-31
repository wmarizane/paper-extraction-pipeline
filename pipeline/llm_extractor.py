"""LLM-based extraction using vLLM for GPU inference."""

import json
import logging
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
    """Extract structured data from text chunks using vLLM."""

    def __init__(
        self,
        model_name: str = None,
        gpu_memory_utilization: float = None,
        max_model_len: int = None,
    ):
        """Initialize vLLM extractor.
        
        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-27B-Instruct")
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length (None = auto-detect)
        """
        self.model_name = model_name or settings.llm_model
        self.gpu_memory = gpu_memory_utilization or settings.vllm_gpu_memory
        self.max_model_len = max_model_len or settings.vllm_max_model_len
        self.max_retries = settings.llm_retry_attempts
        self.temperature = settings.llm_temperature

        logger.info(f"Initializing vLLM with model: {self.model_name}")
        logger.info(f"GPU memory utilization: {self.gpu_memory}")
        
        # Initialize vLLM model
        init_start = time.time()
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory,
            max_model_len=self.max_model_len,
            trust_remote_code=True,  # Required for some models
        )
        init_time = time.time() - init_start
        logger.info(f"vLLM initialization took {init_time:.2f}s")

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=settings.vllm_max_tokens,
            top_p=0.95,
        )

    def extract_batch(self, chunks: List[TextChunk]) -> List[ExtractionResult]:
        """Extract data from multiple chunks in batch (efficient).
        
        Args:
            chunks: List of text chunks to process
            
        Returns:
            List of extraction results (one per chunk)
        """
        logger.info(f"Processing batch of {len(chunks)} chunks")
        
        # Build prompts for all chunks
        prompts = [self._build_extraction_prompt(chunk) for chunk in chunks]
        
        # Batch generation with vLLM (much faster than sequential)
        batch_start = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        batch_time = time.time() - batch_start
        
        logger.info(f"Batch generation took {batch_time:.2f}s for {len(chunks)} chunks")
        
        # Parse each output
        results = []
        for chunk, output in zip(chunks, outputs):
            response_text = output.outputs[0].text
            result = self._parse_and_create_result(
                chunk=chunk,
                response_text=response_text,
                llm_calls=1,
                processing_time=batch_time / len(chunks),  # Average time per chunk
            )
            results.append(result)
        
        return results

    def extract(self, chunk: TextChunk) -> ExtractionResult:
        """Extract data from a single text chunk.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            Extraction result with structured data or error
        """
        logger.debug(f"Extracting from chunk {chunk.chunk_index} ({chunk.section})")
        
        prompt = self._build_extraction_prompt(chunk)
        
        # Try extraction with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                
                # Generate response
                outputs = self.llm.generate([prompt], self.sampling_params)
                response_text = outputs[0].outputs[0].text
                
                processing_time = time.time() - start_time
                
                # Parse response
                result = self._parse_and_create_result(
                    chunk=chunk,
                    response_text=response_text,
                    llm_calls=attempt,
                    processing_time=processing_time,
                )
                
                if result.success:
                    logger.debug(f"Extraction successful (attempt {attempt})")
                    return result
                else:
                    logger.warning(f"Parsing failed (attempt {attempt}): {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Extraction error (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    return ExtractionResult(
                        chunk_index=chunk.chunk_index,
                        section=chunk.section,
                        extracted_data=None,
                        success=False,
                        error_message=f"Failed after {self.max_retries} attempts: {str(e)}",
                        llm_calls=attempt,
                        processing_time=0.0,
                    )
        
        # Should not reach here, but handle gracefully
        return ExtractionResult(
            chunk_index=chunk.chunk_index,
            section=chunk.section,
            extracted_data=None,
            success=False,
            error_message="Unknown error",
            llm_calls=self.max_retries,
            processing_time=0.0,
        )

    def _build_extraction_prompt(self, chunk: TextChunk) -> str:
        """Build extraction prompt for a text chunk.
        
        Args:
            chunk: Text chunk to extract from
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a scientific data extraction assistant. Your task is to extract structured information from scientific papers about polymers.

**TEXT TO ANALYZE:**
Section: {chunk.section}

{chunk.text}

**EXTRACTION TASK:**
Extract the following information from the text above:

1. **Polymers mentioned**: List all polymer names, chemical formulas, or abbreviations
2. **Experimental conditions**: Temperature, pressure, pH, time, concentration, etc.
3. **Measurements**: Any quantitative data (molecular weight, yield, conversion, etc.)
4. **Methods used**: Analytical techniques (NMR, SEC, MALDI, etc.)

**OUTPUT FORMAT:**
Respond with ONLY valid JSON in this exact structure:

{{
  "polymers": [
    {{"name": "polymer name", "abbreviation": "abbreviation if any", "formula": "chemical formula if mentioned"}}
  ],
  "conditions": [
    {{"parameter": "temperature", "value": "25", "unit": "°C"}}
  ],
  "measurements": [
    {{"type": "molecular weight", "value": "45000", "unit": "g/mol"}}
  ],
  "methods": ["NMR", "SEC"]
}}

**RULES:**
- Extract only factual information present in the text
- If no information is found for a category, use an empty list []
- Do not make up or infer data that isn't explicitly stated
- Return ONLY the JSON, no additional text or explanation

JSON OUTPUT:"""
        return prompt

    def _parse_and_create_result(
        self,
        chunk: TextChunk,
        response_text: str,
        llm_calls: int,
        processing_time: float,
    ) -> ExtractionResult:
        """Parse LLM response and create extraction result.
        
        Args:
            chunk: Original text chunk
            response_text: LLM response text
            llm_calls: Number of LLM calls made
            processing_time: Time taken for extraction
            
        Returns:
            ExtractionResult with parsed data or error
        """
        try:
            extracted_data = self._parse_llm_response(response_text)
            return ExtractionResult(
                chunk_index=chunk.chunk_index,
                section=chunk.section,
                extracted_data=extracted_data,
                success=True,
                llm_calls=llm_calls,
                processing_time=processing_time,
            )
        except Exception as e:
            return ExtractionResult(
                chunk_index=chunk.chunk_index,
                section=chunk.section,
                extracted_data=None,
                success=False,
                error_message=f"JSON parsing failed: {str(e)}",
                llm_calls=llm_calls,
                processing_time=processing_time,
            )

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response text into structured data.
        
        Handles markdown code blocks and extracts JSON.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()

        # Try to parse JSON
        try:
            data = json.loads(text)
            
            # Validate structure
            expected_keys = {"polymers", "conditions", "measurements", "methods"}
            if not all(key in data for key in expected_keys):
                raise ValueError(f"Missing required keys. Expected {expected_keys}, got {set(data.keys())}")
            
            return data
            
        except json.JSONDecodeError as e:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}")

            if start >= 0 and end > start:
                json_text = text[start : end + 1]
                try:
                    data = json.loads(json_text)
                    
                    # Validate structure
                    expected_keys = {"polymers", "conditions", "measurements", "methods"}
                    if not all(key in data for key in expected_keys):
                        raise ValueError(f"Missing required keys")
                    
                    return data
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Could not parse JSON: {e}")


def extract_from_chunks(chunks: List[TextChunk], use_batch: bool = True) -> List[ExtractionResult]:
    """Convenience function to extract from multiple chunks.
    
    Args:
        chunks: List of text chunks
        use_batch: If True, use batch processing (faster)
        
    Returns:
        List of extraction results
    """
    extractor = LLMExtractor()
    
    if use_batch and len(chunks) > 1:
        return extractor.extract_batch(chunks)
    else:
        return [extractor.extract(chunk) for chunk in chunks]
