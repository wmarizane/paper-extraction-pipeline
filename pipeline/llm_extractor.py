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
    
    This class handles:
    - Running local inference with vLLM
    - Sending prompts to the LLM
    - Parsing JSON responses
    - Retry logic for failures
    - Error handling
    """
    
    def __init__(
        self,
        model_name: str = None,
        gpu_memory_utilization: float = None,
        max_model_len: int = None
    ):
        """
        Initialize the LLM extractor.
        
        Args:
            model_name: HuggingFace model ID (default: from config)
            gpu_memory_utilization: Fraction of GPU memory for vLLM
            max_model_len: Optional model max sequence length override
        """
        self.model_name = model_name or settings.llm_model
        self.gpu_memory = gpu_memory_utilization or settings.vllm_gpu_memory
        self.max_model_len = max_model_len or settings.vllm_max_model_len
        self.gdn_prefill_backend = settings.vllm_gdn_prefill_backend
        self.max_retries = settings.llm_retry_attempts
        self.temperature = settings.llm_temperature

        logger.info(f"Initializing vLLM with model: {self.model_name}")
        logger.info(f"GPU memory utilization: {self.gpu_memory}")
        logger.info(f"GDN prefill backend: {self.gdn_prefill_backend}")
        
        # Initialize vLLM model
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory,
            max_model_len=self.max_model_len,
            additional_config={"gdn_prefill_backend": self.gdn_prefill_backend},
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=settings.vllm_max_tokens,
            top_p=0.9,
        )
        logger.info("LLM extractor initialized with model: %s", self.model_name)
    
    def extract(self, chunk: TextChunk) -> ExtractionResult:
        """Extract data from a single text chunk.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            Extraction result with structured data or error
        """
        logger.debug(f"Extracting from chunk {chunk.chunk_index} ({chunk.section})")
        
        # Try extraction with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                prompt = (
                    self._build_extraction_prompt(chunk)
                    if attempt == 1
                    else self._build_retry_prompt(chunk)
                )
                
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
        prompt = f"""/no_think
You are a scientific chromatography extraction assistant.

**TEXT TO ANALYZE:**
Section: {chunk.section}

{chunk.text}

**EXTRACTION TASK:**
Extract structured rows for three tables.
Critical condition definition:
"Critical condition is reached when polymers of the same microstructure elute independently of molar mass, due to compensation of enthalpic and entropic interactions with the chromatographic system."
Focus on chromatography systems for critical condition.

**OUTPUT FORMAT:**
Respond with ONLY valid JSON in this exact structure:

{{
  "master_table": [
    {{
      "paper": "Paper1|Paper2|Paper3 or citation id if explicit",
      "system_id": "H1, L1, T1, ... if inferable, else blank",
      "polymer_system": "polymer or block system",
      "target_at_cc": "target species at critical condition",
      "architecture_context": "diblock/triblock/microstructure context",
      "column": "column name",
      "stationary_phase": "stationary phase description",
      "mobile_phase": "solvent names",
      "composition": "ratio values",
      "units": "v/v or w/w etc.",
      "temp_c": "temperature in Celsius",
      "additives": "additives if present",
      "notes": "short note",
      "source_section": "section name",
      "source_text": "verbatim supporting quote"
    }}
  ],
  "separation_mechanism": [
    {{
      "paper": "paper identifier",
      "system": "system or system range",
      "type_of_criticality": "e.g., block criticality (PEO), microstructure criticality",
      "driving_variable": "solvent composition / temperature / column type",
      "non_critical_species_behavior": "what non-target species do",
      "source_section": "section name",
      "source_text": "verbatim supporting quote"
    }}
  ],
  "column_system_metadata": [
    {{
      "paper": "paper identifier",
      "column": "column name",
      "brand_type": "manufacturer or column type",
      "stationary_phase_chemistry": "chemistry",
      "pore_size_a": "pore size in angstrom",
      "dimensions": "column dimensions",
      "notes": "short reproducibility note",
      "source_section": "section name",
      "source_text": "verbatim supporting quote"
    }}
  ],
  "chunk_section": "{chunk.section}"
}}

**RULES:**
- Extract only factual information present in the text
- If no information is found for a table, use an empty list []
- Do not make up or infer data that isn't explicitly stated
- Keep source_text concise and verbatim where possible
- Do NOT output `<think>`, `<reasoning>`, or any analysis text
- Do NOT use markdown fences
- Return ONLY the JSON object, no additional text or explanation
- Start the response directly with a JSON object

JSON OUTPUT:"""
        return prompt

    def _build_retry_prompt(self, chunk: TextChunk) -> str:
        """Build stricter retry prompt when prior output was not valid JSON."""
        return (
            "/no_think\n"
            "Your previous output was invalid. Return ONLY one valid JSON object.\n"
            "No reasoning text. No markdown. No extra commentary.\n\n"
            f"{self._build_extraction_prompt(chunk)}"
        )

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

        def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("Response root must be a JSON object")

            data.setdefault("master_table", [])
            data.setdefault("separation_mechanism", [])
            data.setdefault("column_system_metadata", [])
            data.setdefault("chunk_section", "")

            if not isinstance(data["master_table"], list):
                data["master_table"] = []
            if not isinstance(data["separation_mechanism"], list):
                data["separation_mechanism"] = []
            if not isinstance(data["column_system_metadata"], list):
                data["column_system_metadata"] = []

            normalized_master = []
            for row in data["master_table"]:
                if not isinstance(row, dict):
                    continue
                normalized_master.append(
                    {
                        "paper": row.get("paper", ""),
                        "system_id": row.get("system_id", ""),
                        "polymer_system": row.get("polymer_system", ""),
                        "target_at_cc": row.get("target_at_cc", ""),
                        "architecture_context": row.get("architecture_context", ""),
                        "column": row.get("column", ""),
                        "stationary_phase": row.get("stationary_phase", ""),
                        "mobile_phase": row.get("mobile_phase", ""),
                        "composition": row.get("composition", ""),
                        "units": row.get("units", ""),
                        "temp_c": row.get("temp_c", ""),
                        "additives": row.get("additives", ""),
                        "notes": row.get("notes", ""),
                        "source_section": row.get("source_section", ""),
                        "source_text": row.get("source_text", ""),
                    }
                )
            data["master_table"] = normalized_master

            normalized_mechanism = []
            for row in data["separation_mechanism"]:
                if not isinstance(row, dict):
                    continue
                normalized_mechanism.append(
                    {
                        "paper": row.get("paper", ""),
                        "system": row.get("system", ""),
                        "type_of_criticality": row.get("type_of_criticality", ""),
                        "driving_variable": row.get("driving_variable", ""),
                        "non_critical_species_behavior": row.get("non_critical_species_behavior", ""),
                        "source_section": row.get("source_section", ""),
                        "source_text": row.get("source_text", ""),
                    }
                )
            data["separation_mechanism"] = normalized_mechanism

            normalized_meta = []
            for row in data["column_system_metadata"]:
                if not isinstance(row, dict):
                    continue
                normalized_meta.append(
                    {
                        "paper": row.get("paper", ""),
                        "column": row.get("column", ""),
                        "brand_type": row.get("brand_type", ""),
                        "stationary_phase_chemistry": row.get("stationary_phase_chemistry", ""),
                        "pore_size_a": row.get("pore_size_a", ""),
                        "dimensions": row.get("dimensions", ""),
                        "notes": row.get("notes", ""),
                        "source_section": row.get("source_section", ""),
                        "source_text": row.get("source_text", ""),
                    }
                )
            data["column_system_metadata"] = normalized_meta

            return data

        def _strip_reasoning_blocks(raw: str) -> str:
            # Qwen3.5 frequently emits reasoning tags before JSON.
            cleaned = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", " ", raw)
            cleaned = re.sub(r"(?is)<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>", " ", cleaned)
            cleaned = re.sub(r"(?is)<\s*analysis\s*>.*?<\s*/\s*analysis\s*>", " ", cleaned)

            # If a dangling reasoning preface remains, keep content from first JSON brace onward.
            lowered = cleaned.lower()
            if "<think" in lowered or "<reasoning" in lowered or "<analysis" in lowered:
                first_brace = cleaned.find("{")
                if first_brace != -1:
                    cleaned = cleaned[first_brace:]
            return cleaned.strip()

        def _extract_markdown_fences(raw: str) -> List[str]:
            blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
            return [b.strip() for b in blocks if b and b.strip()]

        def _extract_balanced_json_objects(raw: str) -> List[str]:
            objs: List[str] = []
            depth = 0
            start = -1
            in_string = False
            escaped = False

            for idx, ch in enumerate(raw):
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == "\"":
                        in_string = False
                    continue

                if ch == "\"":
                    in_string = True
                    continue
                if ch == "{":
                    if depth == 0:
                        start = idx
                    depth += 1
                elif ch == "}" and depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        objs.append(raw[start:idx + 1].strip())
                        start = -1
            return objs

        def _repair_json_text(raw: str) -> str:
            repaired = raw
            repaired = repaired.replace("“", "\"").replace("”", "\"").replace("’", "'")
            # Remove trailing commas before object/array close.
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            # Quote unquoted keys like {foo: "x"} or , foo:
            repaired = re.sub(
                r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*)(\s*:)',
                lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
                repaired,
            )
            return repaired

        cleaned = _strip_reasoning_blocks(text)
        parse_candidates: List[str] = []
        parse_candidates.extend(_extract_markdown_fences(cleaned))
        parse_candidates.append(cleaned)

        checked = set()
        errors: List[str] = []
        for candidate in parse_candidates:
            if not candidate:
                continue
            json_candidates = [candidate]
            json_candidates.extend(_extract_balanced_json_objects(candidate))

            for json_text in json_candidates:
                if not json_text or json_text in checked:
                    continue
                checked.add(json_text)
                try:
                    return _normalize(json.loads(json_text))
                except (json.JSONDecodeError, ValueError) as e:
                    errors.append(str(e))
                    repaired = _repair_json_text(json_text)
                    if repaired != json_text:
                        try:
                            return _normalize(json.loads(repaired))
                        except (json.JSONDecodeError, ValueError) as e2:
                            errors.append(str(e2))

        preview = cleaned[:280].replace("\n", " ")
        last_error = errors[-1] if errors else "no JSON object found"
        raise ValueError(
            f"Could not parse JSON from LLM response: {last_error}\nResponse: {preview}"
        )
    
    def extract_from_chunk(self, chunk: TextChunk) -> ExtractionResult:
        """
        Extract structured data from a single text chunk.
        
        This is the main method you'll call. It:
        1. Builds a prompt
        2. Sends it to the LLM
        3. Parses the response
        4. Retries if needed
        5. Returns structured result
        
        Args:
            chunk: Text chunk to extract from
            
        Returns:
            ExtractionResult with extracted data or error info
        """
        start_time = time.time()
        llm_calls = 0
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                llm_calls += 1
                
                # Build prompt
                prompt = self._build_extraction_prompt(chunk)
                
                # Call LLM via vLLM
                outputs = self.llm.generate([prompt], self.sampling_params)
                response_text = outputs[0].outputs[0].text
                
                extracted_data = self._parse_llm_response(response_text)
                
                # Success!
                processing_time = time.time() - start_time
                
                return ExtractionResult(
                    chunk_index=chunk.chunk_index,
                    section=chunk.section,
                    extracted_data=extracted_data,
                    success=True,
                    llm_calls=llm_calls,
                    processing_time=processing_time
                )
                
            except Exception as e:
                last_error = str(e)
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.warning("Attempt %s failed: %s", attempt + 1, e)
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error("All %s attempts failed for chunk %s", self.max_retries, chunk.chunk_index)
        
        # All retries exhausted
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            chunk_index=chunk.chunk_index,
            section=chunk.section,
            extracted_data=None,
            success=False,
            error_message=last_error,
            llm_calls=llm_calls,
            processing_time=processing_time
        )
    
    def extract_from_chunks(self, chunks: List[TextChunk]) -> List[ExtractionResult]:
        """
        Extract data from multiple chunks.

        Uses vLLM batch generation for throughput while preserving
        output order with input chunks.
        
        Args:
            chunks: List of text chunks to process
            
        Returns:
            List of ExtractionResult objects
        """
        if not chunks:
            return []

        results: List[Optional[ExtractionResult]] = [None] * len(chunks)
        llm_calls = [0] * len(chunks)
        pending_indices = list(range(len(chunks)))

        for attempt in range(1, self.max_retries + 1):
            if not pending_indices:
                break

            prompts = [
                self._build_extraction_prompt(chunks[idx])
                if attempt == 1
                else self._build_retry_prompt(chunks[idx])
                for idx in pending_indices
            ]
            outputs = self.llm.generate(prompts, self.sampling_params)

            next_pending: List[int] = []
            for idx, output in zip(pending_indices, outputs):
                llm_calls[idx] += 1
                chunk = chunks[idx]
                response_text = output.outputs[0].text
                try:
                    extracted_data = self._parse_llm_response(response_text)
                    results[idx] = ExtractionResult(
                        chunk_index=chunk.chunk_index,
                        section=chunk.section,
                        extracted_data=extracted_data,
                        success=True,
                        llm_calls=llm_calls[idx],
                        processing_time=0.0,
                    )
                except Exception as e:
                    if attempt < self.max_retries:
                        next_pending.append(idx)
                    else:
                        results[idx] = ExtractionResult(
                            chunk_index=chunk.chunk_index,
                            section=chunk.section,
                            extracted_data=None,
                            success=False,
                            error_message=str(e),
                            llm_calls=llm_calls[idx],
                            processing_time=0.0,
                        )
            pending_indices = next_pending

        finalized: List[ExtractionResult] = []
        for idx, result in enumerate(results):
            if result is not None:
                finalized.append(result)
                continue
            chunk = chunks[idx]
            finalized.append(
                ExtractionResult(
                    chunk_index=chunk.chunk_index,
                    section=chunk.section,
                    extracted_data=None,
                    success=False,
                    error_message="Unknown extraction failure",
                    llm_calls=llm_calls[idx],
                    processing_time=0.0,
                )
            )

        return finalized


def extract_from_chunks(chunks: List[TextChunk], use_batch: bool = True) -> List[ExtractionResult]:
    """Convenience function to extract from multiple chunks.
    
    Args:
        chunks: List of text chunks
        use_batch: If True, use batch processing (faster)
        
    Returns:
        List of extraction results
    """
    extractor = LLMExtractor()
    return extractor.extract_from_chunks(chunks)


if __name__ == "__main__":
    """Test the LLM extractor on sample chunks."""
    import sys
    from pathlib import Path
    from pipeline.pdf_parser import parse_pdf_with_grobid, check_grobid_server
    from pipeline.chunker import chunk_pdf
    
    print("Testing LLM Extractor\n")
    
    # Check GROBID
    if not check_grobid_server():
        print("❌ GROBID server not running")
        sys.exit(1)
    extractor = LLMExtractor()
    print("LLM extractor initialized\n")
    
    # Test on first paper
    test_pdf = "Inputs/polymerPaper1.pdf"
    
    if not Path(test_pdf).exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        sys.exit(1)
    
    print(f"Processing {test_pdf}...\n")
    
    # Get chunks
    xml = parse_pdf_with_grobid(test_pdf)
    chunks = chunk_pdf(xml, "polymerPaper1.pdf")
    
    print(f"Created {len(chunks)} chunks")
    
    # Extract data
    results = extractor.extract_from_chunks(chunks)
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.processing_time for r in results)
    total_llm_calls = sum(r.llm_calls for r in results)
    
    print(f"Total chunks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per chunk: {total_time/len(results):.2f}s")
    
    # Show extracted data
    print("\n" + "="*60)
    print("EXTRACTED DATA")
    print("="*60)
    
    for result in results:
        if result.success:
            print(f"\nChunk {result.chunk_index} ({result.section}):")
            print(f"  Master rows: {len(result.extracted_data.get('master_table', []))}")
            print(f"  Separation rows: {len(result.extracted_data.get('separation_mechanism', []))}")
