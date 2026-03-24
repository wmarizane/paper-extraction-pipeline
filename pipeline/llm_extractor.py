"""
LLM-based data extraction from scientific papers.

Takes text chunks and uses a local LLM (via Ollama) to extract structured
information like polymer names, experimental conditions, and measurements.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama

from config import settings
from pipeline.chunker import TextChunk


@dataclass
class ExtractionResult:
    """
    Result of LLM extraction on a single chunk.
    
    Attributes:
        chunk_index: Which chunk this came from
        section: Section name (e.g., "Methods", "Results")
        extracted_data: The structured data extracted by LLM
        success: Whether extraction succeeded
        error_message: Error details if extraction failed
        llm_calls: Number of LLM calls made (for retry tracking)
        processing_time: Time taken in seconds
    """
    chunk_index: int
    section: str
    extracted_data: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    llm_calls: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_index": self.chunk_index,
            "section": self.section,
            "extracted_data": self.extracted_data,
            "success": self.success,
            "error_message": self.error_message,
            "llm_calls": self.llm_calls,
            "processing_time": self.processing_time
        }


class LLMExtractor:
    """
    Extracts structured data from text using a local LLM.
    
    This class handles:
    - Connecting to Ollama
    - Sending prompts to the LLM
    - Parsing JSON responses
    - Retry logic for failures
    - Error handling
    """
    
    def __init__(self, model_name: str = None, ollama_url: str = None):
        """
        Initialize the LLM extractor.
        
        Args:
            model_name: Ollama model to use (default: from config)
            ollama_url: Ollama server URL (default: from config)
        """
        self.model_name = model_name or settings.llm_model
        self.ollama_url = ollama_url or settings.ollama_url
        self.max_retries = settings.llm_retry_attempts
        self.timeout = settings.llm_timeout
        self.temperature = settings.llm_temperature
        
        # Initialize Ollama client
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=self.temperature,
            timeout=self.timeout
        )
        
        print(f"🤖 LLM Extractor initialized: {self.model_name} @ {self.ollama_url}")
    
    def _build_extraction_prompt(self, chunk: TextChunk) -> str:
        """
        Build a prompt for the LLM to extract data from a chunk.
        
        This is CRITICAL - the quality of extraction depends heavily on
        the prompt design. We use a structured prompt that:
        1. Explains the task clearly
        2. Gives examples of desired output format
        3. Asks for JSON response only
        
        Args:
            chunk: The text chunk to extract from
            
        Returns:
            The prompt string
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
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        LLMs sometimes wrap JSON in markdown code blocks or add extra text.
        This function handles common cases and extracts clean JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Remove markdown code blocks if present
        text = response_text.strip()
        
        if text.startswith('```json'):
            # Extract content between ```json and ```
            text = text.split('```json')[1].split('```')[0].strip()
        elif text.startswith('```'):
            # Extract content between ``` and ```
            text = text.split('```')[1].split('```')[0].strip()
        
        # Try to parse JSON
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to find JSON object in text
            # Look for content between first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start >= 0 and end > start:
                json_text = text[start:end+1]
                try:
                    data = json.loads(json_text)
                    return data
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Could not parse JSON from LLM response: {e}\nResponse: {text[:200]}")
    
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
                
                # Call LLM
                response = self.llm.invoke(prompt)
                
                # Parse response
                # ChatOllama returns a message object, extract content
                response_text = response.content if hasattr(response, 'content') else str(response)
                
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
                    print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"  ⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    print(f"  ❌ All {self.max_retries} attempts failed")
        
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
        
        Processes each chunk sequentially and returns results.
        
        Args:
            chunks: List of text chunks to process
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        print(f"\n🔬 Extracting data from {len(chunks)} chunks...\n")
        
        for i, chunk in enumerate(chunks):
            print(f"[{i+1}/{len(chunks)}] Processing chunk {chunk.chunk_index} ({chunk.section})...")
            
            result = self.extract_from_chunk(chunk)
            results.append(result)
            
            if result.success:
                print(f"  ✅ Extracted {len(result.extracted_data.get('polymers', []))} polymers, "
                      f"{len(result.extracted_data.get('conditions', []))} conditions, "
                      f"{len(result.extracted_data.get('measurements', []))} measurements")
            else:
                print(f"  ❌ Extraction failed: {result.error_message}")
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test if Ollama server is reachable and model is available.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test prompt
            response = self.llm.invoke("Respond with: OK")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def extract_from_pdf(chunks: List[TextChunk], model_name: str = None) -> List[ExtractionResult]:
    """
    Convenience function to extract data from PDF chunks.
    
    Simple wrapper around LLMExtractor for easy use.
    
    Args:
        chunks: List of text chunks from a PDF
        model_name: Optional model override
        
    Returns:
        List of extraction results
    """
    extractor = LLMExtractor(model_name=model_name)
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
    
    # Check Ollama
    print("Testing Ollama connection...")
    extractor = LLMExtractor()
    if not extractor.test_connection():
        print("❌ Ollama not running or model not available")
        print(f"\nStart Ollama and pull model:")
        print(f"  ollama serve")
        print(f"  ollama pull {settings.llm_model}")
        sys.exit(1)
    
    print("✅ Ollama connected\n")
    
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
    results = extract_from_pdf(chunks)
    
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
            print(f"  Polymers: {result.extracted_data.get('polymers', [])}")
            print(f"  Methods: {result.extracted_data.get('methods', [])}")
