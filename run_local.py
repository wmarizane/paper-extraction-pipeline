#!/usr/bin/env python3
"""
End-to-end PDF extraction pipeline.

Takes a PDF, extracts text with GROBID, chunks it, extracts data with LLM,
and saves structured JSON output.

Usage:
    python run_local.py <path_to_pdf>
    python run_local.py Inputs/polymerPaper1.pdf
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config import settings
from pipeline.pdf_parser import check_grobid_server, parse_pdf_with_grobid
from pipeline.chunker import chunk_pdf
from pipeline.llm_extractor import LLMExtractor


class PipelineRunner:
    """
    Orchestrates the end-to-end extraction pipeline.
    
    Tracks progress, handles errors, and produces final JSON output.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the pipeline runner.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = Path(pdf_path)
        self.pdf_name = self.pdf_path.name
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        # Initialize LLM extractor (tests connection)
        self.extractor = LLMExtractor()
        
        # Track metrics
        self.metrics = {
            "pdf_name": self.pdf_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_time_seconds": 0,
            "stages": {},
            "success": False,
            "error": None
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete extraction pipeline.
        
        Returns:
            Dictionary with extracted data and metadata
        """
        pipeline_start = time.time()
        
        try:
            print("\n" + "="*70)
            print("📄 PAPER EXTRACTION PIPELINE")
            print("="*70)
            print(f"PDF: {self.pdf_name}")
            print(f"Model: {settings.llm_model}")
            print(f"Chunk Strategy: {settings.chunk_strategy}")
            print("="*70 + "\n")
            
            # Stage 1: Check prerequisites
            print("🔍 Stage 1/4: Checking prerequisites...")
            self._check_prerequisites()
            print("   ✅ GROBID server running")
            print("   ✅ Ollama server running")
            print()
            
            # Stage 2: Parse PDF with GROBID
            print("📖 Stage 2/4: Parsing PDF with GROBID...")
            stage_start = time.time()
            xml_content = parse_pdf_with_grobid(str(self.pdf_path))
            stage_time = time.time() - stage_start
            self.metrics["stages"]["grobid_parsing"] = {
                "time_seconds": stage_time,
                "success": True
            }
            print(f"   ✅ PDF parsed ({stage_time:.2f}s)")
            print()
            
            # Stage 3: Chunk text
            print("✂️  Stage 3/4: Chunking text...")
            stage_start = time.time()
            chunks = chunk_pdf(xml_content, self.pdf_name)
            stage_time = time.time() - stage_start
            
            # Calculate token statistics
            total_tokens = sum(c.token_count for c in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            self.metrics["stages"]["chunking"] = {
                "time_seconds": stage_time,
                "success": True,
                "num_chunks": len(chunks),
                "total_tokens": total_tokens,
                "avg_tokens_per_chunk": avg_tokens
            }
            
            print(f"   ✅ Created {len(chunks)} chunks ({stage_time:.2f}s)")
            print(f"   📊 Total tokens: {total_tokens:,} | Avg per chunk: {avg_tokens:.0f}")
            
            # Show chunk breakdown
            sections = {}
            for chunk in chunks:
                sections[chunk.section] = sections.get(chunk.section, 0) + 1
            
            print(f"   📑 Sections: {', '.join([f'{s} ({n})' for s, n in sections.items()])}")
            print()
            
            # Stage 4: Extract with LLM
            print("🤖 Stage 4/4: Extracting data with LLM...")
            stage_start = time.time()
            results = self.extractor.extract_from_chunks(chunks)
            stage_time = time.time() - stage_start
            
            # Calculate extraction statistics
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            total_llm_calls = sum(r.llm_calls for r in results)
            avg_time_per_chunk = stage_time / len(results) if results else 0
            
            self.metrics["stages"]["llm_extraction"] = {
                "time_seconds": stage_time,
                "success": successful == len(results),
                "successful_chunks": successful,
                "failed_chunks": failed,
                "total_llm_calls": total_llm_calls,
                "avg_time_per_chunk": avg_time_per_chunk
            }
            
            print(f"\n   ✅ Extraction complete ({stage_time:.2f}s)")
            print(f"   📊 Success: {successful}/{len(results)} | LLM calls: {total_llm_calls} | Avg: {avg_time_per_chunk:.2f}s/chunk")
            print()
            
            # Aggregate extracted data
            print("📦 Aggregating results...")
            final_data = self._aggregate_results(chunks, results)
            
            # Save output
            output_path = self._save_output(final_data)
            
            # Final summary
            self.metrics["success"] = True
            self.metrics["end_time"] = datetime.now().isoformat()
            self.metrics["total_time_seconds"] = time.time() - pipeline_start
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETE")
            print("="*70)
            print(f"Total time: {self.metrics['total_time_seconds']:.2f}s")
            print(f"Output: {output_path}")
            print("="*70 + "\n")
            
            return final_data
            
        except Exception as e:
            self.metrics["success"] = False
            self.metrics["error"] = str(e)
            self.metrics["end_time"] = datetime.now().isoformat()
            self.metrics["total_time_seconds"] = time.time() - pipeline_start
            
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED")
            print("="*70)
            print(f"Error: {e}")
            print("="*70 + "\n")
            
            raise
    
    def _check_prerequisites(self):
        """Check that GROBID and Ollama are running."""
        # Check GROBID
        if not check_grobid_server(settings.grobid_url):
            raise RuntimeError(
                f"GROBID server not running at {settings.grobid_url}. Start it with:\n"
                "  docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0"
            )
        
        # Check Ollama
        if not self.extractor.test_connection():
            raise RuntimeError(
                f"Ollama not running or model '{settings.llm_model}' not available.\n"
                f"Start Ollama and pull model:\n"
                f"  ollama serve\n"
                f"  ollama pull {settings.llm_model}"
            )
    
    def _aggregate_results(self, chunks, results) -> Dict[str, Any]:
        """
        Aggregate extraction results into final output structure.
        
        Args:
            chunks: Original text chunks
            results: Extraction results from LLM
            
        Returns:
            Dictionary with aggregated data and metadata
        """
        # Collect all extracted data
        all_polymers = []
        all_conditions = []
        all_measurements = []
        all_methods = set()
        
        chunk_details = []
        
        for chunk, result in zip(chunks, results):
            chunk_info = {
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "token_count": chunk.token_count,
                "extraction_success": result.success,
                "llm_calls": result.llm_calls,
                "processing_time": result.processing_time
            }
            
            if result.success and result.extracted_data:
                data = result.extracted_data
                
                # Aggregate polymers
                all_polymers.extend(data.get("polymers", []))
                
                # Aggregate conditions
                all_conditions.extend(data.get("conditions", []))
                
                # Aggregate measurements
                all_measurements.extend(data.get("measurements", []))
                
                # Aggregate methods
                all_methods.update(data.get("methods", []))
                
                chunk_info["extracted"] = {
                    "polymers": len(data.get("polymers", [])),
                    "conditions": len(data.get("conditions", [])),
                    "measurements": len(data.get("measurements", [])),
                    "methods": len(data.get("methods", []))
                }
            else:
                chunk_info["error"] = result.error_message
            
            chunk_details.append(chunk_info)
        
        # Build final output
        output = {
            "metadata": {
                "source_pdf": self.pdf_name,
                "extraction_date": datetime.now().isoformat(),
                "model": settings.llm_model,
                "chunk_strategy": settings.chunk_strategy,
                "total_chunks": len(chunks),
                "successful_extractions": sum(1 for r in results if r.success),
                "total_processing_time": self.metrics["total_time_seconds"],
                "pipeline_metrics": self.metrics
            },
            "summary": {
                "total_polymers": len(all_polymers),
                "total_conditions": len(all_conditions),
                "total_measurements": len(all_measurements),
                "methods_used": sorted(list(all_methods))
            },
            "extracted_data": {
                "polymers": all_polymers,
                "conditions": all_conditions,
                "measurements": all_measurements,
                "methods": sorted(list(all_methods))
            },
            "chunk_details": chunk_details
        }
        
        return output
    
    def _save_output(self, data: Dict[str, Any]) -> Path:
        """
        Save extraction results to JSON file.
        
        Args:
            data: Extracted data to save
            
        Returns:
            Path to output file
        """
        # Create output directory
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename: <pdf_name>_extracted_<timestamp>.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.pdf_path.stem
        output_filename = f"{base_name}_extracted_{timestamp}.json"
        output_path = output_dir / output_filename
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved to: {output_path}")
        
        # Also save a "latest" version for easy access
        latest_path = output_dir / f"{base_name}_latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path


def main():
    """Main entry point for the pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python run_local.py <path_to_pdf>")
        print("\nExample:")
        print("  python run_local.py Inputs/polymerPaper1.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        runner = PipelineRunner(pdf_path)
        result = runner.run()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
