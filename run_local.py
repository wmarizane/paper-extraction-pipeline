#!/usr/bin/env python3
"""
End-to-end PDF extraction pipeline.

Takes a PDF, natively extracts Markdown (preserving tables), wraps it as a Chunk,
extracts data via vLLM, and saves structured JSON.

Usage:
    python run_local.py <path_to_pdf>
    python run_local.py Inputs/polymerPaper1.pdf --model qwen3.5-27b
    python run_local.py Inputs/polymerPaper1.pdf --model deepseek-r1-14b
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config.settings import settings
from config.model_registry import get_model_config, list_models
from pipeline.pdf_parser import parse_pdf_to_markdown, check_parser_ready
from pipeline.chunker import chunk_pdf
from pipeline.llm_extractor import LLMExtractor

class PipelineRunner:
    """Orchestrates the end-to-end extraction pipeline."""
    
    def __init__(self, pdf_path: str, model_name: str = None):
        self.pdf_path = Path(pdf_path)
        self.pdf_name = self.pdf_path.name
        self.model_name = model_name
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        # Initialize LLM with optional model override
        self.extractor = LLMExtractor(model_name=model_name)
        
        # Metrics
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
        pipeline_start = time.time()
        
        try:
            print("\n" + "="*70)
            print("📄 PAPER EXTRACTION PIPELINE (PHASE 1 REFACTOR)")
            print("="*70)
            print(f"PDF: {self.pdf_name}")
            print(f"Model: {settings.llm_model}")
            print("="*70 + "\n")
            
            # Stage 1
            print("🔍 Stage 1/4: Checking prerequisites...")
            if not check_parser_ready():
                 raise RuntimeError("PyMuPDF4LLM is not loaded correctly.")
            print("   ✅ PyMuPDF4LLM ready")
            print("   ✅ LLM extractor ready\n")
            
            # Stage 2
            print("📖 Stage 2/4: Parsing PDF natively to Markdown...")
            stage_start = time.time()
            md_content = parse_pdf_to_markdown(str(self.pdf_path))
            stage_time = time.time() - stage_start
            
            self.metrics["stages"]["parsing"] = {
                "time_seconds": stage_time,
                "success": True,
            }
            print(f"   ✅ PDF parsed to Markdown ({stage_time:.2f}s)\n")
            
            # Stage 3
            print("✂️  Stage 3/4: Creating text chunk...")
            stage_start = time.time()
            chunks = chunk_pdf(md_content, self.pdf_name)
            stage_time = time.time() - stage_start
            
            total_tokens = sum(c.token_count for c in chunks)
            
            self.metrics["stages"]["chunking"] = {
                "time_seconds": stage_time,
                "success": True,
                "num_chunks": len(chunks),
                "total_tokens": total_tokens
            }
            
            print(f"   ✅ Wrapped full document: {total_tokens:,} tokens ({stage_time:.2f}s)\n")
            
            # Stage 4
            print("🤖 Stage 4/4: Extracting data with LLM...")
            stage_start = time.time()
            results = self.extractor.extract_from_chunks(chunks)
            stage_time = time.time() - stage_start
            
            successful = sum(1 for r in results if r.success)
            self.metrics["stages"]["llm_extraction"] = {
                "time_seconds": stage_time,
                "success": successful == len(results),
            }
            
            print(f"\n   ✅ Extraction complete ({stage_time:.2f}s)\n")
            
            print("📦 Aggregating results...")
            final_data = self._aggregate_results(chunks, results)
            output_path = self._save_output(final_data)
            
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
            
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED")
            print("="*70)
            print(f"Error: {e}")
            print("="*70 + "\n")
            raise
    
    def _aggregate_results(self, chunks, results) -> Dict[str, Any]:
        all_conditions = []
        chunk_details = []
        
        for chunk, result in zip(chunks, results):
            chunk_info = {
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "token_count": chunk.token_count,
                "extraction_success": result.success,
                "processing_time": result.processing_time
            }
            
            if result.success and result.extracted_data:
                data = result.extracted_data
                all_conditions.extend(data.get("extracted_conditions", []))
                chunk_info["extracted_count"] = len(data.get("extracted_conditions", []))
            else:
                chunk_info["error"] = result.error_message
            
            chunk_details.append(chunk_info)
        
        output = {
            "metadata": {
                "source_pdf": self.pdf_name,
                "extraction_date": datetime.now().isoformat(),
                "model": settings.llm_model,
                "pipeline_metrics": self.metrics
            },
            "summary": {
                "total_conditions": len(all_conditions),
            },
            "extracted_data": {
                "conditions": all_conditions,
            },
            "chunk_details": chunk_details
        }
        
        return output
    
    def _save_output(self, data: Dict[str, Any]) -> Path:
        # Create a model-specific subdirectory (e.g., results/qwen3.5-27b/)
        model_key = self.model_name or settings.llm_model
        output_dir = Path(settings.output_dir) / model_key
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.pdf_path.stem
        output_path = output_dir / f"{base_name}_extracted_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        latest_path = output_dir / f"{base_name}_latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved to: {latest_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Extract LCCC conditions from polymer papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use (e.g., 'qwen3.5-27b', 'deepseek-r1-14b'). "
             "Defaults to LLM_MODEL in .env"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)
    
    try:
        runner = PipelineRunner(args.pdf_path, model_name=args.model)
        runner.run()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
