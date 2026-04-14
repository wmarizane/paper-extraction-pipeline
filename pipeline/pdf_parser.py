"""Native PDF Parsing using PyMuPDF4LLM."""

import json
from pathlib import Path
import pymupdf4llm

import logging

logger = logging.getLogger(__name__)

def check_parser_ready() -> bool:
    """Check if the local parser is ready (PyMuPDF4LLM requires no server)."""
    return True

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parses a PDF natively to Markdown, preserving tabular structures.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
    logger.info(f"Parsing PDF to markdown natively: {pdf_path}")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

def save_markdown(md_content: str, pdf_path: str, output_dir: str = "parsed_md") -> Path:
    """
    Save markdown content for later inspection.
    """
    md_dir = Path(output_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(pdf_path).stem
    md_path = md_dir / f"{pdf_stem}.md"

    md_path.write_text(md_content, encoding="utf-8")
    return md_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)
        
    target_pdf = sys.argv[1]
    
    print(f"Parsing {target_pdf} directly to markdown...")
    try:
        md_result = parse_pdf_to_markdown(target_pdf)
        out_path = save_markdown(md_result, target_pdf)
        
        print(f"Saved Markdown: {out_path}")
        print("\n--- Extraction Preview ---")
        print(md_result[:1000] + "\n...\n")
        print("--- End of Preview ---")
        
    except Exception as e:
        print(f"Failed: {e}")
