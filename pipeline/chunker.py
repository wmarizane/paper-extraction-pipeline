"""
Text chunking for LLM consumption.

Refactored to treat the entire Markdown document as a single chunk 
to leverage large context windows in modern LLMs (Qwen 3.5 27B/35B) 
and avoid contextual amnesia.
"""

from dataclasses import dataclass
from typing import List
import tiktoken
from config.settings import settings


@dataclass
class TextChunk:
    """
    A chunk of text ready for LLM processing.
    """
    text: str
    section: str
    chunk_index: int
    token_count: int
    source_pdf: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "source_pdf": self.source_pdf
        }


class TextChunker:
    """
    Wraps the full markdown document into a single chunk.
    """
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        tokens = self.encoder.encode(text)
        return len(tokens)
    
    def process_markdown(self, md_content: str, source_pdf: str) -> List[TextChunk]:
        """
        Main entry point: return the entire markdown as one chunk.
        """
        token_count = self.count_tokens(md_content)
        
        # We pass the full text; if for some miraculous reason it exceeds Qwen's context 
        # (32k), vLLM might complain, but normal papers are 5k-15k tokens.
        chunk = TextChunk(
            text=md_content,
            section="Full Paper",
            chunk_index=0,
            token_count=token_count,
            source_pdf=source_pdf
        )
        
        return [chunk]


# Convenience function for simple usage
def chunk_pdf(md_content: str, pdf_filename: str) -> List[TextChunk]:
    """
    Wrap the parsed PDF markdown into a single TextChunk list 
    to preserve existing pipeline API compatibility.
    
    Args:
        md_content: Markdown string from PyMuPDF4LLM
        pdf_filename: Name of the PDF file
        
    Returns:
        List of TextChunk objects (List length = 1)
    """
    chunker = TextChunker()
    return chunker.process_markdown(md_content, pdf_filename)


if __name__ == "__main__":
    """Test the chunker on a sample PDF."""
    import sys
    from pipeline.pdf_parser import parse_pdf_to_markdown
    
    test_pdf = "Inputs/polymerPaper1.pdf"
    
    print(f"Parsing {test_pdf} Native...")
    md_content = parse_pdf_to_markdown(test_pdf)
    
    print(f"Chunking text...")
    chunks = chunk_pdf(md_content, "polymerPaper1.pdf")
    
    print(f"\n✅ Created {len(chunks)} chunk(s):\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  Section: {chunk.section}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Preview: {chunk.text[:200]}...")
        print()