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
        Main entry point: return the entire markdown as one chunk if it fits.
        If it exceeds the context window, intelligently split by Markdown headers.
        """
        token_count = self.count_tokens(md_content)
        
        # Max tokens allowed for input. We cap this at 9000 tokens because the 
        # community AWQ quantization degrades and hallucinates EOS tokens on 
        # contexts > 11k tokens.
        max_input_tokens = min(9000, settings.vllm_max_model_len - 2048)
        
        # Fast path: fits in one chunk
        if token_count <= max_input_tokens:
            return [TextChunk(
                text=md_content,
                section="Full Paper",
                chunk_index=0,
                token_count=token_count,
                source_pdf=source_pdf
            )]
            
        # Fallback path: document is too large, split by main Markdown headers
        print(f"⚠️ Document too large ({token_count} > {max_input_tokens}). Using smart fallback chunking.")
        
        chunks = []
        current_text = ""
        current_tokens = 0
        chunk_idx = 0
        
        # Split by H2 headers (standard PyMuPDF4LLM section dividers)
        sections = md_content.split("\n## ")
        
        for i, section in enumerate(sections):
            # Re-add the header prefix if it's not the very first split
            if i > 0:
                section = "\n## " + section
                
            section_tokens = self.count_tokens(section)
            
            # If a single section is too big, we just have to append it and hope it doesn't break vLLM too badly
            # (or we could further split by '\n### ' but usually H2 is fine)
            if current_tokens + section_tokens > max_input_tokens and current_text:
                # Save current chunk
                chunks.append(TextChunk(
                    text=current_text,
                    section=f"Part {chunk_idx + 1}",
                    chunk_index=chunk_idx,
                    token_count=current_tokens,
                    source_pdf=source_pdf
                ))
                # Start new chunk
                chunk_idx += 1
                current_text = section
                current_tokens = section_tokens
            else:
                current_text += section
                current_tokens += section_tokens
                
        # Append the final chunk
        if current_text:
            chunks.append(TextChunk(
                text=current_text,
                section=f"Part {chunk_idx + 1}",
                chunk_index=chunk_idx,
                token_count=current_tokens,
                source_pdf=source_pdf
            ))
            
        return chunks


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