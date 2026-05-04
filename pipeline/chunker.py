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
    
    def _recursive_split(self, text: str, max_tokens: int, level: int = 1) -> List[str]:
        """Recursively splits text until all chunks are <= max_tokens."""
        if self.count_tokens(text) <= max_tokens:
            return [text]
            
        # Define split strategies by level
        if level == 1:
            delimiter = "\n## "
        elif level == 2:
            delimiter = "\n### "
        elif level == 3:
            delimiter = "\n\n"
        else:
            # Level 4: Hard split by characters as a last resort
            hard_limit = max_tokens * 3  # safe approximation
            return [text[i:i+hard_limit] for i in range(0, len(text), hard_limit)]
            
        sections = text.split(delimiter)
        result_chunks = []
        current_chunk = ""
        
        for i, section in enumerate(sections):
            # Re-add delimiter if not first
            if i > 0 and level in [1, 2]:
                section = delimiter + section
            elif i > 0 and level == 3:
                section = "\n\n" + section
                
            section_tokens = self.count_tokens(section)
            
            if section_tokens > max_tokens:
                # Save current if exists
                if current_chunk:
                    result_chunks.append(current_chunk)
                    current_chunk = ""
                # Recursively split this oversized section
                result_chunks.extend(self._recursive_split(section, max_tokens, level + 1))
            else:
                if self.count_tokens(current_chunk + section) > max_tokens:
                    result_chunks.append(current_chunk)
                    current_chunk = section
                else:
                    current_chunk += section
                    
        if current_chunk:
            result_chunks.append(current_chunk)
            
        return result_chunks

    def process_markdown(self, md_content: str, source_pdf: str) -> List[TextChunk]:
        """
        Main entry point: return the entire markdown as one chunk if it fits.
        If it exceeds the context window, use Recursive Splitting and Contextual Retrieval.
        """
        token_count = self.count_tokens(md_content)
        
        # Max tokens allowed for input (cap at 9000 to prevent OOM/hallucinations)
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
            
        # Fallback path: Document too large. 
        print(f"⚠️ Document too large ({token_count} > {max_input_tokens}). Using Contextual Map-Reduce chunking.")
        
        # Extract Global Anchor (First 500 tokens, typically the Abstract)
        first_section = md_content.split("\n## ")[0]
        anchor_tokens = self.encoder.encode(first_section)[:500]
        global_anchor = self.encoder.decode(anchor_tokens)
        
        # Determine remaining budget for the body
        body_max_tokens = max_input_tokens - self.count_tokens(global_anchor) - 50 # 50 for formatting buffer
        
        # Recursively split the document
        raw_chunks = self._recursive_split(md_content, body_max_tokens)
        
        chunks = []
        for i, raw_text in enumerate(raw_chunks):
            # Contextual Retrieval: Prepend the global anchor to subsequent chunks
            if i == 0 or global_anchor in raw_text:
                final_text = raw_text
            else:
                final_text = f"--- GLOBAL CONTEXT (ABSTRACT/INTRO) ---\n{global_anchor}\n--- END GLOBAL CONTEXT ---\n\n{raw_text}"
                
            chunks.append(TextChunk(
                text=final_text,
                section=f"Part {i + 1}",
                chunk_index=i,
                token_count=self.count_tokens(final_text),
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