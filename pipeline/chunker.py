"""
Text chunking for LLM consumption.

Takes GROBID TEI XML and splits it into manageable chunks that fit in LLM context windows.
Supports two strategies:
1. Section-based: Split by paper sections (Introduction, Methods, etc.)
2. Paragraph-based: Split by paragraphs when sections aren't available
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import tiktoken

from config import settings


@dataclass
class TextChunk:
    """
    A chunk of text ready for LLM processing.
    
    Attributes:
        text: The actual text content
        section: Section name (e.g., "Introduction", "Methods")
        chunk_index: Position in the document (0, 1, 2, ...)
        token_count: Number of tokens in this chunk
        source_pdf: Original PDF filename
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
    Chunks text from TEI XML for LLM processing.
    
    This is the core logic that decides HOW to split papers into chunks.
    """
    
    def __init__(self, strategy: Literal["section", "paragraph"] = None):
        """
        Initialize the chunker.
        
        Args:
            strategy: How to chunk ("section" or "paragraph"). 
                     If None, uses config.settings.chunk_strategy
        """
        self.strategy = strategy or settings.chunk_strategy
        self.max_tokens = settings.chunk_size
        self.overlap_tokens = settings.chunk_overlap
        
        # Initialize token encoder (same as GPT-4)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # TEI XML namespace
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        tokens = self.encoder.encode(text)
        return len(tokens)
    
    def chunk_tei_xml(self, xml_content: str, source_pdf: str) -> List[TextChunk]:
        """
        Main entry point: chunk TEI XML into text chunks.
        
        Args:
            xml_content: TEI XML string from GROBID
            source_pdf: Name of the source PDF file
            
        Returns:
            List of TextChunk objects
        """
        # Parse XML
        root = ET.fromstring(xml_content)
        
        # Try section-based chunking first
        if self.strategy == "section":
            chunks = self._chunk_by_sections(root, source_pdf)
            if chunks:  # If we found sections, use them
                return chunks
        
        # Fallback to paragraph-based chunking
        return self._chunk_by_paragraphs(root, source_pdf)
    
    def _chunk_by_sections(self, root: ET.Element, source_pdf: str) -> List[TextChunk]:
        """
        Chunk by paper sections (Introduction, Methods, Results, etc.).
        
        GROBID structures papers into <div> elements with <head> tags for section titles.
        We extract each section and create chunks from it.
        """
        chunks = []
        chunk_index = 0
        
        # Find all sections in the body
        sections = root.findall('.//tei:text/tei:body//tei:div', self.ns)
        
        if not sections:
            return []  # No sections found, caller will fallback
        
        for section in sections:
            # Get section name from <head> tag
            head = section.find('.//tei:head', self.ns)
            if head is not None and head.text:
                section_name = head.text.strip()
            else:
                section_name = "Unknown Section"
            
            # Extract all paragraphs in this section
            paragraphs = section.findall('.//tei:p', self.ns)
            
            # Combine paragraphs into section text
            section_text = "\n\n".join([
                p.text.strip() for p in paragraphs 
                if p.text and p.text.strip()
            ])
            
            if not section_text:
                continue  # Skip empty sections
            
            # Add section header to text
            full_text = f"{section_name}\n\n{section_text}"
            
            # Count tokens
            token_count = self.count_tokens(full_text)
            
            # If section fits in one chunk, add it
            if token_count <= self.max_tokens:
                chunks.append(TextChunk(
                    text=full_text,
                    section=section_name,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    source_pdf=source_pdf
                ))
                chunk_index += 1
            else:
                # Section is too big - split by paragraphs within section
                section_chunks = self._split_large_section(
                    section_name, paragraphs, source_pdf, chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        return chunks
    
    def _split_large_section(
        self, 
        section_name: str, 
        paragraphs: List[ET.Element],
        source_pdf: str,
        starting_index: int
    ) -> List[TextChunk]:
        """
        Split a large section into multiple chunks by paragraphs.
        
        When a section is too big (>max_tokens), we need to split it.
        We group paragraphs together until we hit the token limit.
        """
        chunks = []
        current_chunk_text = []
        current_tokens = 0
        chunk_index = starting_index
        
        for para in paragraphs:
            if not para.text or not para.text.strip():
                continue
            
            para_text = para.text.strip()
            para_tokens = self.count_tokens(para_text)
            
            # If this paragraph alone is too big, we have a problem
            # (shouldn't happen in practice, but handle it)
            if para_tokens > self.max_tokens:
                # Truncate the paragraph (or could split by sentences)
                truncated = self._truncate_to_tokens(para_text, self.max_tokens)
                chunks.append(TextChunk(
                    text=f"{section_name}\n\n{truncated}",
                    section=section_name,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(truncated),
                    source_pdf=source_pdf
                ))
                chunk_index += 1
                continue
            
            # If adding this paragraph would exceed limit, save current chunk
            if current_tokens + para_tokens > self.max_tokens and current_chunk_text:
                chunk_text = f"{section_name}\n\n" + "\n\n".join(current_chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    section=section_name,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    source_pdf=source_pdf
                ))
                chunk_index += 1
                current_chunk_text = []
                current_tokens = 0
            
            # Add paragraph to current chunk
            current_chunk_text.append(para_text)
            current_tokens += para_tokens
        
        # Don't forget the last chunk!
        if current_chunk_text:
            chunk_text = f"{section_name}\n\n" + "\n\n".join(current_chunk_text)
            chunks.append(TextChunk(
                text=chunk_text,
                section=section_name,
                chunk_index=chunk_index,
                token_count=current_tokens,
                source_pdf=source_pdf
            ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, root: ET.Element, source_pdf: str) -> List[TextChunk]:
        """
        Fallback: chunk by paragraphs when sections aren't available.
        
        This is simpler - just group paragraphs until we hit the token limit.
        """
        chunks = []
        chunk_index = 0
        
        # Extract all paragraphs from body
        paragraphs = root.findall('.//tei:text/tei:body//tei:p', self.ns)
        
        current_chunk_text = []
        current_tokens = 0
        
        for para in paragraphs:
            if not para.text or not para.text.strip():
                continue
            
            para_text = para.text.strip()
            para_tokens = self.count_tokens(para_text)
            
            # If adding this paragraph exceeds limit, save current chunk
            if current_tokens + para_tokens > self.max_tokens and current_chunk_text:
                chunk_text = "\n\n".join(current_chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    section="Body",  # Generic section name
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    source_pdf=source_pdf
                ))
                chunk_index += 1
                current_chunk_text = []
                current_tokens = 0
            
            current_chunk_text.append(para_text)
            current_tokens += para_tokens
        
        # Save final chunk
        if current_chunk_text:
            chunk_text = "\n\n".join(current_chunk_text)
            chunks.append(TextChunk(
                text=chunk_text,
                section="Body",
                chunk_index=chunk_index,
                token_count=current_tokens,
                source_pdf=source_pdf
            ))
        
        return chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)


# Convenience function for simple usage
def chunk_pdf(xml_content: str, pdf_filename: str) -> List[TextChunk]:
    """
    Chunk a PDF's TEI XML into text chunks.
    
    Simple wrapper around TextChunker for easy use.
    
    Args:
        xml_content: TEI XML string from GROBID
        pdf_filename: Name of the PDF file
        
    Returns:
        List of TextChunk objects
    """
    chunker = TextChunker()
    return chunker.chunk_tei_xml(xml_content, pdf_filename)


if __name__ == "__main__":
    """Test the chunker on a sample PDF."""
    import sys
    from pipeline.pdf_parser import parse_pdf_with_grobid, check_grobid_server
    
    print("Testing Text Chunker\n")
    
    # Check GROBID is running
    if not check_grobid_server():
        print("❌ GROBID server not running. Start it with:")
        print("   docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        sys.exit(1)
    
    # Test on a sample PDF
    test_pdf = "Inputs/polymerPaper1.pdf"
    
    if not Path(test_pdf).exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        sys.exit(1)
    
    print(f"Parsing {test_pdf} with GROBID...")
    xml_content = parse_pdf_with_grobid(test_pdf)
    
    print(f"Chunking text...")
    chunks = chunk_pdf(xml_content, "polymerPaper1.pdf")
    
    print(f"\n✅ Created {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  Section: {chunk.section}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Preview: {chunk.text[:100]}...")
        print()