"""
Phase 1: Document Ingestion & Preprocessing
======================================================
Handles the loading of parsed Markdown files and segments them into 
context-aware chunks for the Triple Extraction phase.
"""

import os
import re
from typing import List, Dict, Tuple, Set

TOKEN_LIMIT = 4096
INSTRUCTION_TOKEN_ESTIMATE = 200
CHAR_TO_TOKEN_RATIO = 3.5

class MarkdownChunker:
    """
    Splits Markdown content into chunks while preserving header context.
    Strategy:
    1. Accumulate text across multiple small sections until reaching optimal size.
    2. Only create chunks at major section boundaries (H1/H2) or when size limit reached.
    3. Prepend parent headers to each chunk to maintain context.
    4. Use token-based limits (aligned with AutoSchemaKG framework)
    """
    
    def __init__(self, 
                 token_limit: int = TOKEN_LIMIT, 
                 instruction_tokens: int = INSTRUCTION_TOKEN_ESTIMATE,
                 char_ratio: float = CHAR_TO_TOKEN_RATIO,
                 deduplicate: bool = True,
                 min_chunk_size: int = 8000): 
        self.token_limit = token_limit
        self.instruction_tokens = instruction_tokens
        self.char_ratio = char_ratio
        self.deduplicate = deduplicate
        
        available_tokens = token_limit - instruction_tokens
        self.max_chunk_size = int(available_tokens * char_ratio)
        self.min_chunk_size = min_chunk_size  
        
        self.seen_chunks: Set[str] = set()
        
        self.accumulated_sections: List[Tuple[List[Tuple[int, str]], str]] = []  
        self.accumulated_length = 0

    def chunk_file(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._parse_markdown(content)

    def _parse_markdown(self, content: str) -> List[str]:
        lines = content.split('\n')
        chunks = []
        
        current_headers = [] 
        current_text_buffer = []
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            
            if header_match:
                if current_text_buffer:
                    section_text = "\n".join(current_text_buffer).strip()
                    if section_text:
                        self.accumulated_sections.append((list(current_headers), section_text))
                        self.accumulated_length += len(section_text)
                    current_text_buffer = []
                
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                is_major_boundary = level <= 2
                
                if (is_major_boundary and self.accumulated_length >= self.min_chunk_size) or \
                   (self.accumulated_length >= self.max_chunk_size * 0.9):
                    self._flush_accumulated_sections(chunks)
                
                while current_headers and current_headers[-1][0] >= level:
                    current_headers.pop()
                
                current_headers.append((level, title))
            else:
                current_text_buffer.append(line)
        
        if current_text_buffer:
            section_text = "\n".join(current_text_buffer).strip()
            if section_text:
                self.accumulated_sections.append((list(current_headers), section_text))
                self.accumulated_length += len(section_text)
        
        if self.accumulated_sections:
            self._flush_accumulated_sections(chunks)
            
        return chunks
    
    def _flush_accumulated_sections(self, chunks: List[str]):
        """Flush accumulated sections into one or more chunks."""
        if not self.accumulated_sections:
            return
        
        combined_text = []
        context_headers = self.accumulated_sections[0][0]
        
        for headers, text in self.accumulated_sections:
            section_title = " > ".join([h[1] for h in headers])
            combined_text.append(f"[{section_title}]\n{text}")
        
        full_text = "\n\n".join(combined_text)
        context_str = " > ".join([h[1] for h in context_headers])
        
        if len(full_text) <= self.max_chunk_size:
            final_chunk = f"Context: {context_str}\n\nContent:\n{full_text}"
            self._add_chunk_with_dedup(chunks, final_chunk)
        else:
            current_sub_sections = []
            current_length = 0
            
            for headers, text in self.accumulated_sections:
                section_title = " > ".join([h[1] for h in headers])
                section_with_title = f"[{section_title}]\n{text}"
                
                if current_length + len(section_with_title) > self.max_chunk_size and current_sub_sections:
                    body = "\n\n".join(current_sub_sections)
                    final_chunk = f"Context: {context_str}\n\nContent:\n{body}"
                    self._add_chunk_with_dedup(chunks, final_chunk)
                    current_sub_sections = []
                    current_length = 0
                
                current_sub_sections.append(section_with_title)
                current_length += len(section_with_title)
            
            if current_sub_sections:
                body = "\n\n".join(current_sub_sections)
                final_chunk = f"Context: {context_str}\n\nContent:\n{body}"
                self._add_chunk_with_dedup(chunks, final_chunk)
        
        self.accumulated_sections = []
        self.accumulated_length = 0

    def _add_chunk_with_dedup(self, chunks: List[str], chunk: str):
        """Add chunk to list with optional deduplication."""
        if self.deduplicate:
            if chunk in self.seen_chunks:
                return  # Skip duplicate
            self.seen_chunks.add(chunk)
        chunks.append(chunk)


def load_and_segment_text(file_path: str, deduplicate: bool = True) -> List[Dict[str, any]]:
    """
    Load a medical text document (Markdown or Text) and segment it.
    Returns structured chunks with metadata (aligned with AutoSchemaKG format).
    
    Args:
        file_path (str): Path to the file (can be .md or .txt)
        deduplicate (bool): Whether to remove duplicate chunks
        
    Returns:
        List[Dict]: List of chunk dictionaries with 'id', 'text', 'chunk_id', 'metadata'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print(f"  Loading file: {file_path}")
    
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    
    if file_path.endswith('.md'):
        chunker = MarkdownChunker(deduplicate=deduplicate)
        raw_segments = chunker.chunk_file(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        raw_segments = [s.strip() for s in content.split('\n\n') if s.strip()]
    
    raw_segments = [s for s in raw_segments if len(s) > 50]
    
    structured_chunks = []
    for idx, text in enumerate(raw_segments):
        chunk = {
            "id": doc_id,
            "text": text,
            "chunk_id": idx,
            "metadata": {
                "source_file": file_path,
                "total_chunks": len(raw_segments)
            }
        }
        structured_chunks.append(chunk)
    
    return structured_chunks

if __name__ == "__main__":
    import sys
    import json
    
    test_file = "data/parsed/ACP_Home_Guide_content.md"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
    print(f"Testing Phase 1 Ingestion on: {test_file}")
    print(f"Token-based chunking: max ~{int((TOKEN_LIMIT - INSTRUCTION_TOKEN_ESTIMATE) * CHAR_TO_TOKEN_RATIO)} chars/chunk")
    
    try:
        chunks = load_and_segment_text(test_file)
        print(f"\n✅ Successfully created {len(chunks)} chunks.")
        
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chunk size: {avg_chars:.0f} chars")
        
        print("\n--- Sample Chunk 1 ---")
        if len(chunks) > 0:
            print(json.dumps(chunks[0], indent=2, ensure_ascii=False)[:500])
            
        print("\n--- Sample Chunk 2 ---")
        if len(chunks) > 1:
            print(json.dumps(chunks[1], indent=2, ensure_ascii=False)[:500])
            
        print("\n--- Sample Chunk 10 ---")
        if len(chunks) > 10:
            print(json.dumps(chunks[10], indent=2, ensure_ascii=False)[:500])
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()