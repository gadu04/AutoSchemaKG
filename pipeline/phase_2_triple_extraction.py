"""
Phase 2: Triple Extraction (CORE MODULE)
=========================================
This module extracts (Head, Relation, Tail) triples from medical text segments.
OPTIMIZED: Uses Multi-threading + TQDM Progress Bar for monitoring.

TRIPLE TYPES:
1. Entity-Entity (E-E): Links two static entities
   Example: (Metformin, treats, Type 2 Diabetes)

2. Entity-Event (E-Ev): Links an entity to a happening/event
   Example: (Patient, participated_in, [Event: Clinical Trial X])

3. Event-Event (Ev-Ev): Links two happenings/events
   Example: ([Event: Initial Diagnosis], led_to, [Event: Surgical Intervention])
"""

from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_api.interface import call_llm_for_triples
from tqdm import tqdm 
import os
import json
from datetime import datetime
from pathlib import Path

# CẤU HÌNH SỐ LUỒNG
# Với RTX 5080 16GB + Llama 3 8B, mức 9-10 là tối ưu.
MAX_WORKERS = 10

# Replace hard-coded path with env-configurable path
LOG_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
_PHASE2_API_LOG = Path(os.getenv("PHASE2_API_LOG", str(LOG_DIR / "phase2_llm_calls.jsonl")))
_PHASE2_API_LOG.parent.mkdir(parents=True, exist_ok=True)


def _append_phase2_api_log(entry: dict):
    """Append one JSON line to output/phase2_llm_calls.jsonl"""
    try:
        with open(_PHASE2_API_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  ⚠ Warning: failed to write phase2 api log: {e}")


class TripleExtractor:
    """
    Manages the triple extraction process from text segments.
    """
    
    def __init__(self, use_real_llm: bool = False):
        """
        Initialize the TripleExtractor.
        """
        self.use_real_llm = use_real_llm
        self.all_triples = []
        self.unique_nodes = set()
    
    def extract_from_segments(self, text_segments: List[Dict]) -> Tuple[List[Dict], Set[str]]:
        """
        Extract triples from all text segments using Parallel Processing with Progress Bar.
        """
        self.all_triples = []
        self.unique_nodes = set()
        
        total_segments = len(text_segments)
        print(f"  🚀 Starting Parallel Extraction on {total_segments} segments with {MAX_WORKERS} threads...")
        print(f"  ⚡ GPU Utilization target: MAX POWER")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._process_single_segment, segment, idx): idx 
                for idx, segment in enumerate(text_segments, 1)
            }
            
            with tqdm(total=total_segments, desc="  Processing", unit="seg", ncols=100) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    
                    try:
                        segment_triples = future.result()
                        
                        if segment_triples:
                            self.all_triples.extend(segment_triples)
                            pbar.set_postfix_str(f"Found {len(segment_triples)} triples", refresh=False)
                        pbar.update(1)
                        
                    except Exception as e:
                        pbar.write(f"  ✗ Error in segment {idx}: {e}")
                        pbar.update(1)

        print(f"\n  ✅ Parallel extraction complete! Processed {total_segments} segments.")

        print("  Aggregating unique nodes...", end="\r")
        for triple in self.all_triples:
            self.unique_nodes.add(triple['head'])
            self.unique_nodes.add(triple['tail'])
            
        return self.all_triples, self.unique_nodes

    def _process_single_segment(self, segment, idx) -> List[Dict]:
        """
        Helper function to process a single segment (runs inside a thread).
        """
        if isinstance(segment, dict):
            text = segment.get('text', '')
            chunk_id = segment.get('chunk_id', idx)
            doc_id = segment.get('id', 'unknown')
        else:
            text = segment
            chunk_id = idx
            doc_id = 'unknown'
            
        try:
            triples_data = call_llm_for_triples(text, use_real_llm=self.use_real_llm)
            status = "success"
            response_serialized = triples_data  
        except Exception as e:
            triples_data = []
            status = "error"
            response_serialized = {"error": str(e)}
        
        try:
            log_entry = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input": text[:5000],            
                "response": response_serialized,  
                "status": status
            }
            _append_phase2_api_log(log_entry)
        except Exception as e:
            print(f"  ⚠ Warning: failed to prepare phase2 log entry: {e}")
        
        return self._process_triple_response(triples_data, segment_id=chunk_id, doc_id=doc_id)
    
    def _process_triple_response(self, triples_data: Dict, segment_id: int, doc_id: str = 'unknown') -> List[Dict]:
        """
        Process the LLM response and extract structured triples.
        FIXED: Handles None/null values from LLM safely.
        """
        processed_triples = []
        
        def add_triple(t_data, t_type, h_type, t_type_str):
            head = (t_data.get('head') or '').strip()
            relation = (t_data.get('relation') or '').strip()
            tail = (t_data.get('tail') or '').strip()

            if head and relation and tail:
                processed_triples.append({
                    'type': t_type,
                    'head': head,
                    'relation': relation,
                    'tail': tail,
                    'head_type': h_type,
                    'tail_type': t_type_str,
                    'segment_id': segment_id,
                    'doc_id': doc_id,
                    'confidence': t_data.get('confidence', 1.0)
                })


        # Process Entity-Entity triples
        for triple in triples_data.get('entity_entity', []):
            add_triple(triple, 'E-E', 'entity', 'entity')
        
        # Process Entity-Event triples
        for triple in triples_data.get('entity_event', []):
            add_triple(triple, 'E-Ev', 'entity', 'event')
        
        # Process Event-Event triples
        for triple in triples_data.get('event_event', []):
            add_triple(triple, 'Ev-Ev', 'event', 'event')
        
        return processed_triples
    
    def get_triples_by_type(self, triple_type: str) -> List[Dict]:
        """Get all triples of a specific type."""
        return [t for t in self.all_triples if t['type'] == triple_type]
    
    def get_node_statistics(self) -> Dict:
        """Get statistics about extracted nodes."""
        entity_nodes = set()
        event_nodes = set()
        
        for triple in self.all_triples:
            if triple['head_type'] == 'entity':
                entity_nodes.add(triple['head'])
            else:
                event_nodes.add(triple['head'])
            
            if triple['tail_type'] == 'entity':
                entity_nodes.add(triple['tail'])
            else:
                event_nodes.add(triple['tail'])
        
        return {
            'total_nodes': len(self.unique_nodes),
            'entity_nodes': len(entity_nodes),
            'event_nodes': len(event_nodes),
            'entity_list': sorted(list(entity_nodes)),
            'event_list': sorted(list(event_nodes))
        }
    
    def canonicalize_event(text: str) -> str:
        """
        Normalize event names into a consistent representation.
        Ensures all events have the prefix [Event: ...]
        """
        if not text:
            return ""

        t = text.strip()

        if t.startswith("[Event:") and t.endswith("]"):
            return t
        
        t = t.strip('"').strip("'")
        
        return f"[Event: {t}]"
