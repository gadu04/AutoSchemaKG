"""
Phase 3: Hybrid Schema Induction & Ontology Grounding
======================================================
This module performs two tasks:
Part 3a (CORE MODULE): Dynamic concept induction using LLM
Part 3b (UMLS-INTEGRATED): Ontology grounding using UMLS API
"""

import os
import json
import re 
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from llm_api.interface import call_llm_for_concepts

try:
    from pipeline.umls_loader import UMLSLoader
except ImportError:
    UMLSLoader = None


# LOGGING CONFIGURATION
LOG_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PHASE3_LOG_FILE = Path(os.getenv("PHASE3_LOG_FILE", str(LOG_DIR / "phase3_induction_grounding.log")))

def _init_phase3_log():
    """Initialize Phase 3 log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PHASE 3: SCHEMA INDUCTION & ONTOLOGY GROUNDING LOG\n")
        f.write("=" * 100 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

def _log_concept_induction(node_name: str, concept_phrases: str, batch_num: int = None):
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[CONCEPT INDUCTION] (Batch {batch_num})\n")
            f.write(f"  Node: {node_name}\n")
            f.write(f"  Induced Concepts: {concept_phrases}\n")
            f.write("-" * 100 + "\n")
    except Exception: pass

def _log_grounding_result(node_name: str, clean_name: str, grounded_data: Dict):
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[GROUNDING RESULT]\n")
            f.write(f"  Original: {node_name}\n")
            f.write(f"  Clean: {clean_name}\n")
            f.write(f"  ID: {grounded_data.get('ontology_id')}\n")
            f.write("-" * 100 + "\n")
    except Exception: pass

def _log_phase3_summary(induced_count, grounded_count, concept_stats, grounding_stats):
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\nPHASE 3 SUMMARY\n" + "="*50 + "\n")
            f.write(f"Total Grounded: {grounded_count}\n")
    except Exception: pass

def _print_phase3_progress(stage: str, current: int, total: int) -> None:
    pass


# CORE FUNCTIONS
def dynamically_induce_concepts(unique_nodes: Set[str], all_triples: List[Dict] = None, use_real_llm: bool = False) -> Dict[str, str]:
    """
    Part 3a: Dynamically induce abstract concepts for each node.
    """
    _init_phase3_log()
    
    node_list = sorted(list(unique_nodes))
    total_nodes = len(node_list)
    print(f"  Analyzing {total_nodes} unique nodes...")
    
    induced_concepts = {}
    batch_num = 0
    BATCH_SIZE = 50 
    
    for i in range(0, total_nodes, BATCH_SIZE):
        batch_num += 1
        batch_nodes = node_list[i : i + BATCH_SIZE]
        print(f"    Processing batch {batch_num} ({len(batch_nodes)} nodes)...")
        
        try:
            batch_concepts = call_llm_for_concepts(
                batch_nodes, use_real_llm=use_real_llm, triples_list=all_triples 
            )
            for node, concept in batch_concepts.items():
                _log_concept_induction(node, concept, batch_num)
            induced_concepts.update(batch_concepts)
            
        except Exception as e:
            print(f"    ⚠ Batch {batch_num} failed: {e}")
            for node in batch_nodes:
                induced_concepts[node] = "medical concept"
    
    return induced_concepts


def ground_concepts_to_ontology(induced_concepts: Dict[str, str], use_umls: bool = True) -> Dict[str, Dict]:
    """
    Part 3b: Ground induced concepts to ontologies.
    UPDATED: Fix lỗi NameError và hiển thị [EVT]/[ENT].
    """
    print("  Initializing ontology grounding...")
    
    grounded_nodes = {}
    grounding_stats = {}
    
    umls_loader = None
    if use_umls and UMLSLoader is not None:
        try:
            umls_loader = UMLSLoader()
            if umls_loader.is_available():
                print(f"  ✓ UMLS API: Ready")
            else:
                print(f"  ⚠ UMLS API: Not available (Fallback)")
                umls_loader = None
        except Exception as e:
            print(f"  ✗ UMLS API Error: {e}")
            umls_loader = None
    else:
        print(f"  ℹ UMLS grounding disabled")
    
    print(f"  Grounding {len(induced_concepts)} concepts...\n")
    
    total_to_ground = len(induced_concepts)
    processed = 0
    
    for node_name, concept_phrases in induced_concepts.items():
        node_type_label = "ENT"
        if "Event" in node_name or "[Event:" in node_name: 
            node_type_label = "EVT"
        
        clean_node_name = _clean_node_text(node_name)
        search_term = clean_node_name 

        grounded_data = {
            'induced_concept': concept_phrases,
            'semantic_type': _infer_semantic_type(concept_phrases),
            'original_node': node_name,
            'clean_node': clean_node_name,
            'node_type': "Event" if node_type_label == "EVT" else "Entity"
        }
        
        if umls_loader and umls_loader.is_available():
            try:
                all_results = umls_loader.search_concept(search_term)
                
                umls_match = None
                umls_alternatives = []

                if all_results:
                    best = all_results[0]
                    if best['score'] >= 0.3:
                        umls_match = best
                    if len(all_results) > 1:
                        umls_alternatives = [r for r in all_results[1:4] if r['score'] >= 0.2]

                if umls_match:
                    cui = umls_match['umls_id']
                    src = umls_match['source'] or 'UMLS'
                    grounded_data.update({
                        'ontology_id': cui,
                        'ontology_name': src,
                        'umls_id': cui,
                        'match_score': umls_match['score'],
                        'label': umls_match['name'],
                        'source': 'umls_api',
                        'alternative_matches': umls_alternatives
                    })
                    print(f"  [{processed+1}/{total_to_ground}] [{node_type_label}] ✓ '{clean_node_name}' -> {cui} ({src})")
                else:
                    grounded_data.update(_create_fallback_data(clean_node_name, "fallback"))
                    if len(clean_node_name.split()) <= 3:
                        print(f"  [{processed+1}/{total_to_ground}] [{node_type_label}] ⚠ '{clean_node_name}' -> No match")
                    else:
                        print(f"  [{processed+1}/{total_to_ground}] [{node_type_label}] - '{clean_node_name}' (Complex)")

            except Exception as e:
                grounded_data.update(_create_fallback_data(clean_node_name, "error"))
                print(f"  [{processed+1}/{total_to_ground}] [{node_type_label}] ✗ Error: {str(e)[:50]}")
        else:
            grounded_data.update(_create_fallback_data(clean_node_name, "unavailable"))
        
        _log_grounding_result(node_name, clean_node_name, grounded_data)
        ont_name = grounded_data.get('ontology_name', 'UNKNOWN')
        grounding_stats[ont_name] = grounding_stats.get(ont_name, 0) + 1
        grounded_nodes[clean_node_name] = grounded_data
        processed += 1
    
    print("\n" + "-"*50)
    
    _export_csv_phase3(grounded_nodes)
    
    return grounded_nodes


def _clean_node_text(text: str) -> str:
    cleaned = re.sub(r'\[(Event|Entity):\s*(.*?)\]', r'\2', text)
    return cleaned.strip()

def _create_fallback_data(name, source_type):
    """Tạo dữ liệu mặc định khi không tìm thấy"""
    return {
        'ontology_id': f"UNKNOWN:{name.upper()[:8]}",
        'ontology_name': 'UNKNOWN',
        'umls_id': None,
        'uri': '',
        'match_score': 0.0,
        'label': name,
        'source': source_type,
        'alternative_matches': []
    }

def _export_csv_phase3(grounded_nodes):
    """Xuất file CSV kết quả"""
    try:
        import csv
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "Eval" / "import" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "phase3_grounded_nodes.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvf:
            fieldnames = ["clean_node", "original_node", "ontology_id", "ontology_name", "umls_id", "label", "match_score", "semantic_type", "source", "alternative_matches"]
            writer = csv.DictWriter(csvf, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for data in grounded_nodes.values():
                row = data.copy()
                if "alternative_matches" in row: 
                    row["alternative_matches"] = json.dumps(row["alternative_matches"], ensure_ascii=False)
                writer.writerow(row)
        print(f"  ✓ Phase 3b CSV exported: {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to write Phase 3 CSV: {e}")

def _infer_semantic_type(concept: str) -> str:
    """Infer semantic type from concept string."""
    if not concept: return "Medical Concept"
    concept_lower = concept.lower()
    if any(word in concept_lower for word in ['drug', 'medication', 'medicine', 'inhibitor']):
        return "Pharmacologic Substance"
    elif any(word in concept_lower for word in ['disease', 'disorder', 'syndrome', 'pain']):
        return "Disease or Syndrome"
    elif any(word in concept_lower for word in ['procedure', 'surgery', 'test']):
        return "Therapeutic or Preventive Procedure"
    return "Medical Concept"