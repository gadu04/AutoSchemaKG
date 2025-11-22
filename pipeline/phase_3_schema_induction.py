"""
Phase 3: Hybrid Schema Induction & Ontology Grounding
======================================================
This module performs two tasks:

Part 3a (CORE MODULE): Dynamic concept induction using LLM
    - Analyzes extracted nodes and generates abstract "induced concepts"
    - Example: "Metformin" → "a type of diabetes medication"

Part 3b (UMLS-INTEGRATED): Ontology grounding using UMLS API
    - Maps induced concepts to UMLS identifiers
    - Example: "Metformin" → "UMLS:C0025598"
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from llm_api.interface import call_llm_for_concepts

try:
    from pipeline.umls_loader import UMLSLoader
except ImportError:
    UMLSLoader = None


# =========================================================================
# LOGGING CONFIGURATION
# =========================================================================
LOG_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PHASE3_LOG_FILE = Path(os.getenv("PHASE3_LOG_FILE", str(LOG_DIR / "phase3_induction_grounding.log")))

def _init_phase3_log():
    """Initialize Phase 3 log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear previous log
    with open(PHASE3_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PHASE 3: SCHEMA INDUCTION & ONTOLOGY GROUNDING LOG\n")
        f.write("=" * 100 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")


def _log_concept_induction(node_name: str, concept_phrases: str, batch_num: int = None):
    """Log concept induction results."""
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[CONCEPT INDUCTION]")
            if batch_num:
                f.write(f" (Batch {batch_num})")
            f.write(f"\n")
            f.write(f"  Node: {node_name}\n")
            f.write(f"  Induced Concepts: {concept_phrases}\n")
            f.write(f"  Timestamp: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write("-" * 100 + "\n")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to log concept induction: {e}")


def _log_grounding_result(node_name: str, clean_name: str, grounded_data: Dict):
    """Log grounding results."""
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[GROUNDING RESULT]\n")
            f.write(f"  Original Node: {node_name}\n")
            f.write(f"  Clean Node: {clean_name}\n")
            f.write(f"  Semantic Type: {grounded_data['semantic_type']}\n")
            f.write(f"  Ontology Name: {grounded_data['ontology_name']}\n")
            f.write(f"  Ontology ID: {grounded_data['ontology_id']}\n")
            f.write(f"  Match Score: {grounded_data['match_score']}\n")
            f.write(f"  Source: {grounded_data['source']}\n")
            f.write(f"  Timestamp: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write("-" * 100 + "\n")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to log grounding result: {e}")


def _log_phase3_summary(induced_count: int, grounded_count: int, concept_stats: Dict, grounding_stats: Dict):
    """Log Phase 3 summary statistics."""
    try:
        with open(PHASE3_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write("PHASE 3 SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("PART 3a: CONCEPT INDUCTION\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Total Nodes Induced: {induced_count}\n")
            f.write(f"\n  Concept Statistics:\n")
            for semantic_type, count in sorted(concept_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"    - {semantic_type}: {count}\n")
            
            f.write(f"\nPART 3b: ONTOLOGY GROUNDING\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Total Nodes Grounded: {grounded_count}\n")
            f.write(f"\n  Grounding Distribution:\n")
            for ont_name, count in sorted(grounding_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"    - {ont_name}: {count}\n")
            
            f.write(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to log summary: {e}")


# ===== NEW helper for printing progress =====
def _print_phase3_progress(stage: str, current: int, total: int) -> None:
    """Print one-line progress: 'stage: current/total (xx.x%)'"""
    try:
        if total <= 0:
            return
        pct = (current / total) * 100
        # carriage return to overwrite same line, newline when complete
        end_char = "\n" if current >= total else ""
        print(f"\r  {stage}: {current}/{total} ({pct:.1f}%)", end=end_char, flush=True)
    except Exception:
        # never raise from progress print
        pass


# =========================================================================
# CORE FUNCTIONS
# =========================================================================

def dynamically_induce_concepts(unique_nodes: Set[str], all_triples: List[Dict] = None, use_real_llm: bool = False) -> Dict[str, str]:
    """
    Part 3a: Dynamically induce abstract concepts for each node.
    FIXED: Implements batching to avoid Context Window Exceeded errors.
    WITH LOGGING.
    """
    _init_phase3_log()
    
    node_list = sorted(list(unique_nodes))
    total_nodes = len(node_list)
    
    print(f"  Analyzing {total_nodes} unique nodes...")
    
    induced_concepts = {}
    batch_num = 0
    
    BATCH_SIZE = 50 
    processed_nodes = 0
    
    for i in range(0, total_nodes, BATCH_SIZE):
        batch_num += 1
        batch_nodes = node_list[i : i + BATCH_SIZE]
        print(f"    Processing batch {batch_num}/{(total_nodes + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_nodes)} nodes)...")
        
        try:
            batch_concepts = call_llm_for_concepts(
                batch_nodes, 
                use_real_llm=use_real_llm,
                triples_list=all_triples 
            )
            
            # Log each concept in batch
            for node_name, concept_phrases in batch_concepts.items():
                _log_concept_induction(node_name, concept_phrases, batch_num)
            
            induced_concepts.update(batch_concepts)
            # update progress
            processed_nodes += len(batch_nodes)
            _print_phase3_progress("Phase 3a Progress", processed_nodes, total_nodes)
            
        except Exception as e:
            print(f"    ⚠ Batch {batch_num} failed: {e}")
            for node in batch_nodes:
                induced_concepts[node] = "medical concept, entity"
                _log_concept_induction(node, "medical concept, entity (fallback)", batch_num)
            processed_nodes += len(batch_nodes)
            _print_phase3_progress("Phase 3a Progress", processed_nodes, total_nodes)
    
    missing_nodes = set(node_list) - set(induced_concepts.keys())
    if missing_nodes:
        print(f"  ⚠ Warning: {len(missing_nodes)} nodes missing concept induction")
        for node in missing_nodes:
            induced_concepts[node] = "medical concept, entity"
            _log_concept_induction(node, "medical concept, entity (fallback - missing)", batch_num)
    
    # ensure progress shows 100%
    _print_phase3_progress("Phase 3a Progress", total_nodes, total_nodes)
    
    return induced_concepts


def ground_concepts_to_ontology(induced_concepts: Dict[str, str], use_umls: bool = True) -> Dict[str, Dict]:
    """
    Part 3b: Ground induced concepts to ontologies.
    FIXED: Remove [Event: ...] wrapper before grounding.
    WITH LOGGING.
    """
    print("  Initializing ontology grounding...")
    
    grounded_nodes = {}
    grounding_stats = {}
    
    # Try to initialize UMLS loader if enabled
    umls_loader = None
    if use_umls and UMLSLoader is not None:
        try:
            umls_loader = UMLSLoader()
            if umls_loader.is_available():
                print(f"  ✓ UMLS API: Ready for grounding")
            else:
                print(f"  ⚠ UMLS API: Not available - using fallback mode")
                umls_loader = None
        except Exception as e:
            print(f"  ✗ UMLS API Error: {e} - using fallback mode")
            umls_loader = None
    else:
        if not use_umls:
            print(f"  ℹ UMLS grounding disabled")
    
    print(f"  Grounding {len(induced_concepts)} concepts...\n")
    
    total_to_ground = len(induced_concepts)
    processed = 0
    
    # Process each concept
    for node_name, concept_phrases in induced_concepts.items():
        # ===== NEW: Remove [Event: ...] wrapper before any printing =====
        clean_node_name = _clean_event_wrapper(node_name)
        print(f"    Grounding '{clean_node_name}'...", end=" ", flush=True)
        
        # Extract primary concept (first phrase)
        primary_concept = concept_phrases.split(',')[0].strip() if concept_phrases else clean_node_name
        
        grounded_data = {
            'induced_concept': concept_phrases,
            'semantic_type': _infer_semantic_type(concept_phrases),
            'original_node': node_name,  # Keep original with [Event: ...]
            'clean_node': clean_node_name,  # Clean version without [Event: ...]
        }
        
        # Try UMLS grounding if available
        if umls_loader and umls_loader.is_available():
            try:
                # Search using clean name
                umls_match = umls_loader.get_best_match(primary_concept, threshold=0.4)
                umls_alternatives = umls_loader.get_all_matches(primary_concept, threshold=0.2)
                
                if umls_match:
                    grounded_data.update({
                        'ontology_id': umls_match['umls_id'],
                        'ontology_name': umls_match['source'] or 'UMLS',
                        'umls_id': umls_match['umls_id'],
                        'uri': f"https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/{umls_match['umls_id']}" if umls_match['umls_id'] else '',
                        'match_score': umls_match['score'],
                        'label': umls_match['name'],
                        'source': 'umls_api',
                        'alternative_matches': [
                            {
                                'ontology_id': m['umls_id'],
                                'ontology': m['source'] or 'UMLS',
                                'score': m['score']
                            }
                            for m in umls_alternatives[1:4]
                        ]
                    })
                    print(f"✓ UMLS:{umls_match['umls_id']}")
                else:
                    # No match - use fallback
                    grounded_data.update({
                        'ontology_id': f"UNKNOWN:{clean_node_name.upper()[:8]}",
                        'ontology_name': 'UNKNOWN',
                        'umls_id': None,
                        'uri': '',
                        'match_score': 0.0,
                        'label': clean_node_name,
                        'source': 'fallback',
                        'alternative_matches': []
                    })
                    print(f"⚠ No match")
            
            except Exception as e:
                print(f"✗ Error: {str(e)[:30]}")
                grounded_data.update({
                    'ontology_id': f"ERROR:{clean_node_name.upper()[:8]}",
                    'ontology_name': 'ERROR',
                    'umls_id': None,
                    'uri': '',
                    'match_score': 0.0,
                    'label': clean_node_name,
                    'source': 'error',
                    'alternative_matches': []
                })
        else:
            # UMLS not available
            grounded_data.update({
                'ontology_id': f"UNKNOWN:{clean_node_name.upper()[:8]}",
                'ontology_name': 'UNKNOWN',
                'umls_id': None,
                'uri': '',
                'match_score': 0.0,
                'label': clean_node_name,
                'source': 'unavailable',
                'alternative_matches': []
            })
            print(f"⚠ UMLS unavailable")
        
        # Log grounding result
        _log_grounding_result(node_name, clean_node_name, grounded_data)
        
        # Update statistics
        ont_name = grounded_data['ontology_name']
        grounding_stats[ont_name] = grounding_stats.get(ont_name, 0) + 1
        
        # Use clean node name as key in output
        grounded_nodes[clean_node_name] = grounded_data
        
        # update and print progress
        processed += 1
        _print_phase3_progress("Phase 3b Progress", processed, total_to_ground)
    
    print()
    
    # Calculate concept statistics
    concept_stats = {}
    for data in grounded_nodes.values():
        semantic_type = data['semantic_type']
        concept_stats[semantic_type] = concept_stats.get(semantic_type, 0) + 1
    
    # Log summary
    _log_phase3_summary(
        len(induced_concepts),
        len(grounded_nodes),
        concept_stats,
        grounding_stats
    )

    try:
        import csv
        # Use Eval/import/data directory for output
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "Eval" / "import" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "phase3_grounded_nodes.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvf:
            fieldnames = [
                "clean_node",
                "original_node",
                "ontology_id",
                "ontology_name",
                "umls_id",
                "label",
                "match_score",
                "semantic_type",
                "source",
                "uri",
                "alternative_matches"
            ]
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            for clean_name, data in grounded_nodes.items():
                writer.writerow({
                    "clean_node": clean_name,
                    "original_node": data.get("original_node", ""),
                    "ontology_id": data.get("ontology_id", ""),
                    "ontology_name": data.get("ontology_name", ""),
                    "umls_id": data.get("umls_id", ""),
                    "label": data.get("label", ""),
                    "match_score": data.get("match_score", 0.0),
                    "semantic_type": data.get("semantic_type", ""),
                    "source": data.get("source", ""),
                    "uri": data.get("uri", ""),
                    "alternative_matches": json.dumps(data.get("alternative_matches", []), ensure_ascii=False)
                })
        print(f"  ✓ Phase 3b CSV exported: {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to write Phase 3 CSV: {e}")
    
    return grounded_nodes


def _clean_event_wrapper(node_name: str) -> str:
    """
    Remove [Event: ...] wrapper from node names.
    
    Examples:
        "[Event: patient's participation in a diabetes education program]" 
        → "patient's participation in a diabetes education program"
        
        "Metformin" → "Metformin"
    """
    if node_name.startswith("[Event:") and node_name.endswith("]"):
        # Remove [Event: and trailing ]
        clean = node_name[7:-1].strip()
        return clean
    
    # Already clean or not an event
    return node_name


def _infer_semantic_type(concept: str) -> str:
    """Infer semantic type from concept."""
    concept_lower = concept.lower()
    
    if any(word in concept_lower for word in ['drug', 'medication', 'medicine', 'pharmaceutical', 'inhibitor', 'metformin', 'statin', 'ace']):
        return "Pharmacologic Substance"
    elif any(word in concept_lower for word in ['disease', 'disorder', 'illness', 'syndrome', 'diabetes', 'hypertension', 'nephropathy']):
        return "Disease or Syndrome"
    elif any(word in concept_lower for word in ['symptom', 'sign', 'pain', 'pressure', 'glucose', 'blood']):
        return "Sign or Symptom"
    elif any(word in concept_lower for word in ['procedure', 'treatment', 'therapy', 'surgery', 'intervention', 'monitoring', 'counseling', 'detection']):
        return "Therapeutic or Preventive Procedure"
    elif any(word in concept_lower for word in ['trial', 'study', 'program', 'education', 'research', 'protocol']):
        return "Research Activity"
    elif any(word in concept_lower for word in ['patient', 'person', 'individual', 'participant', 'provider']):
        return "Patient or Healthcare Provider"
    elif any(word in concept_lower for word in ['board', 'committee', 'institution']):
        return "Organization"
    else:
        return "Medical Concept"

