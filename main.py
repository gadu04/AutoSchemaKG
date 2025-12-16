"""
Medical-SchemaKG Framework - Main Orchestrator
================================================
Entry point for the four-phase pipeline that builds a Knowledge Graph from medical text.

Pipeline Phases:
1. Document Ingestion & Preprocessing (Stubbed)
2. Triple Extraction (Core Module)
3. Hybrid Schema Induction & Ontology Grounding (Partial Stub)
4. Knowledge Graph Construction (Core Module)
"""

import os
import sys
import json
import pickle
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
    else:
        print(f"ℹ No .env file found. Using system environment variables.")
except ImportError:
    print("ℹ python-dotenv not installed. Using system environment variables only.")
    print("  Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Warning: Could not load .env file: {e}")

from pipeline.phase_1_ingestion import load_and_segment_text
from pipeline.phase_2_triple_extraction import TripleExtractor
from pipeline.phase_3_schema_induction import dynamically_induce_concepts, ground_concepts_to_ontology
from pipeline.phase_4_kg_construction import build_knowledge_graph, export_graph_to_neo4j_csv
from utils.visualization import print_pipeline_summary, save_graph_visualization


# =============================================================================
# CẤU HÌNH CHẠY (DEBUG CONFIG)
# =============================================================================
# Đặt True nếu bạn muốn bỏ qua Phase 1 & 2 và dùng file Phase2_Response.pkl đã lưu
RESUME_FROM_PHASE_3 = False 
# =============================================================================

def main():
    """
    Main orchestrator function that executes the four-phase pipeline.
    """
    print("=" * 80)
    print("MEDICAL-SCHEMAKG FRAMEWORK - PIPELINE EXECUTION")
    print("=" * 80)
    print()
    
    # Load configuration from environment variables (set in .env file)
    use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() == "true"
    # input_file = os.getenv("INPUT_FILE", "data/parsed/AMA_Family_Guide_content.md")
    input_dir = os.getenv("INPUT_DIR", "data/parsed/")
    output_dir = os.getenv("OUTPUT_DIR", "output")
    model_name = os.getenv("MODEL_NAME", "local-model")
    lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    
    print(f"Configuration:")
    print(f"  - LLM Mode: {'REAL API (LM Studio)' if use_real_llm else 'STUBBED'}")
    if use_real_llm:
        print(f"  - LM Studio URL: {lm_studio_url}")
        print(f"  - Model: {model_name}")
    # print(f"  - Input File: {input_file}")
    print(f"  - Input Directory: {input_dir}")
    print(f"  - Output Directory: {output_dir}")
    print(f"  - Resume Mode: {'ON (Skipping P1/P2)' if RESUME_FROM_PHASE_3 else 'OFF (Running full pipeline)'}")
    print()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to hold state across phases
    all_triples = []
    unique_nodes = set()
    grounded_nodes = []

    # =========================================================================
    # LOGIC: RUN PHASE 1 & 2 OR RESUME FROM CHECKPOINT
    # =========================================================================
    
    if RESUME_FROM_PHASE_3:
        print("-" * 80)
        print("RESUMING PIPELINE FROM PHASE 2 CHECKPOINT")
        print("-" * 80)
        checkpoint_path = os.path.join(output_dir, "Phase2_Response.pkl")
        
        if os.path.exists(checkpoint_path):
            try:
                print(f"📂 Loading data from {checkpoint_path}...")
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    all_triples = checkpoint_data.get("all_triples", [])
                    unique_nodes = checkpoint_data.get("unique_nodes", set())
                
                print(f"✅ Checkpoint loaded successfully!")
                print(f"   - Triples: {len(all_triples)}")
                print(f"   - Unique Nodes: {len(unique_nodes)}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                return
        else:
            print(f"❌ Checkpoint file not found at: {checkpoint_path}")
            print("   Please run with RESUME_FROM_PHASE_3 = False first.")
            return

    else:
        # =========================================================================
        # PHASE 1: DOCUMENT INGESTION & PREPROCESSING (STUBBED)
        # =========================================================================
        print("-" * 80)
        print("PHASE 1: DOCUMENT INGESTION & PREPROCESSING")
        print("-" * 80)
        print(f"Status: Loading and segmenting medical text from directory: {input_dir}...")
        
        text_segments = []
        try:
            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                for filename in os.listdir(input_dir):
                    if filename.endswith(".md") or filename.endswith(".txt"):
                        file_path = os.path.join(input_dir, filename)
                        print(f"   Processing file: {filename}...")
                        segments = load_and_segment_text(file_path)
                        text_segments.extend(segments)
            else:
                input_file = os.getenv("INPUT_FILE", "data/parsed/AMA_Family_Guide_content.md")
                text_segments = load_and_segment_text(input_file)

            print(f"✓ Phase 1 Complete. Found total {len(text_segments)} text chunks from all files.")
            print(f"  Sample chunk: {text_segments[0]['text'][:100]}..." if text_segments else "  No chunks found.")
            print()
        except Exception as e:
            print(f"✗ Phase 1 Failed: {e}")
            return
        
        # =========================================================================
        # PHASE 2: TRIPLE EXTRACTION
        # =========================================================================
        print("-" * 80)
        print("PHASE 2: TRIPLE EXTRACTION")
        print("-" * 80)
        print("Status: Extracting (Head, Relation, Tail) triples from text...")
        
        try:
            extractor = TripleExtractor(use_real_llm=use_real_llm)
            all_triples, unique_nodes = extractor.extract_from_segments(text_segments)
            
            print(f"✓ Phase 2 Complete.")
            print(f"  - Total Triples Extracted: {len(all_triples)}")
            print(f"  - Entity-Entity (E-E): {sum(1 for t in all_triples if t['type'] == 'E-E')}")
            print(f"  - Entity-Event (E-Ev): {sum(1 for t in all_triples if t['type'] == 'E-Ev')}")
            print(f"  - Event-Event (Ev-Ev): {sum(1 for t in all_triples if t['type'] == 'Ev-Ev')}")
            print(f"  - Unique Nodes: {len(unique_nodes)}")
            print()
            
            # -------------------------------------------------------
            # [NEW] SAVE PHASE 2 CHECKPOINT
            # -------------------------------------------------------
            print(f"💾 Saving Phase 2 Checkpoints to '{output_dir}'...")
            
            checkpoint_data = {
                "all_triples": all_triples,
                "unique_nodes": unique_nodes
            }
            
            # 1. Save Pickle (For Machine/Resume)
            pkl_path = os.path.join(output_dir, "Phase2_Response.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"   -> Saved Checkpoint (Pickle): {pkl_path}")
            
            # 2. Save JSON (For Human Reading)
            json_path = os.path.join(output_dir, "Phase2_Response.json")
            
            # Helper for serializing Sets
            def set_default(obj):
                if isinstance(obj, set): return list(obj)
                return str(obj)
                
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=set_default)
            print(f"   -> Saved Output (JSON): {json_path}")
            print()

        except Exception as e:
            print(f"✗ Phase 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # =========================================================================
    # PHASE 3: HYBRID SCHEMA INDUCTION & ONTOLOGY GROUNDING
    # =========================================================================
    print("-" * 80)
    print("PHASE 3: HYBRID SCHEMA INDUCTION & ONTOLOGY GROUNDING")
    print("-" * 80)
    
    # Part 3a: Dynamic Induction (Core Module)
    print("Status: Part 3a - Dynamically inducing abstract concepts...")
    try:
        induced_concepts = dynamically_induce_concepts(
            unique_nodes, 
            all_triples=all_triples,
            use_real_llm=use_real_llm
        )
        print(f"✓ Part 3a Complete. Induced concepts for {len(induced_concepts)} nodes.")
        print()
    except Exception as e:
        print(f"✗ Part 3a Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Part 3b: Ontology Grounding (Stubbed)
    print("Status: Part 3b - Grounding concepts to medical ontologies...")
    try:
        grounded_nodes = ground_concepts_to_ontology(induced_concepts)
        print(f"✓ Part 3b Complete. Grounded {len(grounded_nodes)} nodes to ontology IDs.")
        print()

        # -------------------------------------------------------
        # [NEW] SAVE PHASE 3 OUTPUT
        # -------------------------------------------------------
        print(f"💾 Saving Phase 3 Output...")
        phase3_json_path = os.path.join(output_dir, "Phase3_Response.json")
        
        # Helper for serializing objects
        def obj_dict_serializer(obj):
            if hasattr(obj, '__dict__'): return obj.__dict__
            return str(obj)
            
        with open(phase3_json_path, "w", encoding="utf-8") as f:
            json.dump(grounded_nodes, f, indent=2, ensure_ascii=False, default=obj_dict_serializer)
        print(f"   -> Saved Phase 3 Output (JSON): {phase3_json_path}")
        print()

    except Exception as e:
        print(f"✗ Part 3b Failed: {e}")
        return
    
    # =========================================================================
    # PHASE 4: KNOWLEDGE GRAPH CONSTRUCTION
    # =========================================================================
    print("-" * 80)
    print("PHASE 4: KNOWLEDGE GRAPH CONSTRUCTION")
    print("-" * 80)
    print("Status: Building the final Onto-MedKG graph...")
    
    try:
        knowledge_graph = build_knowledge_graph(all_triples, grounded_nodes)
        print(f"✓ Phase 4 Complete.")
        print(f"  - Total Nodes: {knowledge_graph.number_of_nodes()}")
        print(f"  - Total Edges: {knowledge_graph.number_of_edges()}")
        print()
    except Exception as e:
        print(f"✗ Phase 4 Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # FINALIZATION
    # =========================================================================
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print()
    
    # Print summary
    print_pipeline_summary(text_segments if not RESUME_FROM_PHASE_3 else [], all_triples, grounded_nodes, knowledge_graph)
    
    # Save visualization
    try:
        viz_path = os.path.join(output_dir, "knowledge_graph.png")
        save_graph_visualization(knowledge_graph, viz_path)
        print(f"\n✓ Knowledge graph visualization saved to: {viz_path}")
    except Exception as e:
        print(f"\n⚠ Could not save visualization: {e}")

    # Export Neo4j CSVs for bulk import
    try:
        neo4j_files = export_graph_to_neo4j_csv(knowledge_graph, output_dir)
        print(f"\n✓ Neo4j CSVs written:")
        print(f"  - Nodes: {neo4j_files.get('neo4j_nodes')}")
        print(f"  - Relationships: {neo4j_files.get('neo4j_rels')}")
    except Exception as e:
        print(f"\n⚠ Could not export Neo4j CSVs: {e}")
    
    print("\n" + "=" * 80)
    print("Thank you for using Medical-SchemaKG Framework!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# unified header — 2025-12-16.
# ------------------------------------------------------------