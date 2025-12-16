"""
Medical-SchemaKG - Resume Script (main3.py)
===========================================
Chế độ: CHẠY TỪ ĐẦU PHASE 3a (LLM Concept Induction) -> 3b -> 4
Đã bao gồm: FIX lỗi kết nối Event
"""

import os
import sys
import json
import pickle
import re  # <--- Bắt buộc có để fix lỗi tên
from pathlib import Path

# 1. Cấu hình đường dẫn
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 2. Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    print("✓ Loaded .env file")
except ImportError:
    pass

from pipeline.phase_3_schema_induction import dynamically_induce_concepts, ground_concepts_to_ontology
from pipeline.phase_4_kg_construction import build_knowledge_graph, export_graph_to_neo4j_csv
from utils.visualization import save_graph_visualization

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
# Bắt buộc True để Phase 3a gọi LLM thật
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true" 

# ===========================================================
# CẤU HÌNH CHẠY
# False = Chạy đầy đủ Phase 3a (Gọi LLM)
SKIP_PHASE_3A = False 
# ===========================================================

# --- HÀM HỖ TRỢ CLEAN ---
def clean_triple_text(text):
    """Hàm làm sạch chuỗi: Loại bỏ [Event: ...], Event:, Entity:"""
    if not text: return ""
    # Loại bỏ [Event: ...], [Entity: ...]
    text = re.sub(r'\[(Event|Entity):\s*(.*?)\]', r'\2', text)
    # Loại bỏ prefix Event:, Entity: nếu có
    text = re.sub(r'^(Event|Entity):\s*', '', text)
    return text.strip()
# ------------------------

def main():
    print("=" * 60)
    print("RESUMING PIPELINE: PHASE 2 -> 3a (LLM) -> 3b -> 4")
    print("=" * 60)

    # ---------------------------------------------------------
    # BƯỚC 1: LOAD DỮ LIỆU TỪ PHASE 2 (Input cho Phase 3a)
    # ---------------------------------------------------------
    print("\n📂 [BƯỚC 1] Loading Phase 2 Checkpoint...")
    possible_paths = [
        os.path.join(OUTPUT_DIR, "Phase2_Response.pkl"),
        os.path.join("pipeline", "Phase2_Response.pkl"),
        "Phase2_Response.pkl"
    ]
    checkpoint_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not checkpoint_path:
        print("❌ LỖI: Không tìm thấy file 'Phase2_Response.pkl'.")
        return

    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                all_triples = data.get("all_triples", [])
                unique_nodes = data.get("unique_nodes", set())
            else:
                all_triples = data
                unique_nodes = {t['head'] for t in all_triples} | {t['tail'] for t in all_triples}
        print(f"✅ Đã load: {len(all_triples)} triples, {len(unique_nodes)} nodes.")
    except Exception as e:
        print(f"❌ Lỗi đọc file pickle: {e}")
        return

    # ---------------------------------------------------------
    # BƯỚC 2: CHẠY PHASE 3a (CONCEPT INDUCTION)
    # ---------------------------------------------------------
    induced_concepts = {}
    
    if SKIP_PHASE_3A:
        print("\n⏩ [BƯỚC 2] SKIPPING PHASE 3a...")
        for node in unique_nodes:
            induced_concepts[node] = "Medical Concept"
    else:
        print(f"\n🚀 [BƯỚC 2] CHẠY PHASE 3a: Concept Induction (LLM)...")
        print("   (Quá trình này có thể mất thời gian tùy vào số lượng node và GPU)")
        try:
            # Gọi hàm sinh concept từ LLM
            induced_concepts = dynamically_induce_concepts(
                unique_nodes, 
                all_triples=all_triples,
                use_real_llm=True # Force True để gọi API
            )
            print(f"✅ Đã sinh concept cho {len(induced_concepts)} nodes.")
        except Exception as e:
            print(f"❌ Lỗi Phase 3a: {e}")
            return

    # ---------------------------------------------------------
    # BƯỚC 3: CHẠY PHASE 3b (ONTOLOGY GROUNDING)
    # ---------------------------------------------------------
    print("\n🚀 [BƯỚC 3] CHẠY PHASE 3b: Ontology Grounding...")
    try:
        grounded_nodes = ground_concepts_to_ontology(induced_concepts)
        
        # Lưu kết quả Phase 3
        p3_out = os.path.join(OUTPUT_DIR, "Phase3_Response.json")
        with open(p3_out, "w", encoding="utf-8") as f:
            def default_ser(obj): return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
            json.dump(grounded_nodes, f, indent=2, ensure_ascii=False, default=default_ser)
        print(f"💾 Đã lưu Phase 3 Output: {p3_out}")

    except Exception as e:
        print(f"❌ Lỗi Phase 3b: {e}")
        return

    # ---------------------------------------------------------
    # [FIX] BƯỚC LÀM SẠCH TRIPLES (Clean Triples)
    # ---------------------------------------------------------
    print("\n🛠 [FIX] Cleaning Triple Formats to match Nodes...")
    cleaned_triples = []
    count_fixed = 0
    
    for triple in all_triples:
        new_triple = triple.copy()
        
        # Làm sạch tên Head và Tail (bỏ [Event: ...])
        new_head = clean_triple_text(triple['head'])
        new_tail = clean_triple_text(triple['tail'])
        
        if new_head != triple['head'] or new_tail != triple['tail']:
            count_fixed += 1
            
        new_triple['head'] = new_head
        new_triple['tail'] = new_tail
        cleaned_triples.append(new_triple)
        
    all_triples = cleaned_triples
    print(f"   -> Đã chuẩn hóa {count_fixed} triples.")

    # ---------------------------------------------------------
    # BƯỚC 4: CHẠY PHASE 4 (GRAPH CONSTRUCTION)
    # ---------------------------------------------------------
    print("\n🚀 [BƯỚC 4] CHẠY PHASE 4: Graph Construction...")
    try:
        kg = build_knowledge_graph(all_triples, grounded_nodes)
        print(f"✅ Graph created: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")
        
        # Xuất Neo4j CSV
        export_graph_to_neo4j_csv(kg, OUTPUT_DIR)
        print("✅ Export Neo4j CSV thành công.")
        
        # Xuất ảnh (nếu cài pyvis/networkx visualization)
        viz_path = os.path.join(OUTPUT_DIR, "knowledge_graph.png")
        try:
            save_graph_visualization(kg, viz_path)
            print(f"🖼️ Visualization saved: {viz_path}")
        except: pass

    except Exception as e:
        print(f"❌ Lỗi Phase 4: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ HOÀN TẤT QUY TRÌNH!")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# unified header — 2025-12-16.
# ------------------------------------------------------------