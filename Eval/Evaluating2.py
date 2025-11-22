import json
import os
import re
from openai import OpenAI

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# 1. File văn bản gốc (Markdown)
TEXT_FILE_PATH = r"E:\AutoSchemaKG\data\parsed\AMA_Family_Guide_content.md"

# 2. File kết quả Triples (JSON)
TRIPLES_FILE_PATH = r"E:\AutoSchemaKG\output\Phase2_Response.json"

# Cấu hình LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "local-model" 

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

def clean_json_string(text):
    """Làm sạch chuỗi JSON trả về từ LLM"""
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()

def load_full_text(file_path):
    """Đọc toàn bộ nội dung file Markdown"""
    print(f"📖 Loading text from: {file_path}...")
    if not os.path.exists(file_path):
        print("❌ Error: Text file not found.")
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def find_context_window(full_text, head, tail, window_size=1000):
    """
    Tìm đoạn văn bản chứa cả Head và Tail (hoặc ít nhất là Head).
    Trả về đoạn text xung quanh (context window) để làm bằng chứng.
    """
    # Chuyển về chữ thường để tìm kiếm không phân biệt hoa thường
    text_lower = full_text.lower()
    head_lower = head.lower()
    tail_lower = tail.lower()
    
    # Ưu tiên 1: Tìm vị trí mà cả Head và Tail xuất hiện gần nhau
    # Tìm vị trí head
    start_idx = text_lower.find(head_lower)
    
    if start_idx == -1:
        # Nếu không thấy Head, thử tìm Tail (fallback)
        start_idx = text_lower.find(tail_lower)
    
    if start_idx != -1:
        # Nếu tìm thấy, lấy đoạn text xung quanh vị trí đó
        start_window = max(0, start_idx - window_size // 2)
        end_window = min(len(full_text), start_idx + window_size // 2)
        return full_text[start_window:end_window]
    
    return None

def evaluate_triple_accuracy(evidence_text, triple_str):
    """Gửi bằng chứng và triple cho LLM để chấm điểm"""
    prompt = f"""You are an expert Knowledge Graph Evaluator.
Verify if the extracted Triple is supported by the Source Text snippet.

### Source Text Snippet:
"...{evidence_text}..."

### Extracted Triple:
{triple_str}

### Task:
Determine if the triple is correct based **ONLY** on the provided text snippet.
- **TP (True Positive)**: The text explicitly supports this relationship.
- **FP (False Positive)**: The text contradicts this or does not mention this relationship.
- **FN (False Negative)**: (Ignore for single triple verification).

### Output (JSON Only):
{{
  "reasoning": "Brief explanation",
  "result": "TP" or "FP"
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": "Return valid JSON only."}, 
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            stream=False
        )
        res = json.loads(clean_json_string(response.choices[0].message.content))
        return res.get('result', 'FP')
    except Exception:
        return 'FP' # Nếu lỗi coi như sai

def main():
    print(f"🚀 STARTING EVALUATION (Search & Verify Mode)")
    print("-" * 60)

    # 1. Load dữ liệu
    full_text = load_full_text(TEXT_FILE_PATH)
    if not full_text: return

    if not os.path.exists(TRIPLES_FILE_PATH):
        print(f"❌ Error: Triples file not found at {TRIPLES_FILE_PATH}")
        return

    with open(TRIPLES_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_triples = data.get('all_triples', [])

    total_tp = 0
    total_fp = 0
    processed_count = 0
    
    # Để tiết kiệm thời gian, ta có thể giới hạn số lượng triple test (ví dụ 100 cái đầu)
    # Nếu muốn chạy hết thì bỏ dòng [:50]
    test_limit = 2200
    print(f"Found {len(all_triples)} triples. Evaluating first {test_limit} triples...")

    for i, t in enumerate(all_triples[:test_limit]):
        head = t.get('head', '')
        tail = t.get('tail', '')
        relation = t.get('relation', '')
        
        triple_str = f"({head}) --[{relation}]--> ({tail})"
        
        # 2. Tìm bằng chứng trong văn bản gốc
        evidence = find_context_window(full_text, head, tail)
        
        print(f"[{i+1}/{test_limit}] Checking: {triple_str} ... ", end="", flush=True)
        
        if evidence:
            # 3. Nhờ LLM chấm điểm
            result = evaluate_triple_accuracy(evidence, triple_str)
            if result == 'TP':
                total_tp += 1
                print("✅ TP")
            else:
                total_fp += 1
                print("❌ FP")
        else:
            # Nếu không tìm thấy Head/Tail trong văn bản -> Chắc chắn là hallucination (FP)
            total_fp += 1
            print("⚠️ Not found in text (FP)")
        
        processed_count += 1

    # 4. Báo cáo
    print("\n" + "=" * 60)
    print("📊 ACCURACY REPORT (Sampled)")
    print("=" * 60)
    
    if processed_count == 0: return

    precision = total_tp / processed_count
    
    print(f"Triples Evaluated: {processed_count}")
    print("-" * 30)
    print(f"✅ True Positives:  {total_tp}")
    print(f"❌ False Positives: {total_fp}")
    print("-" * 30)
    print(f"🎯 PRECISION: {precision:.2%}")
    print("(Note: Recall cannot be calculated accurately without Ground Truth labeling)")
    print("=" * 60)

if __name__ == "__main__":
    main()