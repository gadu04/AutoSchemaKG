"""
BERTScore Evaluation for ToG vs LLM Answers - CUDA VERSION
Forces usage of GPU for maximum performance AND saves results to CSV.
"""

import csv
import sys
from pathlib import Path
from bert_score import score
from typing import Dict, List
import torch
from datetime import datetime

def check_cuda_availability():
    """Ensure CUDA is available before running."""
    if not torch.cuda.is_available():
        print("❌ LỖI: Không tìm thấy GPU (CUDA).")
        print("   Code này được cấu hình để CHỈ chạy trên GPU để đảm bảo tốc độ.")
        print("   Vui lòng kiểm tra lại driver hoặc cài đặt PyTorch với hỗ trợ CUDA.")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ Đã tìm thấy GPU: {device_name}")
    print(f"   VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return 'cuda'

def load_csv_data(csv_path: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from CSV."""
    data = []
    if not Path(csv_path).exists():
        print(f"⚠ Warning: File not found: {csv_path}")
        return []
        
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore using CUDA."""
    # Handle empty strings
    predictions = [p if p.strip() else "no answer" for p in predictions]
    references = [r if r.strip() else "no answer" for r in references]
    
    # Calculate BERTScore
    # model_type: 'microsoft/deberta-xlarge-mnli' is best but heavy. 
    # If GPU VRAM < 8GB, consider switching to 'roberta-large'
    P, R, F1 = score(
        predictions, 
        references, 
        model_type='microsoft/deberta-xlarge-mnli',
        lang='en',
        verbose=True,
        device='cuda', # <--- BẮT BUỘC DÙNG CUDA
        batch_size=4   # <--- Tăng tốc độ xử lý theo batch trên GPU
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def main():
    """Main evaluation."""
    print("="*80)
    print("BERTScore Evaluation: ToG vs LLM (CUDA MODE + CSV EXPORT)")
    print("="*80)
    
    # 1. Bắt buộc kiểm tra GPU
    device = check_cuda_availability()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    ground_truth_path = base_dir / "Eval" / "data" / "1000.csv"
    
    # Đường dẫn file kết quả
    tog_answer_path = base_dir / "Eval" / "data" / "ToG_answer.csv"
    llm_answer_path = base_dir / "Eval" / "data" / "llm_answer.csv"
    results_path = base_dir / "Eval" / "data" / "bertscore_evaluation.csv"
    
    # Load data
    print("\nLoading data...")
    ground_truth = load_csv_data(str(ground_truth_path))
    tog_answers = load_csv_data(str(tog_answer_path))
    llm_answers = load_csv_data(str(llm_answer_path))
    
    if not ground_truth or not tog_answers:
        print("❌ Không đủ dữ liệu để đánh giá.")
        return

    print(f"Ground truth: {len(ground_truth)} entries")
    print(f"ToG answers: {len(tog_answers)} entries")
    print(f"LLM answers: {len(llm_answers)} entries")
    
    # Extract answers
    references = [row.get('answer', '') or row.get('Answer', '') for row in ground_truth]
    tog_predictions = [row.get('answer', '') or row.get('Answer', '') for row in tog_answers]
    llm_predictions = [row.get('answer', '') or row.get('Answer', '') for row in llm_answers]
    
    # Ensure all lists have same length
    min_len = min(len(references), len(tog_predictions))
    if llm_predictions:
        min_len = min(min_len, len(llm_predictions))
    
    references = references[:min_len]
    tog_predictions = tog_predictions[:min_len]
    if llm_predictions:
        llm_predictions = llm_predictions[:min_len]
    
    print(f"\nEvaluating {min_len} question-answer pairs on GPU...\n")
    
    # Calculate BERTScore for ToG
    print("="*80)
    print("Calculating BERTScore for ToG...")
    print("="*80)
    tog_scores = calculate_bertscore(tog_predictions, references)
    
    print(f"ToG Results -> F1: {tog_scores['f1']:.4f} | Precision: {tog_scores['precision']:.4f} | Recall: {tog_scores['recall']:.4f}")
    
    # Calculate BERTScore for LLM
    llm_scores = None
    if llm_predictions:
        print("\n" + "="*80)
        print("Calculating BERTScore for LLM...")
        print("="*80)
        llm_scores = calculate_bertscore(llm_predictions, references)
        print(f"LLM Results -> F1: {llm_scores['f1']:.4f} | Precision: {llm_scores['precision']:.4f} | Recall: {llm_scores['recall']:.4f}")
        
        # Comparison
        print("\n" + "="*80)
        print(f"F1 Difference (ToG - LLM): {tog_scores['f1'] - llm_scores['f1']:+.4f}")
        print("="*80)
    
    # === SAVE RESULTS TO CSV ===
    print(f"\nSaving results to {results_path}...")
    
    # Chuẩn bị dữ liệu để ghi
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Tạo danh sách các dòng dữ liệu (Rows)
    rows_to_write = []
    
    # Dòng kết quả cho ToG
    rows_to_write.append({
        'Timestamp': timestamp,
        'Model': 'ToG',
        'Samples': min_len,
        'Precision': f"{tog_scores['precision']:.4f}",
        'Recall': f"{tog_scores['recall']:.4f}",
        'F1_Score': f"{tog_scores['f1']:.4f}",
        'Device': device,
        'Model_Type': 'microsoft/deberta-xlarge-mnli'
    })
    
    # Dòng kết quả cho LLM (nếu có)
    if llm_scores:
        rows_to_write.append({
            'Timestamp': timestamp,
            'Model': 'LLM (Baseline)',
            'Samples': min_len,
            'Precision': f"{llm_scores['precision']:.4f}",
            'Recall': f"{llm_scores['recall']:.4f}",
            'F1_Score': f"{llm_scores['f1']:.4f}",
            'Device': device,
            'Model_Type': 'microsoft/deberta-xlarge-mnli'
        })

    # Fieldnames cho CSV
    fieldnames = ['Timestamp', 'Model', 'Samples', 'Precision', 'Recall', 'F1_Score', 'Device', 'Model_Type']
    
    # Ghi file (Chế độ 'a' để append - thêm vào cuối file thay vì ghi đè)
    file_exists = results_path.exists()
    
    try:
        with open(results_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Chỉ ghi header nếu file chưa tồn tại
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(rows_to_write)
            
        print(f"✅ Successfully saved evaluation results to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

if __name__ == "__main__":
    main()