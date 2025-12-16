"""
ROUGE Score Evaluation for ToG vs LLM Answers
Compares answers from Think on Graph and direct LLM against ground truth.
"""

import csv
from pathlib import Path
from rouge_score import rouge_scorer
from typing import Dict, List


def load_csv_data(csv_path: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_f = []
    rouge2_f = []
    rougeL_f = []
    
    for pred, ref in zip(predictions, references):
        # Handle empty strings
        if not pred.strip():
            pred = "no answer"
        if not ref.strip():
            ref = "no answer"
            
        scores = scorer.score(ref, pred)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_f) / len(rouge1_f) if rouge1_f else 0.0,
        'rouge2': sum(rouge2_f) / len(rouge2_f) if rouge2_f else 0.0,
        'rougeL': sum(rougeL_f) / len(rougeL_f) if rougeL_f else 0.0
    }


def main():
    """Main evaluation."""
    print("="*80)
    print("ROUGE Score Evaluation: ToG vs LLM")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    ground_truth_path = base_dir / "Eval" / "data" / "1000.csv"
    tog_answer_path = base_dir / "Eval" / "data" / "ToG_answer.csv"
    llm_answer_path = base_dir / "Eval" / "data" / "llm_answer.csv"
    
    # Load data
    print("\nLoading data...")
    ground_truth = load_csv_data(str(ground_truth_path))
    tog_answers = load_csv_data(str(tog_answer_path))
    llm_answers = load_csv_data(str(llm_answer_path))
    
    print(f"Ground truth: {len(ground_truth)} entries")
    print(f"ToG answers: {len(tog_answers)} entries")
    print(f"LLM answers: {len(llm_answers)} entries")
    
    # Extract answers
    references = [row.get('answer', '') for row in ground_truth]
    tog_predictions = [row.get('answer', '') for row in tog_answers]
    llm_predictions = [row.get('answer', '') for row in llm_answers]
    
    # Ensure all lists have same length
    min_len = min(len(references), len(tog_predictions), len(llm_predictions))
    references = references[:min_len]
    tog_predictions = tog_predictions[:min_len]
    llm_predictions = llm_predictions[:min_len]
    
    print(f"\nEvaluating {min_len} question-answer pairs...\n")
    
    # Calculate ROUGE scores for ToG
    print("Calculating ROUGE scores for ToG...")
    tog_scores = calculate_rouge_scores(tog_predictions, references)
    
    print("\n" + "="*80)
    print("ToG Results:")
    print("="*80)
    print(f"ROUGE-1 F1: {tog_scores['rouge1']:.4f}")
    print(f"ROUGE-2 F1: {tog_scores['rouge2']:.4f}")
    print(f"ROUGE-L F1: {tog_scores['rougeL']:.4f}")
    
    # Calculate ROUGE scores for LLM
    print("\nCalculating ROUGE scores for LLM...")
    llm_scores = calculate_rouge_scores(llm_predictions, references)
    
    print("\n" + "="*80)
    print("LLM Results:")
    print("="*80)
    print(f"ROUGE-1 F1: {llm_scores['rouge1']:.4f}")
    print(f"ROUGE-2 F1: {llm_scores['rouge2']:.4f}")
    print(f"ROUGE-L F1: {llm_scores['rougeL']:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison (ToG - LLM):")
    print("="*80)
    print(f"ROUGE-1 Δ: {tog_scores['rouge1'] - llm_scores['rouge1']:+.4f}")
    print(f"ROUGE-2 Δ: {tog_scores['rouge2'] - llm_scores['rouge2']:+.4f}")
    print(f"ROUGE-L Δ: {tog_scores['rougeL'] - llm_scores['rougeL']:+.4f}")
    
    # Save results
    results_path = base_dir / "Eval" / "data" / "rouge_evaluation.csv"
    with open(results_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        writer.writerow(['ToG', f"{tog_scores['rouge1']:.4f}", f"{tog_scores['rouge2']:.4f}", f"{tog_scores['rougeL']:.4f}"])
        writer.writerow(['LLM', f"{llm_scores['rouge1']:.4f}", f"{llm_scores['rouge2']:.4f}", f"{llm_scores['rougeL']:.4f}"])
        writer.writerow(['Difference', f"{tog_scores['rouge1'] - llm_scores['rouge1']:+.4f}", 
                        f"{tog_scores['rouge2'] - llm_scores['rouge2']:+.4f}", 
                        f"{tog_scores['rougeL'] - llm_scores['rougeL']:+.4f}"])
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
