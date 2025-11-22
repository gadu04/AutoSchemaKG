"""
BERTScore Evaluation for ToG vs LLM Answers
Uses semantic similarity to evaluate answer quality against ground truth.
"""

import csv
from pathlib import Path
from bert_score import score
from typing import Dict, List
import torch


def load_csv_data(csv_path: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from CSV."""
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore (Precision, Recall, F1)."""
    # Handle empty strings
    predictions = [p if p.strip() else "no answer" for p in predictions]
    references = [r if r.strip() else "no answer" for r in references]
    
    # Calculate BERTScore
    # model_type options: 'bert-base-uncased', 'roberta-large', 'microsoft/deberta-xlarge-mnli'
    # Use deberta for better performance on medical text
    P, R, F1 = score(
        predictions, 
        references, 
        model_type='microsoft/deberta-xlarge-mnli',
        lang='en',
        verbose=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


def main():
    """Main evaluation."""
    print("="*80)
    print("BERTScore Evaluation: ToG vs LLM")
    print("="*80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
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
    
    # Calculate BERTScore for ToG
    print("="*80)
    print("Calculating BERTScore for ToG...")
    print("="*80)
    tog_scores = calculate_bertscore(tog_predictions, references)
    
    print("\n" + "="*80)
    print("ToG Results:")
    print("="*80)
    print(f"BERTScore Precision: {tog_scores['precision']:.4f}")
    print(f"BERTScore Recall:    {tog_scores['recall']:.4f}")
    print(f"BERTScore F1:        {tog_scores['f1']:.4f}")
    
    # Calculate BERTScore for LLM
    print("\n" + "="*80)
    print("Calculating BERTScore for LLM...")
    print("="*80)
    llm_scores = calculate_bertscore(llm_predictions, references)
    
    print("\n" + "="*80)
    print("LLM Results:")
    print("="*80)
    print(f"BERTScore Precision: {llm_scores['precision']:.4f}")
    print(f"BERTScore Recall:    {llm_scores['recall']:.4f}")
    print(f"BERTScore F1:        {llm_scores['f1']:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison (ToG - LLM):")
    print("="*80)
    print(f"Precision Δ: {tog_scores['precision'] - llm_scores['precision']:+.4f}")
    print(f"Recall Δ:    {tog_scores['recall'] - llm_scores['recall']:+.4f}")
    print(f"F1 Δ:        {tog_scores['f1'] - llm_scores['f1']:+.4f}")
    
    # Save results
    results_path = base_dir / "Eval" / "data" / "bertscore_evaluation.csv"
    with open(results_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Precision', 'Recall', 'F1'])
        writer.writerow(['ToG', f"{tog_scores['precision']:.4f}", f"{tog_scores['recall']:.4f}", f"{tog_scores['f1']:.4f}"])
        writer.writerow(['LLM', f"{llm_scores['precision']:.4f}", f"{llm_scores['recall']:.4f}", f"{llm_scores['f1']:.4f}"])
        writer.writerow(['Difference', 
                        f"{tog_scores['precision'] - llm_scores['precision']:+.4f}", 
                        f"{tog_scores['recall'] - llm_scores['recall']:+.4f}", 
                        f"{tog_scores['f1'] - llm_scores['f1']:+.4f}"])
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
