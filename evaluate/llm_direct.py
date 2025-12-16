"""
Direct LLM Answer Generator
Uses local LLM (LM Studio) to answer questions directly without knowledge graph.
"""

import csv
from pathlib import Path
from tqdm import tqdm


class LLMGenerator:
    """Wrapper for LLM API calls."""
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            self.available = True
            print("✓ Connected to LM Studio")
        except Exception as e:
            print(f"⚠ Error: Could not connect to LM Studio: {e}")
            print("  Make sure LM Studio is running at http://localhost:1234")
            self.available = False
        
    def answer_question(self, question: str) -> str:
        """Generate answer from LLM."""
        if not self.available:
            return "ERROR: LLM not available"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Answer questions accurately and concisely."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            response = self.client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"


def process_questions_from_csv(llm: LLMGenerator, input_csv_path: str, output_csv_path: str):
    """Process questions from CSV and save answers."""
    import csv
    from tqdm import tqdm
    
    print(f"\nProcessing questions from: {input_csv_path}")
    
    # Read questions
    questions = []
    with open(input_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get('question', '').strip()
            if question:  # Skip empty questions
                questions.append(question)
    
    print(f"Found {len(questions)} questions to process")
    
    # Process each question
    results = []
    for i, question in enumerate(tqdm(questions, desc="Answering questions")):
        
        print(f"\n{'='*80}")
        print(f"[Question {i+1}/{len(questions)}]")
        print(f"Q: {question}")
        print(f"{'='*80}")
        
        try:
            # Get answer from LLM
            answer = llm.answer_question(question)
            print(f"LLM Answer: {answer}")
            
            results.append({
                'question': question,
                'answer': answer
            })
            
        except Exception as e:
            print(f"\nError processing question {i+1}: {e}")
            results.append({
                'question': question,
                'answer': f"ERROR: {str(e)}"
            })
    
    # Save results
    with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n✓ Results saved to: {output_csv_path}")


def main():
    """Main execution."""
    print("="*80)
    print("Direct LLM Answer Generation")
    print("="*80)
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = LLMGenerator()
    
    if not llm.available:
        print("\n❌ Cannot proceed without LLM connection")
        return
    
    # Get base directory and setup paths
    base_dir = Path(__file__).parent.parent
    input_csv = base_dir / "Eval" / "data" / "1000.csv"
    output_csv = base_dir / "Eval" / "data" / "llm_answer.csv"
    
    # Process questions from CSV
    if input_csv.exists():
        process_questions_from_csv(llm, str(input_csv), str(output_csv))
    else:
        print(f"\n⚠ Input file not found: {input_csv}")


if __name__ == "__main__":
    main()
