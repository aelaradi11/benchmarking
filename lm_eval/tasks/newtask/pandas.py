from lm_eval.api.task import ConfigurableTask
from datasets import Dataset
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import glob

def load_my_test_dataset(metadata=None, **kwargs):
    data_path = kwargs.get("data_path", "lm_eval/tasks/newtask/converted_benchmark_with_bloom.jsonl")

    with open(data_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines starting with //
            if line and not line.startswith('//'):
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip any lines that can't be parsed as JSON
                    continue
    
    dataset = Dataset.from_list(lines)
    return {
        "test": dataset
    }

class NEWTASK(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "json"  
    DATASET_NAME = None

    def __init__(self, config=None):
        super().__init__(config={
            "custom_dataset": load_my_test_dataset,
            "metadata": {"version": self.VERSION},
        })
        self.OUTPUT_TYPE = "multiple_choice"
        
        # Initialize detailed results tracking
        self.detailed_results = []
        self.bloom_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.output_dir = "detailed_benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Question: {doc['query']}\nAnswer:"

    def doc_to_target(self, doc):
        # Return the correct answer string
        return doc["choices"][doc["gold"]]

    def doc_to_choice(self, doc):
        # Return all multiple choice options
        return doc["choices"]
  
    def process_results(self, doc, results):
        print(doc, results)
        
        # Get index of best loglikelihood score
        pred_idx = int(np.argmax([r[0] for r in results]))
        gold_idx = doc["gold"]
        is_correct = pred_idx == gold_idx
        
        # Get Bloom taxonomy level from doc (handle the space in field name)
        bloom_level = doc.get("Bloom Taxonomy Level", doc.get("bloom_level", "Unknown"))
        
        # Store detailed result
        detailed_result = {
            "question_id": doc.get("id", len(self.detailed_results)),
            "question": doc["query"],
            "choices": doc["choices"],
            "correct_answer_idx": gold_idx,
            "correct_answer": doc["choices"][gold_idx],
            "predicted_answer_idx": pred_idx,
            "predicted_answer": doc["choices"][pred_idx],
            "is_correct": is_correct,
            "bloom_level": bloom_level,
            "loglikelihoods": [r[0] for r in results],
            "confidence_scores": [r[1] if len(r) > 1 else None for r in results]
        }
        
        self.detailed_results.append(detailed_result)
        
        # Update Bloom taxonomy statistics
        self.bloom_stats[bloom_level]['total'] += 1
        if is_correct:
            self.bloom_stats[bloom_level]['correct'] += 1
        
        return {
            "exact_match": is_correct
        }
    
    def create_styled_dataframe(self, model_name, comprehensive_results):
        """Create a styled pandas DataFrame with seaborn color mapping"""
        bloom_analysis = comprehensive_results['bloom_taxonomy_analysis']
        
        # Prepare data for DataFrame
        data = []
        bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        
        for level in bloom_order:
            if level in bloom_analysis:
                stats = bloom_analysis[level]
                data.append({
                    'Bloom Level': level,
                    'Total Questions': stats['total'],
                    'Correct Answers': stats['correct'],
                    'Wrong Answers': stats['total'] - stats['correct'],
                    'Accuracy (%)': stats['percentage']
                })
        
        # Handle any levels not in standard order
        for level, stats in bloom_analysis.items():
            if level not in bloom_order:
                data.append({
                    'Bloom Level': level,
                    'Total Questions': stats['total'],
                    'Correct Answers': stats['correct'],
                    'Wrong Answers': stats['total'] - stats['correct'],
                    'Accuracy (%)': stats['percentage']
                })
        
        # Add overall statistics
        metadata = comprehensive_results['metadata']
        data.append({
            'Bloom Level': 'OVERALL',
            'Total Questions': metadata['total_questions'],
            'Correct Answers': metadata['correct_answers'],
            'Wrong Answers': metadata['wrong_answers'],
            'Accuracy (%)': metadata['overall_percentage']
        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create seaborn color palette
        cm = sns.light_palette("green", as_cmap=True)
        
        # Apply styling
        styled_df = df.style.background_gradient(cmap=cm, subset=['Accuracy (%)'])
        
        # Save styled DataFrame as HTML
        html_file = os.path.join(self.output_dir, f"styled_results_{model_name}.html")
        styled_df.to_html(html_file)
        
        # Save DataFrame as CSV for further analysis
        csv_file = os.path.join(self.output_dir, f"results_table_{model_name}.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"Styled results saved to: {html_file}")
        print(f"CSV results saved to: {csv_file}")
        
        return styled_df, df
    
    def process_existing_benchmark_files(self):
        """Process all existing benchmark JSON files and create styled DataFrames"""
        # Look for existing benchmark files
        json_pattern = os.path.join(self.output_dir, "detailed_benchmark_*.json")
        json_files = glob.glob(json_pattern)
        
        for json_file in json_files:
            try:
                # Extract model name from filename
                filename = os.path.basename(json_file)
                # Remove detailed_benchmark_results_ prefix and .json suffix
                model_part = filename.replace("detailed_benchmark_results_", "").replace(".json", "")
                
                # Try to extract model name (assuming format includes model name)
                # This is a heuristic - adjust based on your actual filename format
                model_name = model_part.split('_')[0] if '_' in model_part else model_part
                
                # Load the JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Create styled DataFrame
                styled_df, df = self.create_styled_dataframe(model_name, results)
                
                print(f"Processed {json_file} for model: {model_name}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    def aggregation(self):
        """Override aggregation to save detailed results"""
        # Calculate overall accuracy
        total_questions = len(self.detailed_results)
        correct_answers = sum(1 for r in self.detailed_results if r['is_correct'])
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Calculate Bloom level statistics
        bloom_analysis = {}
        for level, stats in self.bloom_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            bloom_analysis[level] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': accuracy,
                'percentage': accuracy * 100
            }
        
        # Prepare comprehensive results
        comprehensive_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "wrong_answers": total_questions - correct_answers,
                "overall_accuracy": overall_accuracy,
                "overall_percentage": overall_accuracy * 100
            },
            "bloom_taxonomy_analysis": bloom_analysis,
            "detailed_questions": self.detailed_results,
            "wrong_answers_detail": [r for r in self.detailed_results if not r['is_correct']],
            "correct_answers_detail": [r for r in self.detailed_results if r['is_correct']]
        }
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"detailed_benchmark_results_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        # Also save a summary report
        summary_file = os.path.join(self.output_dir, f"summary_report_{timestamp}.txt")
        self._save_summary_report(summary_file, comprehensive_results)
        
        # Create styled DataFrame for current run
        model_name = f"model_{timestamp}"  # Default name, can be customized
        self.create_styled_dataframe(model_name, comprehensive_results)
        
        # Process any existing benchmark files
        self.process_existing_benchmark_files()
        
        print(f"\nDetailed results saved to: {output_file}")
        print(f"Summary report saved to: {summary_file}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
        
        # Return the standard aggregation format expected by lm-evaluation-harness
        return {
            "exact_match": overall_accuracy
        }
    
    def _save_summary_report(self, filepath, results):
        """Save a human-readable summary report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            metadata = results['metadata']
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Total Questions: {metadata['total_questions']}\n")
            f.write(f"Correct Answers: {metadata['correct_answers']}\n")
            f.write(f"Wrong Answers: {metadata['wrong_answers']}\n")
            f.write(f"Overall Accuracy: {metadata['overall_accuracy']:.4f} ({metadata['overall_percentage']:.2f}%)\n\n")
            
            # Bloom Taxonomy Analysis
            f.write("BLOOM TAXONOMY LEVEL ANALYSIS\n")
            f.write("-" * 50 + "\n")
            bloom_analysis = results['bloom_taxonomy_analysis']
            
            # Sort by common Bloom taxonomy order
            bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            sorted_levels = sorted(bloom_analysis.items(), 
                                 key=lambda x: bloom_order.index(x[0]) if x[0] in bloom_order else 999)
            
            for level, stats in sorted_levels:
                f.write(f"{level}:\n")
                f.write(f"  Questions: {stats['total']}\n")
                f.write(f"  Correct: {stats['correct']}\n")
                f.write(f"  Wrong: {stats['total'] - stats['correct']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['percentage']:.2f}%)\n\n")
            
            # Wrong answers summary by Bloom level
            wrong_answers = results['wrong_answers_detail']
            f.write(f"WRONG ANSWERS BY BLOOM LEVEL ({len(wrong_answers)} total)\n")
            f.write("-" * 50 + "\n")
            
            # Group wrong answers by Bloom level
            wrong_by_level = defaultdict(list)
            for wrong in wrong_answers:
                wrong_by_level[wrong['bloom_level']].append(wrong)
            
            for level in bloom_order:
                if level in wrong_by_level:
                    f.write(f"\n{level} Level - {len(wrong_by_level[level])} wrong answers:\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, wrong in enumerate(wrong_by_level[level], 1):
                        f.write(f"{i}. ID: {wrong['question_id']}\n")
                        f.write(f"   Question: {wrong['question']}\n")
                        f.write(f"   Choices: {wrong['choices']}\n")
                        f.write(f"   Correct: {wrong['correct_answer']}\n")
                        f.write(f"   Model chose: {wrong['predicted_answer']}\n\n")
            
            # Handle any levels not in the standard order
            for level, wrongs in wrong_by_level.items():
                if level not in bloom_order:
                    f.write(f"\n{level} Level - {len(wrongs)} wrong answers:\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, wrong in enumerate(wrongs, 1):
                        f.write(f"{i}. ID: {wrong['question_id']}\n")
                        f.write(f"   Question: {wrong['question']}\n")
                        f.write(f"   Choices: {wrong['choices']}\n")
                        f.write(f"   Correct: {wrong['correct_answer']}\n")
                        f.write(f"   Model chose: {wrong['predicted_answer']}\n\n")

# Utility function to process existing benchmark files separately
def generate_styled_reports_for_existing_files():
    """Standalone function to generate styled reports for existing benchmark files"""
    output_dir = "detailed_benchmark_results"
    
    # Look for existing benchmark files
    json_pattern = os.path.join(output_dir, "detailed_benchmark_*.json")
    json_files = glob.glob(json_pattern)
    
    for json_file in json_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(json_file)
            
            # Try to extract model name from existing files in detailed_benchmark_results
            # Look for patterns like detailed_benchmark_results_ModelName_palestine.json
            if "_palestine" in filename:
                model_name = filename.replace("detailed_benchmark_results_", "").replace("_palestine.json", "")
            else:
                # Fallback pattern
                model_part = filename.replace("detailed_benchmark_results_", "").replace(".json", "")
                model_name = model_part.split('_')[0] if '_' in model_part else model_part
            
            # Load the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Create DataFrame
            bloom_analysis = results['bloom_taxonomy_analysis']
            data = []
            bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            
            for level in bloom_order:
                if level in bloom_analysis:
                    stats = bloom_analysis[level]
                    data.append({
                        'Bloom Level': level,
                        'Total Questions': stats['total'],
                        'Correct Answers': stats['correct'],
                        'Wrong Answers': stats['total'] - stats['correct'],
                        'Accuracy (%)': stats['percentage']
                    })
            
            # Handle any levels not in standard order
            for level, stats in bloom_analysis.items():
                if level not in bloom_order:
                    data.append({
                        'Bloom Level': level,
                        'Total Questions': stats['total'],
                        'Correct Answers': stats['correct'],
                        'Wrong Answers': stats['total'] - stats['correct'],
                        'Accuracy (%)': stats['percentage']
                    })
            
            # Add overall statistics
            metadata = results['metadata']
            data.append({
                'Bloom Level': 'OVERALL',
                'Total Questions': metadata['total_questions'],
                'Correct Answers': metadata['correct_answers'],
                'Wrong Answers': metadata['wrong_answers'],
                'Accuracy (%)': metadata['overall_percentage']
            })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Create seaborn color palette
            cm = sns.light_palette("green", as_cmap=True)
            
            # Apply styling
            styled_df = df.style.background_gradient(cmap=cm, subset=['Accuracy (%)'])
            
            # Save styled DataFrame as HTML
            html_file = os.path.join(output_dir, f"styled_results_{model_name}.html")
            styled_df.to_html(html_file)
            
            # Save DataFrame as CSV for further analysis
            csv_file = os.path.join(output_dir, f"results_table_{model_name}.csv")
            df.to_csv(csv_file, index=False)
            
            print(f"Processed {json_file}")
            print(f"  Model: {model_name}")
            print(f"  Styled HTML: {html_file}")
            print(f"  CSV: {csv_file}")
            print()
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    # Run this to generate styled reports for existing files
    generate_styled_reports_for_existing_files()