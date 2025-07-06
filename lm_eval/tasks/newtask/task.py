# from lm_eval.api.task import ConfigurableTask
# from datasets import Dataset
# import os
# import json
# import numpy as np
# def load_my_test_dataset(metadata=None, **kwargs):
#     data_path = kwargs.get("data_path", "lm_eval/tasks/newtask/converted_benchmark_with_bloom.jsonl")

#     with open(data_path, "r", encoding="utf-8") as f:
#         lines = []
#         for line in f:
#             line = line.strip()
#             # Skip empty lines and comment lines starting with //
#             if line and not line.startswith('//'):
#                 try:
#                     lines.append(json.loads(line))
#                 except json.JSONDecodeError:
#                     # Skip any lines that can't be parsed as JSON
#                     continue
    
#     dataset = Dataset.from_list(lines)
#     return {
#         "test": dataset
#     }

# class NEWTASK(ConfigurableTask):
#     VERSION = 0
#     DATASET_PATH = "json"  
#     DATASET_NAME = None

#     def __init__(self, config=None):
#         super().__init__(config={
#             "custom_dataset": load_my_test_dataset,
#             "metadata": {"version": self.VERSION},
#         })
#         self.OUTPUT_TYPE = "multiple_choice"

#     def has_training_docs(self):
#         return False

#     def has_validation_docs(self):
#         return False

#     def has_test_docs(self):
#         return True

#     def test_docs(self):
#         return self.dataset["test"]

#     def doc_to_text(self, doc):
#         return f"Question: {doc['query']}\nAnswer:"

#     def doc_to_target(self, doc):
#         # Return the correct answer string
#         return doc["choices"][doc["gold"]]

#     def doc_to_choice(self, doc):
#         # Return all multiple choice options
#         return doc["choices"]
  
#     def process_results(self, doc, results):
#         print(doc,results)
#         # Get index of best loglikelihood score
#         pred_idx = int(np.argmax([r[0] for r in results]))
#         gold_idx = doc["gold"]
#         return {
#             "exact_match": pred_idx == gold_idx
#         }




from lm_eval.api.task import ConfigurableTask
from datasets import Dataset
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

def load_my_test_dataset(metadata=None, **kwargs):
    data_path = kwargs.get("data_path", "lm_eval/tasks/newtask/questions_benchmark1.jsonl")

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
        self.output_dir = "detailed_benchmark_results_sports"
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
        output_file = os.path.join(self.output_dir, f"detailed_benchmark_results_Sports{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        # Also save a summary report
        summary_file = os.path.join(self.output_dir, f"summary_report_{timestamp}.txt")
        self._save_summary_report(summary_file, comprehensive_results)
        
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