#!/usr/bin/env python3
"""
Script to generate styled pandas DataFrames with seaborn color mapping
for existing benchmark JSON files in detailed_benchmark_results directory.
"""

import os
import json
import pandas as pd
import seaborn as sns
import glob
from collections import defaultdict

def generate_combined_styled_report():
    """Generate a single combined styled report for all models"""
    output_dir = "detailed_benchmark_results"
    
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist!")
        return
    
    # Look for existing benchmark files
    json_pattern = os.path.join(output_dir, "detailed_benchmark_*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No benchmark JSON files found in {output_dir}")
        return
    
    print(f"Found {len(json_files)} benchmark files to process...")
    print()
    
    all_model_data = []
    
    for json_file in json_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(json_file)
            
            # Handle different filename patterns
            if "_palestine" in filename:
                # Pattern: detailed_benchmark_results_ModelName_palestine.json
                model_name = filename.replace("detailed_benchmark_results_", "").replace("_palestine.json", "")
            elif filename.startswith("detailed_benchmark_results_"):
                # Pattern: detailed_benchmark_results_timestamp.json or similar
                model_part = filename.replace("detailed_benchmark_results_", "").replace(".json", "")
                # Try to extract model name from the beginning
                model_name = model_part.split('_')[0] if '_' in model_part else model_part
            else:
                model_name = filename.replace(".json", "")
            
            print(f"Processing: {filename}")
            print(f"Model name: {model_name}")
            
            # Load the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Extract bloom taxonomy analysis and metadata
            bloom_analysis = results.get('bloom_taxonomy_analysis', {})
            metadata = results.get('metadata', {})
            
            if not bloom_analysis:
                print(f"  Warning: No bloom_taxonomy_analysis found in {filename}")
                continue
            
            # Create row for this model
            model_row = {'Model': model_name}
            
            # Add overall statistics
            model_row['Overall_Total'] = metadata.get('total_questions', 0)
            model_row['Overall_Correct'] = metadata.get('correct_answers', 0)
            model_row['Overall_Accuracy'] = metadata.get('overall_percentage', 0)
            
            # Add Bloom level statistics
            bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            
            for level in bloom_order:
                if level in bloom_analysis:
                    stats = bloom_analysis[level]
                    model_row[f'{level}_Total'] = stats['total']
                    model_row[f'{level}_Correct'] = stats['correct']
                    model_row[f'{level}_Accuracy'] = stats['percentage']
                else:
                    model_row[f'{level}_Total'] = 0
                    model_row[f'{level}_Correct'] = 0
                    model_row[f'{level}_Accuracy'] = 0
            
            # Handle any non-standard levels
            for level, stats in bloom_analysis.items():
                if level not in bloom_order:
                    model_row[f'{level}_Total'] = stats['total']
                    model_row[f'{level}_Correct'] = stats['correct']
                    model_row[f'{level}_Accuracy'] = stats['percentage']
            
            all_model_data.append(model_row)
            print(f"  ✓ Processed {model_name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_file}: {e}")
    
    if not all_model_data:
        print("No valid model data found!")
        return
    
    # Create combined DataFrame
    df = pd.DataFrame(all_model_data)
    
    # Sort by overall accuracy (descending)
    df = df.sort_values('Overall_Accuracy', ascending=False)
    df = df.reset_index(drop=True)
    
    # Create seaborn green color palette
    cm = sns.light_palette("green", as_cmap=True)
    
    # Apply styling with background gradient for accuracy columns
    accuracy_columns = [col for col in df.columns if col.endswith('_Accuracy')]
    
    styled_df = df.style.background_gradient(
        cmap=cm, 
        subset=accuracy_columns,
        vmin=0,
        vmax=100
    ).format({
        col: '{:.2f}%' for col in accuracy_columns
    }).set_properties(**{
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold'), ('background-color', '#90EE90')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]},
    ]).set_caption("Combined Model Benchmark Results - Bloom Taxonomy Analysis (Green Palette)")
    
    # Save combined CSV
    csv_file = os.path.join(output_dir, "combined_benchmark_results_all_models.csv")
    df.to_csv(csv_file, index=False)
    
    # Save styled HTML
    html_file = os.path.join(output_dir, "combined_benchmark_results_all_models_styled.html")
    styled_df.to_html(html_file, escape=False)
    
    print("\n" + "=" * 60)
    print("COMBINED RESULTS GENERATED")
    print("=" * 60)
    print(f"✓ Combined CSV saved: {csv_file}")
    print(f"✓ Styled HTML saved: {html_file}")
    print(f"✓ Total models processed: {len(all_model_data)}")
    
    # Display summary table
    print("\nModel Performance Summary:")
    summary_cols = ['Model', 'Overall_Accuracy'] + [f'{level}_Accuracy' for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"] if f'{level}_Accuracy' in df.columns]
    summary_df = df[summary_cols]
    print(summary_df.to_string(index=False, float_format='{:.2f}'.format))
    
    return df


if __name__ == "__main__":
    print("Generating combined styled benchmark report with green palette...")
    print("=" * 60)
    
    try:
        combined_df = generate_combined_styled_report()
        if combined_df is not None:
            print("\nSuccessfully generated combined styled report!")
            print("Open the HTML file in your browser to view the styled results!")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nDone!")
