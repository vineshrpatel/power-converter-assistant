import json
import os
from qa_metrics.f1 import f1_score_with_precision_recall
from qa_metrics.pedant import PEDANT

def extract_scores_and_texts(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    references = []
    candidates = []
    queries = []

    for entry in data:
        queries.append(entry.get("evolved_query", ""))
        references.append(entry.get("ref_ans", ""))
        candidates.append(entry.get("generated_text", ""))

    return queries, references, candidates

def calc_f1(reference, candidate):
    f1_stats = f1_score_with_precision_recall(reference, candidate)
    return f1_stats

def pendant(question, reference, candidate):
    pedant = PEDANT()
    pendant_score = pedant.get_score(reference, candidate, question)
    return pendant_score

def load_results_file(results_file):
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {"scores": []}

def save_results(results, results_file):
    with open(results_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    input_json_file = "basemodel_generation.json"  # For fine-tuned model change to finetuned_generation.json
    results_file = "f1pendant_base.json"  # For fine-tuned model change to f1pendant_finetuned.json
    
    # Load existing results if any (for resuming)
    results = load_results_file(results_file)
    
    # Find the highest index already processed
    last_index = -1
    if results["scores"]:
        last_index = max(item["index"] for item in results["scores"])
    
    queries, references, candidates = extract_scores_and_texts(input_json_file)
    
    # Process only items that haven't been processed yet
    for i, (query, ref, cand) in enumerate(zip(queries, references, candidates)):
        if i <= last_index:
            continue
            
        try:
            print(f"Processing item {i+1}/{len(queries)}...")
            
            # Calculate F1 score
            f1_result = calc_f1(ref, cand)
            
            # Calculate PEDANT score
            pedant_score = pendant(query, ref, cand)
            
            # Add results to our tracking
            results["scores"].append({
                "index": i,
                "f1": f1_result["f1"],
                "pendant": pedant_score
            })
            
            # Save after each item to enable resuming
            save_results(results, results_file)
            
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            # Save progress so far
            save_results(results, results_file)
            raise
