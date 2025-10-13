import requests
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm # A library for progress bars, run `pip install tqdm`

# --- 1. CONFIGURATION ---
BASE_URL = "http://127.0.0.1:8000"

# --- 2. LOAD AND PREPARE THE DATASET ---
print("Loading the CoIR-Retrieval/cosqa dataset components...")
# Load the code snippets
corpus = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
# Load the queries
queries = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
# Load the validation set which links queries to code
test_set = load_dataset("CoIR-Retrieval/cosqa", split="test")
print("Dataset components loaded.")

# Create fast lookup dictionaries for code and queries
corpus_map = {item['_id']: item['text'] for item in corpus}
queries_map = {item['_id']: item['text'] for item in queries}

# Prepare the data for evaluation
all_code_snippets = list(corpus_map.values())
queries_with_answers = [
    {"query": queries_map[item['query-id']], "correct_code": corpus_map[item['corpus-id']]}
    for item in test_set
]

# --- 3. INDEX THE CODE SNIPPETS ---
print(f"\nIndexing {len(all_code_snippets)} code snippets...")
# Format data as in-memory text files for the API
upload_files = [
    ('files', (f'doc_{i}.txt', code, 'text/plain'))
    for i, code in enumerate(all_code_snippets)
]

try:
    response = requests.post(f"{BASE_URL}/index", files=upload_files)
    response.raise_for_status()
    print("✅ Indexing successful!")
except requests.exceptions.RequestException as e:
    print(f"❌ Error during indexing: {e}")
    exit()
finally:
    for _, file_tuple in upload_files:
        file_tuple[1].close()

# --- 4. METRIC IMPLEMENTATION (Unchanged) ---
def calculate_metrics(results, correct_answer):
    rank = -1
    for i, result in enumerate(results):
        # We now compare the 'context' field which holds the full sentence's source
        if result['context'] == correct_answer:
            rank = i + 1
            break
            
    recall_at_10 = 1 if rank != -1 and rank <= 10 else 0
    mrr_at_10 = 1 / rank if rank != -1 and rank <= 10 else 0
    ndcg_at_10 = 1 / np.log2(rank + 1) if rank != -1 and rank <= 10 else 0
    
    return recall_at_10, mrr_at_10, ndcg_at_10

# --- 5. EVALUATION LOOP ---
print("\n--- Starting Evaluation ---")
scores = []

# Using tqdm for a nice progress bar
for item in tqdm(queries_with_answers, desc="Evaluating Queries"):
    query = item['query']
    correct_code = item['correct_code']
    
    try:
        response = requests.get(f"{BASE_URL}/search", params={'query': query})
        response.raise_for_status()
        search_results = response.json().get('results', [])
        
        metrics = calculate_metrics(search_results, correct_code)
        scores.append(metrics)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during search for query '{query}': {e}")

# --- 6. AGGREGATE AND PRINT RESULTS ---
if scores:
    recalls, mrrs, ndcgs = zip(*scores)
    mean_recall_at_10 = np.mean(recalls)
    mean_mrr_at_10 = np.mean(mrrs)
    mean_ndcg_at_10 = np.mean(ndcgs)

    print("\n--- Evaluation Results ---")
    print(f"Total Queries Evaluated: {len(scores)}")
    print(f"Recall@10: {mean_recall_at_10:.4f}")
    print(f"MRR@10:    {mean_mrr_at_10:.4f}")
    print(f"NDCG@10:   {mean_ndcg_at_10:.4f}")
else:
    print("Evaluation could not be completed.")