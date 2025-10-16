from datasets import load_dataset
import json
from search_engine import Search
from collections import defaultdict
from typing import List, Set
import numpy as np

def calculate_recall_at_10(found_ids: List[str], correct_ids: Set[str]) -> int:
    """
    Calculates Recall@10.
    Returns 1 if any of the correct IDs are in the found_ids list, otherwise 0.
    """
    return 1 if not set(found_ids).isdisjoint(correct_ids) else 0

def calculate_mrr_at_10(found_ids: List[str], correct_ids: Set[str]) -> float:
    """
    Calculates Mean Reciprocal Rank @ 10.
    Finds the rank of the first correct document and returns 1/rank.
    """
    for i, found_id in enumerate(found_ids):
        if found_id in correct_ids:
            return 1 / (i + 1)
    return 0.0

def calculate_ndcg_at_10(found_ids: List[str], correct_ids: Set[str]) -> float:
    """
    Calculates normalized Discounted Cumulative Gain @ 10.
    This metric considers the position of all relevant documents.
    """
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, found_id in enumerate(found_ids):
        if found_id in correct_ids:
            # Relevance is 1 if correct, 0 otherwise.
            # Add to DCG, discounting by position.
            dcg += 1 / np.log2(i + 2) # Use i+2 because log2(1) is 0

    # Calculate IDCG (Ideal DCG), the best possible score
    idcg = 0.0
    num_correct = min(len(correct_ids), 10) # Consider at most top 10 positions
    for i in range(num_correct):
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

# Load Corupus in
corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
print(f"Corpus loaded with {len(corpus_dataset)} documents.")

# Load the queries
queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
print(f"Queries loaded with {len(queries_dataset)} queries.")


# Load in test set.
eval_dataset = load_dataset("CoIR-Retrieval/cosqa", name="default", split="test")
print(f"Evaluation 'test' split loaded with {len(eval_dataset)} query-document pairs.")

documents = [item['text'] for item in corpus_dataset]
doc_ids = [item['_id'] for item in corpus_dataset]

# Initialise Search class object
search_instance = Search()
# Initilaise score arrays.
recall_scores = []
mrr_scores = []
ndcg_scores = []
# Index the Corpus

print("Building the search index... (This may take a while)")
search_instance.index(documents=documents, doc_ids=doc_ids)
print("\nIndexing complete! Your search engine is ready.")

# --- Main Evaluation Loop ---

print("\n--- Starting Search Loop for Evaluation ---")

# Reconstruct the dataset as a dictionary for ease of use.
queries_map = {item['_id']: item['text'] for item in queries_dataset}

ground_truth = defaultdict(set)
for item in eval_dataset:
    ground_truth[item['query-id']].add(item['corpus-id'])

test_query_ids = sorted(list(ground_truth.keys()))
print(f"Found {len(test_query_ids)} unique queries to test.")

# 3. Initialize the Search class (this will load the pre-built index)
search_instance = Search()

# 4. Loop through the first 5 queries and print the IDs found
for i, query_id in enumerate(test_query_ids):
    query_text = queries_map.get(query_id)
    if not query_text:
        continue
        
    # Get the set of all correct doc IDs for this query
    correct_doc_ids = ground_truth[query_id]
    # Call search with return_ids=True to get a list of found document IDs
    found_doc_ids = search_instance.search(query=query_text, top_k=10, return_ids=True)

    recall_scores.append(calculate_recall_at_10(found_doc_ids, correct_doc_ids))
    mrr_scores.append(calculate_mrr_at_10(found_doc_ids, correct_doc_ids))
    ndcg_scores.append(calculate_ndcg_at_10(found_doc_ids, correct_doc_ids))

    # Print progress intermittently
    if (i + 1) % 50 == 0:
        print(f"  ...processed {i+1}/{len(test_query_ids)} queries")
    
# --- Calculate and print the final average scores ---
if recall_scores:
    final_recall = np.mean(recall_scores)
    final_mrr = np.mean(mrr_scores)
    final_ndcg = np.mean(ndcg_scores)
    
    print("\n\n✅ Evaluation complete.")
    print("\n--- FINAL RESULTS ---")
    print(f"Recall@10: {final_recall:.4f}")
    print(f"MRR@10:    {final_mrr:.4f}")
    print(f"nDCG@10:   {final_ndcg:.4f}")

else:
    print("\nEvaluation could not be completed.")
