# implement a search engine (embeddings)
#   accepts a collection of docs (input)
#   provides API to search over this collection by text query (fast api?)(end product)

# get a pretrained model from huggingface -calc vector representations
#   vector storage + retrival -> some library
#   do an example on any source (report)

# 1. import a pretrained model DONE
# 2. make it accept input = multiple documents -> done but need to change to api later
# 3. make it generate embdeddings for the documents
# 4. store embeddings in a vector db
# 5. make it search with a text query
# 6. search query to vector
# 7. index db with vector
# 8. use new vectors found to go back to original
# 4. build api around it


from sentence_transformers import SentenceTransformer
import numpy as np
from usearch.index import Index

# Load the pretrained model from Hugging Face
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Documents
documents = [
    "The Canadian government has announced new immigration policies.",
    "Lionel Messi scored a hat-trick in the last match.",
    "Python is an interpreted, high-level, general-purpose programming language.",
    "The new iPhone features a faster processor and a better camera.",
    "Machine learning is a field of study in artificial intelligence."
]
print(f"Loaded {len(documents)} documents.")

# Encode the Documents
document_embeddings = model.encode(documents).astype(np.float32)
vector_size = document_embeddings.shape[1]
print(f"Document embeddings created with shape: {document_embeddings.shape}")

# Initialize the index. We specify the number of dimensions (ndim) and the metric.
# 'cos' is for cosine similarity, which is great for sentence embeddings.
index = Index(ndim=vector_size, metric='cos')

# Create an array of integer keys for our documents
keys = np.arange(len(documents))

# Add the document vectors to the index with their corresponding keys
index.add(keys, document_embeddings)
print(f"Index built. Total vectors in index: {len(index)}")

# Perform a Search ---
query = "what is a good language for AI?"
print(f"\nSearching for: '{query}'")

# Encode the query and ensure it's the correct data type
query_embedding = model.encode(query).astype(np.float32)

# Search the index for the top 2 closest vectors
# The 'search' method returns a results object
results = index.search(query_embedding, count=2)

# --- 6. Show the Results ---
print(f"\nTop 2 results:")
# The 'keys' attribute of the results contains the integer keys we added
for key in results.keys:
    print(f"  - Document {key}: {documents[key]}")
