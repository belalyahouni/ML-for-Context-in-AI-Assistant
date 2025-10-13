# implement a search engine (embeddings)
#   accepts a collection of docs (input)
#   provides API to search over this collection by text query (fast api?)(end product)

# get a pretrained model from huggingface -calc vector representations
#   vector storage + retrival -> some library
#   do an example on any source (report)

# 1. import a pretrained model
# 2. make it accept input = multiple documents
# 3. make it generate embdeddings for the documents
# 4. store embeddings in a vector db
# 5. make it search with a text query
# 6. search query to vector
# 7. index db with vector
# 8. use new vectors found to go back to original
# 4. build api around it


from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load the pretrained model from Hugging Face
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# 2. Define a single sentence to be encoded
sentence = "This is a test sentence."

# 3. Use the model's encode() method to get the embedding
embedding = model.encode(sentence)

# 4. Print the results to see what it looks like
print("\nOriginal sentence:", sentence)
print("\nEmbedding (vector) shape:", embedding.shape)
print("\nFirst 5 values of the embedding:", embedding[:5])

# The output is a NumPy array. You can see its type.
print("\nType of embedding:", type(embedding))