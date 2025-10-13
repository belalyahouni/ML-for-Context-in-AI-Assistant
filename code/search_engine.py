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

#imports
from sentence_transformers import SentenceTransformer
import numpy as np
from usearch.index import Index
import json
from typing import List
import os

class Search:
    def __init__(self):
        """
        Defines the constants and loads the model.
        """
        # constants
        self.INDEX_PATH = "search_index.usearch"
        self.DOCUMENTS_PATH = "documents.json"
        self.VECTOR_SIZE = 384

        # Load the pretrained model from Hugging Face
        print("Loading model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

    def index(self, documents: List[str]):
        """
        Takes in documents and encodes them, storing the embeddings and index.
        """
        # Encode the Documents
        print(f"Generating embeddings for {len(documents)} documents...")
        document_embeddings = self.model.encode(documents).astype(np.float32)
        print("Embeddings generated.")

        # Initialize the index. We specify the number of dimensions (ndim) and the metric.
        index = Index(ndim=self.VECTOR_SIZE, metric='cos')

        # Create an array of integer keys for our documents
        keys = np.arange(len(documents))

        # Add the document vectors to the index with their corresponding keys
        index.add(keys, document_embeddings)
        print(f"Index built. Total vectors in index: {len(index)}")

        # Save the index to disk
        index.save(self.INDEX_PATH)
        print(f"Index saved to {self.INDEX_PATH}")

        # Create and save the document map (key -> text)
        document_map = {str(key): doc for key, doc in zip(keys, documents)}
        with open(self.DOCUMENTS_PATH, 'w') as f:
            json.dump(document_map, f)
        print(f"Document map saved to {self.DOCUMENTS_PATH}")
    def search(self, query: str, top_k: int = 1) -> List[str]:
        """
        Searches the embeddings, finds cos similarity matches, looks at the path in json, returns it.
        """
        # Check if the index file exists
        if not os.path.exists(self.INDEX_PATH):
            return [] # Or raise an error, or return a specific message

        # Load the index and the document map
        index = Index.restore(self.INDEX_PATH)
        with open(self.DOCUMENTS_PATH, 'r') as f:
            document_map = json.load(f)

        # Encode query
        print(f"Searching for: '{query}'")
        query_embedding = self.model.encode(query).astype(np.float32)

        # Perform search
        results = index.search(query_embedding, count=top_k)

        found_documents = [document_map[str(key)] for key in results.keys]
        return found_documents
