# imports
from sentence_transformers import SentenceTransformer
import numpy as np
from usearch.index import Index
import json
from typing import List, Optional # FIXED: Import Optional
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

    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """
        Takes in documents and encodes them, storing the embeddings and index.
        """
        # 1. Determine the keys for our map
        if doc_ids:
            map_keys = doc_ids
        else:
            map_keys = [str(i) for i in range(len(documents))]

        # 2. Create the single document map
        document_map = {key: doc for key, doc in zip(map_keys, documents)}

        # Encode the Documents
        print(f"Generating embeddings for {len(documents)} documents...")
        document_embeddings = self.model.encode(documents).astype(np.float32)
        print("Embeddings generated.")

        # Initialize and build the index
        index = Index(ndim=self.VECTOR_SIZE, metric='cos')
        keys = np.arange(len(documents))
        index.add(keys, document_embeddings)
        print(f"Index built. Total vectors in index: {len(index)}")

        # Save the index to disk
        index.save(self.INDEX_PATH)
        print(f"Index saved to {self.INDEX_PATH}")

        # We save the document map AND the ordered list of its keys
        data_to_save = {
            "map_keys": map_keys,
            "document_map": document_map
        }
        with open(self.DOCUMENTS_PATH, 'w') as f:
            json.dump(data_to_save, f)
        print(f"Document map saved to {self.DOCUMENTS_PATH}")

    def search(self, query: str, top_k: int = 1, return_ids: bool = False) -> List[str]:
        """
        Searches the embeddings, finds cos similarity matches, looks at the path in json, returns it.
        """
        # Check if the index file exists
        if not os.path.exists(self.INDEX_PATH):
            return []

        # Load the index and the document map
        index = Index.restore(self.INDEX_PATH)
        
        # FIXED: Load data into 'saved_data' then extract the map and keys from it.
        with open(self.DOCUMENTS_PATH, 'r') as f:
            saved_data = json.load(f)

        map_keys = saved_data['map_keys']
        document_map = saved_data['document_map']

        # Encode query
        print(f"Searching for: '{query}'")
        query_embedding = self.model.encode(query).astype(np.float32).reshape(1, -1)

        # Perform search
        results = index.search(query_embedding, count=top_k)

        if return_ids:
                found_doc_ids = [map_keys[internal_key] for internal_key in results.keys]
                return found_doc_ids

        # Translate results and retrieve documents
        found_documents = []
        for internal_key in results.keys:
            real_map_key = map_keys[internal_key]
            document_text = document_map.get(real_map_key)
            if document_text:
                found_documents.append(document_text)

        return found_documents