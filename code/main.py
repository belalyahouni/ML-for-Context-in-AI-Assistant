# imports
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
# import files
from search_engine import Search

class Document(BaseModel):
    text:str

class DocumentList(BaseModel):
    documents: List[Document]

# instance of fastapi class
app = FastAPI()
# instance of search class
search_instance = Search()

# API endpoints

# define get
# called when someone visits main url
@app.get("/")
def read_root():
    return {"message": "Search Engine API is running"}

@app.post("/index")
def index_documents(data: DocumentList):
    documents_text = [doc.text for doc in data.documents]

    search_instance.index(documents_text)

    return {"message": f"Successfully indexed {len(documents_text)} documents."}

@app.get("/search")
def search(query: str):
    # get query from /search
    try:
        results = search_instance.search(query)
        if not results:
            return {"message": "No documents found matching your query."}
        return {"results": results}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Please index documents first using the /index endpoint.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {e}")
