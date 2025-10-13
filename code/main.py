# imports
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from search_engine import Search

class Document(BaseModel):
    text:str

class DocumentList(BaseModel):
    documents: List[Document]

# instance of fastapi class
app = FastAPI()
# instance of search class
search = Search()

# API endpoints

# define get
# called when someone visits main url
@app.get("/")
def read_root():
    return {"message": "Search Engine API is running"}

@app.post("/index")
def index_documents(data: DocumentList):
    documents_text = [doc.text for doc in data.documents]

    search.index(documents_text)

    return {"message": f"Successfully indexed {len(documents_text)} documents."}