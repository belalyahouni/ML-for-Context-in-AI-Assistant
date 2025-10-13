# imports
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    text:str

class DocumentList(BaseModel):
    documents: List[Document]

# instance of fastapi class
app = FastAPI()

# API endpoints

# define get
# called when someone visits main url
@app.get("/")
def read_root():
    return {"message": "Search Engine API is running"}

@app.post("/index")
def index_documents(data: DocumentList):
    num_docs = len(data.documents)
    return {"message": f"Successfully received {num_docs} documents for indexing."}