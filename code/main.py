# imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import fitz
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
async def index_documents(files: List[UploadFile] = File(...)):
    documents_text = []
    for file in files:
        contents = await file.read()
        
        # Check if the uploaded file is a PDF
        if file.content_type == 'application/pdf':
            try:
                # Open the PDF from the in-memory bytes
                pdf_doc = fitz.open(stream=contents, filetype="pdf")
                full_text = ""
                # Iterate through each page and extract text
                for page in pdf_doc:
                    full_text += page.get_text()
                documents_text.append(full_text)
                print(f"Successfully extracted text from PDF: {file.filename}")
            except Exception as e:
                print(f"Error processing PDF {file.filename}: {e}")
                continue # Skip to the next file
        
        # Handle plain text files as before
        elif file.content_type == 'text/plain':
            try:
                documents_text.append(contents.decode("utf-8"))
            except UnicodeDecodeError:
                print(f"Warning: Could not decode file '{file.filename}' as UTF-8. Skipping.")
                continue
        
        # Skip other file types
        else:
            print(f"Warning: Unsupported file type '{file.content_type}' for file '{file.filename}'. Skipping.")

    if not documents_text:
        raise HTTPException(status_code=400, detail="No valid text or PDF files were uploaded or processed.")

    search_instance.index(documents_text)
    return {"message": f"Successfully indexed content from {len(documents_text)} files."}

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
