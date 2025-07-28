# aleks_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import re
from datetime import datetime
import traceback 
import asyncio 

# Import core Aleks functions and constants from the refactored files
from aleks_core import initialize_aleks_components, get_rag_response_stream, detect_document_request 
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR 


# --- FastAPI App Setup ---
app = FastAPI(
    title="Aleks AI API",
    description="API for Aleks - AI Legal Assistant",
    version="1.0.0",
)

# --- CORS Configuration ---
origins = [
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Models (Pydantic for data validation) ---
class ChatRequest(BaseModel):
    message: str

class DocumentFillRequest(BaseModel):
    template_key: str
    filled_data: dict

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """
    Initializes Aleks components when the FastAPI application starts.
    This ensures the LLM and vector store are loaded once.
    """
    print("Starting up Aleks API...")
    try:
        initialize_aleks_components() 
        print("Aleks API ready!")
    except Exception as e:
        print(f"Failed to initialize Aleks components: {e}. Please check your setup (Ollama, ChromaDB, etc.).")
        traceback.print_exc() 
        raise 

# FIX: Changed endpoint path back to "/api/chat" to match the incoming request path from Netlify
@app.post("/api/chat") 
async def chat_with_aleks(request: ChatRequest):
    """
    Main chat endpoint. Detects document requests or performs RAG.
    """
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    detected_doc_type = detect_document_request(user_message)

    if detected_doc_type != "NONE":
        template_filename = DOCUMENT_TEMPLATES.get(detected_doc_type)
        if not template_filename:
            return {"type": "text", "response": f"Sorry, I don't have a template for '{detected_doc_type}'."}

        template_path = os.path.join(TEMPLATE_DIR, template_filename)
        if not os.path.exists(template_path):
            return {"type": "text", "response": f"Sorry, the template file for '{detected_doc_type}' could not be found."}

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            return {"type": "text", "response": f"Error reading template: {e}"}

        placeholders = set(re.findall(r'\[(.*?)\]|\{\{(.*?)\}\}', template_content))
        placeholders = {p.strip() for tup in placeholders for p in tup if p.strip()}
        
        if 'current_date' in placeholders:
            placeholders.remove('current_date')

        placeholder_details = []
        for p in sorted(list(placeholders)):
            description = PLACEHOLDER_DESCRIPTIONS.get(p, p.replace('_', ' ').title())
            placeholder_details.append({"name": p, "description": description})
        
        return {
            "type": "document_request",
            "document_type": detected_doc_type,
            "message": f"Okay, let's fill out your '{detected_doc_type}' template. Please provide the following details:",
            "placeholders_to_fill": placeholder_details
        }
    else:
        return StreamingResponse(get_rag_response_stream(user_message), media_type="text/event-stream")

@app.post("/api/generate_document")
async def generate_document(request: DocumentFillRequest):
    """
    Generates the final document after all placeholders are filled.
    """
    template_key = request.template_key
    filled_data = request.filled_data

    template_filename = DOCUMENT_TEMPLATES.get(template_key)
    if not template_filename:
        raise HTTPException(status_code=400, detail=f"No template found for '{template_key}'.")

    template_path = os.path.join(TEMPLATE_DIR, template_filename)
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail=f"Template file '{template_filename}' not found.")

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading template file: {e}")

    if 'current_date' in template_content:
        filled_data['current_date'] = datetime.now().strftime("%B %d, %Y")

    filled_document = template_content
    for placeholder, value in filled_data.items():
        filled_document = re.sub(rf"\[{re.escape(placeholder)}\]|" + r"\{\{" + re.escape(placeholder) + r"\}\}", str(value), filled_document)

    output_filename = f"filled_{template_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(TEMPLATE_DIR, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(filled_document)
        print(f"Document saved as '{output_filename}' in '{TEMPLATE_DIR}'.")
    except Exception as e:
        print(f"Error saving document: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving document: {e}")

    return {
        "status": "success",
        "message": f"Document '{output_filename}' generated and saved.",
        "generated_document_preview": filled_document 
    }
