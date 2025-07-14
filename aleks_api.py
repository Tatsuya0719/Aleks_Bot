# aleks_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import re
from datetime import datetime

# Import core aleks functions and constants from the refactored file
from aleks_core import initialize_aleks_components, get_rag_response, detect_document_request
# Import document related constants from document_manager
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR 


# --- FastAPI App Setup ---
app = FastAPI(
    title="Aleks AI API",
    description="API for Aleks - AI Legal Assistant",
    version="1.0.0",
)

# --- CORS Configuration ---
# You might need to adjust `allow_origins` based on where your React app is served.
# During development, "http://localhost:5173" (Vite's default) is common.
# In production, this would be your actual website domain.
origins = [
    "http://localhost",
    "http://localhost:5173", # Default Vite dev server port
    "http://10.147.18.65:5173"
    # Add your deployed website domain here, e.g., "https://your-bolt-app.com"
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
        # In a production app, you might want more sophisticated error handling,
        # but for now, this will clearly show if initialization failed.
        raise # Re-raise the exception to indicate a critical startup failure

@app.post("/api/chat")
async def chat_with_aleks(request: ChatRequest):
    """
    Main chat endpoint. Detects document requests or performs RAG.
    """
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # 1. Detect if it's a document request
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

        # Extract placeholders from the template
        placeholders = set(re.findall(r'\[(.*?)\]|\{\{(.*?)\}\}', template_content))
        placeholders = {p.strip() for tup in placeholders for p in tup if p.strip()}
        
        # Remove 'current_date' as it's auto-filled
        if 'current_date' in placeholders:
            placeholders.remove('current_date')

        # Get descriptions for the placeholders
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
        # 2. Perform RAG query
        try:
            rag_response = get_rag_response(user_message)
            return {"type": "rag_response", "response": rag_response["answer"], "sources": rag_response["sources"]}
        except Exception as e:
            # Catch any error from RAG and return as HTTPException
            raise HTTPException(status_code=500, detail=f"Error processing RAG query: {e}")

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

    # Add current_date automatically if placeholder exists
    if 'current_date' in template_content:
        filled_data['current_date'] = datetime.now().strftime("%B %d, %Y")

    filled_document = template_content
    for placeholder, value in filled_data.items():
        # Ensure value is string before substitution to avoid TypeError
        filled_document = re.sub(rf"\[{re.escape(placeholder)}\]|" + r"\{\{" + re.escape(placeholder) + r"\}\}", str(value), filled_document)

    # Save the document
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
        "generated_document_preview": filled_document # Provide a preview for the frontend
    }