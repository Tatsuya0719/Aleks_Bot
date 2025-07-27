# aleks_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import re
from datetime import datetime
from typing import Optional # Import Optional for optional fields

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
# IMPORTANT: For production, change "*" to your specific Netlify URL (e.g., "https://your-site.netlify.app")
origins = [
    "*" # Temporarily allow all origins for debugging. CHANGE THIS FOR PRODUCTION!
    # "http://localhost",
    # "http://localhost:5173",
    # "http://10.147.18.65:5173", # Your Host Machine's ZeroTier IP and frontend port
    # "https://YOUR-NETLIFY-SITE-NAME.netlify.app", # <--- ADD YOUR ACTUAL NETLIFY URL HERE
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
    language: Optional[str] = "en" # NEW: Added optional language field

class DocumentFillRequest(BaseModel):
    template_key: str
    filled_data: dict
    language: Optional[str] = "en" # NEW: Added optional language field

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
        raise # Re-raise the exception to indicate a critical startup failure

@app.post("/api/chat")
async def chat_with_aleks(request: ChatRequest):
    """
    Main chat endpoint. Detects document requests or performs RAG.
    """
    user_message = request.message.strip()
    user_language = request.language # Get language from the request

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # 1. Detect if it's a document request
    # Pass the language to the detection function
    detected_doc_type = detect_document_request(user_message, user_language)

    if detected_doc_type != "NONE":
        template_filename = DOCUMENT_TEMPLATES.get(detected_doc_type)
        if not template_filename:
            # Localized response for missing template
            response_message = "Sorry, I don't have a template for that document type."
            if user_language == 'fil':
                response_message = "Paumanhin, wala akong template para sa uri ng dokumentong iyon."
            return {"type": "text", "response": response_message}

        template_path = os.path.join(TEMPLATE_DIR, template_filename)
        if not os.path.exists(template_path):
            # Localized response for missing template file
            response_message = "Sorry, the template file could not be found."
            if user_language == 'fil':
                response_message = "Paumanhin, hindi mahanap ang template file."
            return {"type": "text", "response": response_message}

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            # Localized response for template read error
            response_message = f"Error reading template: {e}"
            if user_language == 'fil':
                response_message = f"Error sa pagbasa ng template: {e}"
            return {"type": "text", "response": response_message}

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
        
        # Localized message for document request
        doc_request_message = f"Okay, let's fill out your '{detected_doc_type}' template. Please provide the following details:"
        if user_language == 'fil':
            doc_request_message = f"Sige, punan natin ang iyong '{detected_doc_type}' template. Mangyaring ibigay ang mga sumusunod na detalye:"

        return {
            "type": "document_request",
            "document_type": detected_doc_type,
            "message": doc_request_message,
            "placeholders_to_fill": placeholder_details
        }
    else:
        # 2. Perform RAG query
        try:
            # Pass the language to the RAG function
            rag_response = get_rag_response(user_message, user_language)
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
    user_language = request.language # Get language from the request

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

    # Localized success message
    success_message = f"Document '{output_filename}' generated and saved."
    if user_language == 'fil':
        success_message = f"Ang dokumento '{output_filename}' ay nabuo at nailigtas."

    return {
        "status": "success",
        "message": success_message,
        "generated_document_preview": filled_document # Provide a preview for the frontend
    }
