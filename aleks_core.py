# aleks_core.py
import os
import re
from datetime import datetime
import traceback 

# Core LangChain components for RAG - make sure these are the updated ones
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# REMOVED: from langchain_ollama import OllamaLLM 

# NEW: Import for Google Generative AI
import google.generativeai as genai 

# REMOVED: from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# REMOVED: from langchain.chains import LLMChain 

# Import constants from document_manager
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR 

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- REMOVED: Ollama Configuration ---
# REMOVED: OLLAMA_BASE_URL = "http://localhost:11434"
# REMOVED: OLLAMA_MODEL_NAME = "phi3:mini"

# --- NEW: Gemini API Configuration ---
# This will fetch the key from the environment variables set on your VM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_MODEL_NAME = "gemini-pro" # Or gemini-1.5-flash for faster/cheaper

# Global variables for the AI components (will be initialized once)
qa_chain = None
llm = None # This will now be our Gemini model instance
retriever = None # Make retriever global for direct testing if needed

def initialize_aleks_components():
    """
    Initializes the RAG chain and LLM, making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global qa_chain, llm, retriever # Add retriever to global
    print("Initializing Aleks AI components...")

    # NEW: Configure Gemini API
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it on your VM before running the application.")
    genai.configure(api_key=GEMINI_API_KEY)

    # NEW: Initialize the Gemini model. This is now our 'llm'
    llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Initialized LLM: Google Gemini ({GEMINI_MODEL_NAME})")

    # Initialize Embeddings for RAG (keep HuggingFaceEmbeddings as is)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    print(f"Initialized Embeddings model: {EMBEDDINGS_MODEL_NAME}")

    # Initialize ChromaDB retriever
    # This will load an existing DB or create a new one if it doesn't exist
    try:
        retriever = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings).as_retriever()
        print(f"Loaded ChromaDB from {CHROMA_DB_DIR}")
    except Exception as e:
        print(f"Error loading ChromaDB from {CHROMA_DB_DIR}: {e}")
        print("Please ensure you have run 'python vector_db_creator.py' to create the vector database.")
        retriever = None # Ensure retriever is None if loading fails

    # REMOVED: Ollama LLM setup and RetrievalQA chain setup
    # If using Ollama, your original code for llm = OllamaLLM(...) and RetrievalQA.from_chain_type(...)
    # would go here. Now replaced by Gemini setup.

    print("Aleks AI components initialized successfully!")

# MODIFIED: get_rag_response function to use Gemini API
async def get_rag_response(query: str):
    global llm, retriever 

    if llm is None or retriever is None:
        await initialize_aleks_components() 

    try:
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # Define rag_template inside the function or globally if it's not already
        rag_template = """You are Aleks, an AI legal assistant knowledgeable in Philippine law.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't have enough information from the provided legal documents.
        Limit your answer to 200 words.

        Context: {context}

        Question: {question}

        Answer in Filipino unless the question is in English. If the question is in English, answer in English.
        """
        rag_prompt_template = PromptTemplate.from_template(rag_template)
        
        # Prepare content for Gemini API
        prompt_parts = [
            {"text": rag_prompt_template.format(context=context_text, question=query)}
        ]

        print(f"Sending prompt to Gemini: {prompt_parts[0]['text'][:200]}...") 

        response = await llm.generate_content(
            prompt_parts,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config={"temperature": 0.1, "max_output_tokens": 400} 
        )
        
        generated_text = response.text
        print(f"Received response from Gemini: {generated_text[:200]}...")

        return {
            "response": generated_text,
            "sources": [{"source": doc.metadata.get("source", "N/A"), "startIndex": "N/A", "snippet": doc.page_content[:200]} for doc in docs]
        }

    except Exception as e:
        print(f"CRITICAL ERROR IN RAG RESPONSE (Gemini API): {e}")
        traceback.print_exc()
        return {
            "response": "I apologize, but I encountered an issue while processing your request. Please try again or rephrase your question.",
            "sources": []
        }

# MODIFIED: detect_document_request function to use Gemini API
async def detect_document_request(query: str, template_names: list):
    global llm 

    if llm is None:
        await initialize_aleks_components() 

    formatted_template_names = ", ".join(template_names)

    # Simplified prompt for classification with Gemini
    prompt_text = f"""You are Aleks, an AI assistant specializing in identifying requests for legal document templates.
    Review the user's query and determine if they are asking for a legal document template.
    If they are, identify which specific document they are asking for from the following types: {formatted_template_names}.
    If you identify a document, respond ONLY with the exact document type from the provided list (e.g., "nda", "non-disclosure agreement").
    If the query is NOT a document request, respond ONLY with "NONE".
    Do not add any other text, greetings, or explanations to your response.

    Examples:
    User: I need an NDA.
    Response: nda

    User: Can you help me draft a non-disclosure agreement?
    Response: non-disclosure agreement

    User: What are the tax requirements for a new business?
    Response: NONE

    User: Draft a simple contract.
    Response: NONE

    User: Kailangan ko ng NDA. (I need an NDA.)
    Response: nda

    Query: {query}
    Response:"""

    try:
        response = await llm.generate_content(
            [{"text": prompt_text}],
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config={
                "temperature": 0.0, # Keep temperature low for deterministic classification
                "max_output_tokens": 50 # Limit output length
            }
        )
        
        detected_document_type = response.text.strip().lower()

        # Basic validation: ensure the response is one of the expected types or "none"
        if detected_document_type not in template_names and detected_document_type != "none":
            print(f"Warning: Gemini returned an unexpected document type: '{detected_document_type}'. Returning NONE.")
            return "NONE" 
        
        return detected_document_type
    except Exception as e:
        print(f"CRITICAL ERROR IN DOCUMENT DETECTION (Gemini API): {e}")
        traceback.print_exc()
        return "NONE"