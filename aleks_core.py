# aleks_core.py
import os
import re
from datetime import datetime
import traceback 

# Core LangChain components for RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# REMOVED: from langchain_ollama import OllamaLLM 

# NEW: Import for Google Generative AI
import google.generativeai as genai 

# REMOVED: from langchain.chains import RetrievalQA # Not directly used with raw genai.GenerativeModel for RAG
from langchain_core.prompts import PromptTemplate
# REMOVED: from langchain.chains import LLMChain # Not directly used with raw genai.GenerativeModel for classification

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
GEMINI_MODEL_NAME = "gemini-pro" # Or gemini-1.5-flash for faster/cheaper inference

# Global variables for the AI components (will be initialized once)
llm = None # This will now be our Gemini model instance
retriever = None 

async def initialize_aleks_components(): # Made async
    """
    Initializes the RAG components and the LLM (Gemini), making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global llm, retriever 
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
    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"WARNING: Chroma DB directory '{CHROMA_DB_DIR}' not found. Please run 'python vector_db_creator.py' to create it.")
            retriever = None
        else:
            retriever = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
            print(f"Loaded ChromaDB from {CHROMA_DB_DIR}")
    except Exception as e:
        print(f"Error loading ChromaDB from {CHROMA_DB_DIR}: {e}")
        print("Please ensure 'langchain-chroma' is installed and your database exists and is not corrupted.")
        retriever = None 

    print("Aleks AI components initialized successfully!")

# MODIFIED: get_rag_response function to use Gemini API and accept language
async def get_rag_response(query: str, language: str = "en") -> dict: # Added language parameter
    """
    Performs RAG query using the initialized LLM (Gemini) and retriever.
    Returns the full answer and sources.
    """
    global llm, retriever 

    if llm is None or retriever is None:
        await initialize_aleks_components() # Ensure components are initialized

    if retriever is None: # Check again after initialization attempt
        return {
            "response": "I'm sorry, the legal document database is not available right now. Please try again later.",
            "sources": []
        }

    try:
        # 1. Retrieve relevant documents
        print(f"DEBUG: Starting document retrieval for query: '{query}'")
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"DEBUG: Retrieved {len(docs)} documents.")

        # Define RAG prompt template for Gemini
        rag_template = """You are Aleks, an AI legal assistant knowledgeable in Philippine law.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't have enough information from the provided legal documents.
        Limit your answer to 200 words.

        Context: {context}

        Question: {question}

        Answer in Filipino unless the question is in English. If the question is in English, answer in English.
        """
        rag_prompt_template = PromptTemplate.from_template(rag_template)
        
        # Prepare content for Gemini API, including language instruction
        prompt_parts = [
            {"text": rag_prompt_template.format(context=context_text, question=query, language=language)}
        ]

        print(f"DEBUG: Sending prompt to Gemini: {prompt_parts[0]['text'][:200]}...") 

        # 2. Call the Gemini API
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
        print(f"DEBUG: Received response from Gemini: {generated_text[:200]}...")

        # Format source documents nicely for API response
        sources_info = []
        for doc in docs:
            source_name = doc.metadata.get('source', 'Unknown Document')
            start_index = doc.metadata.get('start_index', 'N/A')
            sources_info.append({
                "source": source_name,
                "startIndex": start_index,
                "snippet": doc.page_content[:200] + "..." 
            })

        return {
            "response": generated_text,
            "sources": sources_info
        }

    except Exception as e:
        print(f"CRITICAL ERROR IN RAG RESPONSE (Gemini API): {e}")
        traceback.print_exc()
        return {
            "response": "I apologize, but I encountered an issue while processing your request. Please try again or rephrase your question.",
            "sources": []
        }

# MODIFIED: detect_document_request function to use Gemini API and accept language
async def detect_document_request(query: str, template_names: list, language: str = "en") -> str: # Added language parameter
    """
    Uses the LLM (Gemini) to determine if the query is a request for a document template
    and identifies which document type, considering the user's language.
    """
    global llm 

    if llm is None:
        await initialize_aleks_components() 

    formatted_template_names = ", ".join(template_names)

    # Craft a precise prompt for Gemini for classification.
    # It's crucial to instruct the LLM to respond ONLY with the document type or "NONE".
    prompt_text = f"""You are an AI assistant specializing in identifying requests for legal document templates.
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
                "temperature": 0.0, # Keep temperature very low for deterministic classification
                "max_output_tokens": 50 # Limit output length to prevent verbose responses
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
