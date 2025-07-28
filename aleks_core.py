# aleks_core.py
import os
import re
from datetime import datetime
import traceback 

# Core LangChain components for RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# NEW: Import for Google Generative AI
import google.generativeai as genai 

from langchain_core.prompts import PromptTemplate

# Import constants from document_manager
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR 

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- NEW: Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_MODEL_NAME = "gemini-1.5-flash" 

# Global variables for the AI components (will be initialized once)
llm = None 
retriever = None 

def initialize_aleks_components(): 
    """
    Initializes the RAG components and the LLM (Gemini), making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global llm, retriever 
    print("Initializing Aleks AI components...")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it on your VM before running the application.")
    genai.configure(api_key=GEMINI_API_KEY)

    try:
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Initialized LLM: Google Gemini ({GEMINI_MODEL_NAME})")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini model '{GEMINI_MODEL_NAME}'. Check model availability and API key: {e}")
        traceback.print_exc()
        raise 

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    print(f"Initialized Embeddings model: {EMBEDDINGS_MODEL_NAME}")

    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"WARNING: Chroma DB directory '{CHROMA_DB_DIR}' not found. Please run 'python vector_db_creator.py' to create it.")
            retriever = None
        else:
            retriever = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings).as_retriever(search_kwargs={"k": 5}) 
            print(f"Loaded ChromaDB from {CHROMA_DB_DIR}")
    except Exception as e:
        print(f"Error loading ChromaDB from {CHROMA_DB_DIR}: {e}")
        print("Please ensure 'langchain-chroma' is installed and your database exists and is not corrupted.")
        retriever = None 

    print("Aleks AI components initialized successfully!")

def get_rag_response(query: str, language: str = "en") -> dict: 
    global llm, retriever 

    if llm is None or retriever is None:
        initialize_aleks_components() 

    if retriever is None: 
        return {
            "response": "I'm sorry, the legal document database is not available right now. Please try again later.",
            "sources": []
        }

    try:
        print(f"DEBUG: Starting document retrieval for query: '{query}'")
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"DEBUG: Retrieved {len(docs)} documents.")

        # MODIFIED RAG PROMPT: Example of customizing AI's chat style
        rag_template = """You are Aleks, a highly knowledgeable, professional, and helpful AI legal assistant for Filipino citizens.
        Your goal is to provide clear, accurate, and concise information based ONLY on the provided legal documents.
        If you can correlate some information to other legal documents that you think are relevant that you currently have, you may, but don't guess answers.
        Present your answers directly and professionally, avoiding overly casual language.
        Limit your answer to 300 words.
        If you could not answer the questions confidently, tell the user that one of your functionalities is that you could also redirect them to our partner Law Firms. Do not make up information.
        If you could detect that, on your answer, there might be a legal document that is involved, tell the user that one of your functionalities is to generate that document for them, to be directed to proper authorities.

        Context: {context}

        Question: {question}

        Answer in Filipino unless the question is in English. If the question is in English, answer in English.
        """
        rag_prompt_template = PromptTemplate.from_template(rag_template)
        
        prompt_parts = [
            {"text": rag_prompt_template.format(context=context_text, question=query, language=language)}
        ]

        print(f"DEBUG: Sending prompt to Gemini: {prompt_parts[0]['text'][:200]}...") 

        response = llm.generate_content(
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

def detect_document_request(query: str, template_names: list, language: str = "en") -> str: 
    global llm 

    if llm is None:
        initialize_aleks_components() 

    formatted_template_names = ", ".join(template_names)

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

    User: What are the tax requirements for a new business?\
    Response: NONE

    User: Draft a simple contract.
    Response: NONE

    User: Kailangan ko ng NDA. (I need an NDA.)
    Response: nda

    Query: {query}
    Response:"""

    try:
        response = llm.generate_content(
            [{"text": prompt_text}],
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config={
                "temperature": 0.0, 
                "max_output_tokens": 50 
            }
        )
        
        detected_document_type = response.text.strip().lower()

        if detected_document_type not in template_names and detected_document_type != "none":
            print(f"Warning: Gemini returned an unexpected document type: '{detected_document_type}'. Returning NONE.")
            return "NONE" 
        
        return detected_document_type
    except Exception as e:
        print(f"CRITICAL ERROR IN DOCUMENT DETECTION (Gemini API): {e}")
        traceback.print_exc()
        return "NONE" 
