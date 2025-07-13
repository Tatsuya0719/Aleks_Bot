# aleks_core.py
import os
import re
from datetime import datetime

# Core LangChain components for RAG - make sure these are the updated ones
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM # Note: The class name often changes to OllamaLLM

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain # Still used, but deprecated

# Import constants from document_manager
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR # Also need placeholder descriptions and TEMPLATE_DIR now

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral"

# Global variables for the AI components (will be initialized once)
qa_chain = None
llm = None

def initialize_aleks_components():
    """
    Initializes the RAG chain and LLM, making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global qa_chain, llm
    print("Initializing Aleks AI components...")
    
    print("Loading embedding model for retrieval...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure 'langchain-huggingface' and 'torch' are installed.")
        raise # Re-raise to stop app if critical component fails

    print(f"Loading vector database from {CHROMA_DB_DIR}...")
    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"Error: Chroma DB directory '{CHROMA_DB_DIR}' not found. Please ensure you have run the data ingestion script.")
            raise # Re-raise
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        print("Vector database loaded.")
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Please ensure 'langchain-chroma' is installed and your database exists.")
        raise # Re-raise

    print("Initializing LLM...")
    try:
        llm = OllamaLLM( # Use OllamaLLM
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running, and 'langchain-ollama' is installed.")
        raise # Re-raise

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("Aleks AI components loaded successfully!")

def get_rag_response(query: str) -> dict:
    """
    Performs RAG query using the initialized qa_chain.
    """
    if qa_chain is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")
    
    response = qa_chain.invoke({"query": query})
    
    # Format source documents nicely for API response
    sources_info = []
    if response.get("source_documents"):
        for i, doc in enumerate(response["source_documents"]):
            source_name = doc.metadata.get('source', 'Unknown Document')
            start_index = doc.metadata.get('start_index', 'N/A')
            sources_info.append({
                "source": source_name,
                "startIndex": start_index,
                "snippet": doc.page_content[:200] + "..." # Limit snippet length
            })

    return {
        "answer": response["result"],
        "sources": sources_info
    }

def detect_document_request(query: str) -> str:
    """
    Uses an LLM to determine if the query is a request for a document template
    and identifies which document type.
    """
    if llm is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")

    template_names = ", ".join(DOCUMENT_TEMPLATES.keys())
    
    prompt_template = PromptTemplate(
        input_variables=["query", "template_names"],
        template="""You are an AI assistant. Analyze the user's query to determine if they are asking for a legal document template.
        If they are, identify which specific document they are asking for from the following types: {template_names}.
        If you identify a document, respond ONLY with the document type (e.g., "nda", "non-disclosure agreement").
        If the query is NOT a document request, respond ONLY with "NONE".

        Examples:
        User: I need an NDA.
        Response: nda

        User: Can you help me draft a non-disclosure agreement?
        Response: non-disclosure agreement

        User: What are the tax requirements for a new business?
        Response: NONE

        User: Draft a simple contract.
        Response: NONE

        Query: {query}
        Response:"""
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    # Temporarily adjust temperature for classification task
    original_temperature = llm.temperature
    llm.temperature = 0.3
    response = llm_chain.invoke({"query": query, "template_names": template_names})
    llm.temperature = original_temperature

    detected_type = response['text'].strip().lower()

    if detected_type in DOCUMENT_TEMPLATES:
        return detected_type
    return "NONE"

# Removed the interactive handle_document_filling as its logic will be split across API calls in aleks_api.py
# document_manager.py will still contain TEMPLATE_DIR, DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS