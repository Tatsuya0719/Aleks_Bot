# aleks_core.py
import os
import re
from datetime import datetime
import traceback 

# Core LangChain components for RAG - make sure these are the updated ones
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM 

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain 

# Import constants from document_manager
from document_manager import DOCUMENT_TEMPLATES, PLACEHOLDER_DESCRIPTIONS, TEMPLATE_DIR 

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "phi3:mini" # Keep this smaller model for better performance

# Global variables for the AI components (will be initialized once)
qa_chain = None # This will be less directly used for streaming, but kept for structure
llm = None
retriever = None 

def initialize_aleks_components():
    """
    Initializes the RAG chain and LLM, making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global qa_chain, llm, retriever 
    print("Initializing Aleks AI components...")
    
    print("Loading embedding model for retrieval...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure 'langchain-huggingface' and 'torch' are installed.")
        raise 

    print(f"Loading vector database from {CHROMA_DB_DIR}...")
    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"Error: Chroma DB directory '{CHROMA_DB_DIR}' not found. Please ensure you have run the data ingestion script.")
            raise 
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        print("Vector database loaded.")
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Please ensure 'langchain-chroma' is installed and your database exists.")
        raise 

    print("Initializing LLM...")
    try:
        # NEW: Set `streaming=True` for OllamaLLM
        llm = OllamaLLM( 
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
            verbose=True, 
            streaming=True # Enable streaming for the LLM
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME} with streaming enabled.")
    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running, and 'langchain-ollama' is installed.")
        raise 

    # Keep k=2 for now, can adjust later if needed
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

    # Custom RAG Prompt Template 
    rag_template = """You are Aleks, an AI legal assistant specializing in Philippine law.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer in English.

Context: {context}
Question: {question}

Helpful Answer:"""
    global RAG_PROMPT_CUSTOM # Make it global for direct use
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(rag_template)

    # qa_chain is no longer directly used for streaming, but kept for structure
    # If we were not streaming, this is how it would be set up:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    print("Aleks AI components loaded successfully!")

# NEW: Asynchronous generator for streaming responses
async def get_rag_response_stream(query: str):
    """
    Performs RAG query and yields response tokens as they are generated.
    """
    global retriever, llm, RAG_PROMPT_CUSTOM # Ensure access to globals
    if retriever is None or llm is None or RAG_PROMPT_CUSTOM is None:
        yield "Error: Aleks components not initialized for streaming. Please restart the backend."
        return

    print(f"DEBUG: Invoking RAG chain STREAMING with query: '{query}'") 
    
    try: 
        print("DEBUG: Starting document retrieval for streaming...") 
        retrieval_start_time = datetime.now()
        
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieval_end_time = datetime.now()
        retrieval_duration = (retrieval_end_time - retrieval_start_time).total_seconds()
        print(f"DEBUG: Document retrieval completed. Found {len(retrieved_docs)} documents in {retrieval_duration:.2f} seconds.")

        if not retrieved_docs:
            yield "Sorry, I couldn't find relevant information in the documents to answer that."
            return

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        print(f"DEBUG: Starting LLM generation STREAMING with context... (Time: {retrieval_end_time.strftime('%H:%M:%S.%f')})") 
        
        # Create the full prompt for the LLM
        full_prompt_input = RAG_PROMPT_CUSTOM.format(context=context_text, question=query)

        # Iterate over the streamed response from OllamaLLM
        llm_generation_start_time = datetime.now()
        for chunk in llm.stream(full_prompt_input): # Use llm.stream() directly
            # Each chunk is a dictionary like {'text': 'word'}
            if 'text' in chunk:
                yield chunk['text'] # Yield the text part of the chunk
        
        llm_generation_end_time = datetime.now()
        llm_generation_duration = (llm_generation_end_time - llm_generation_start_time).total_seconds()
        print(f"DEBUG: LLM generation STREAMING completed in {llm_generation_duration:.2f} seconds.")

        # Optionally yield source information at the end of the stream
        sources_info = []
        if retrieved_docs: 
            for i, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get('source', 'Unknown Document')
                # For streaming, you might want to send sources as a separate final message or not at all
                # For simplicity, we'll just print them to the backend log for now
                # In a real app, you'd structure this as a final JSON message or SSE event
                sources_info.append(f"Source: {source_name}, Snippet: {doc.page_content[:100]}...")
        if sources_info:
            print("DEBUG: Generated Sources:", sources_info) # Print to backend log

    except Exception as e:
        print(f"CRITICAL ERROR IN RAG CHAIN STREAMING INVOCATION: {e}")
        traceback.print_exc() 
        yield f"Error: An error occurred during processing: {e}" # Send error message to client
        return # Ensure the generator stops

# detect_document_request remains unchanged from previous version
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
Always respond in English if you need to clarify, otherwise just the document type or NONE.

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
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    original_temperature = llm.temperature
    llm.temperature = 0.3
    try: 
        response = llm_chain.invoke({"query": query, "template_names": template_names}) 
    except Exception as e:
        print(f"CRITICAL ERROR IN LLMCHAIN INVOCATION (Document Detection): {e}")
        traceback.print_exc() 
        raise 
    llm.temperature = original_temperature

    detected_type = response['text'].strip().lower()

    if detected_type in DOCUMENT_TEMPLATES:
        return detected_type
    return "NONE"
