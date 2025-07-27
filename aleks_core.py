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
OLLAMA_MODEL_NAME = "phi3:mini"

# Global variables for the AI components (will be initialized once)
qa_chain = None
llm = None
retriever = None # Make retriever global for direct testing if needed

def initialize_aleks_components():
    """
    Initializes the RAG chain and LLM, making them globally accessible for API endpoints.
    This function should be called once when the FastAPI application starts.
    """
    global qa_chain, llm, retriever # Add retriever to global
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
        llm = OllamaLLM( 
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
            verbose=True, # For more debugging output from LangChain
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running, and 'langchain-ollama' is installed.")
        raise 

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Custom RAG Prompt Template 
    rag_template = """You are Aleks, an AI legal assistant specializing in Philippine law.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer in English.

Context: {context}
Question: {question}

Helpful Answer:"""
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(rag_template)


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # REMOVED: chain_type_kwargs={"prompt": RAG_PROMPT_CUSTOM} 
    )
    print("Aleks AI components loaded successfully!")

# MODIFIED: get_rag_response no longer accepts language
def get_rag_response(query: str) -> dict:
    """
    Performs RAG query using the initialized qa_chain.
    """
    if qa_chain is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")
    
    # DEBUG: print statements for more visibility and timing
    print(f"DEBUG: Invoking RAG chain with query: '{query}'") 
    
    try: 
        print("DEBUG: Before qa_chain.invoke - attempting RAG process...") 
        
        # --- Start timing for retrieval ---
        retrieval_start_time = datetime.now()
        print(f"DEBUG: Starting document retrieval... (Time: {retrieval_start_time.strftime('%H:%M:%S.%f')})") 
        
        # The qa_chain.invoke implicitly calls the retriever first, then the LLM.
        # We need to explicitly call retriever for accurate timing.
        
        # Step 1: Document Retrieval
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieval_end_time = datetime.now()
        retrieval_duration = (retrieval_end_time - retrieval_start_time).total_seconds()
        print(f"DEBUG: Document retrieval completed. Found {len(retrieved_docs)} documents in {retrieval_duration:.2f} seconds.")
        # If this point is reached quickly, the hang is in the LLM part

        # Step 2: LLM Generation with retrieved context
        print(f"DEBUG: Starting LLM generation with context... (Time: {retrieval_end_time.strftime('%H:%M:%S.%f')})") # This is retrieval_end_time which is start of LLM
        
        # Manually create the input for the LLM based on retrieved docs
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Reconstruct the prompt template with context and question
        llm_prompt = PromptTemplate.from_template(
            """You are Aleks, an AI legal assistant specializing in Philippine law.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer in English.

Context: {context}
Question: {question}

Helpful Answer:"""
        )
        
        llm_chain = LLMChain(prompt=llm_prompt, llm=llm)
        
        llm_generation_start_time = datetime.now() # More accurate start time for LLM
        
        # Pass context and query to the LLM chain
        response_from_llm_chain = llm_chain.invoke({"context": context_text, "question": query})
        
        llm_generation_end_time = datetime.now()
        llm_generation_duration = (llm_generation_end_time - llm_generation_start_time).total_seconds()
        print(f"DEBUG: LLM generation completed in {llm_generation_duration:.2f} seconds.")
        
        # Combine the results as RetrievalQA would
        final_result = response_from_llm_chain['text'] # Assuming LLMChain returns text in 'text' key
        
        # Format source documents nicely for API response
        sources_info = []
        if retrieved_docs: # Use retrieved_docs directly here
            for i, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get('source', 'Unknown Document')
                start_index = doc.metadata.get('start_index', 'N/A')
                sources_info.append({
                    "source": source_name,
                    "startIndex": start_index,
                    "snippet": doc.page_content[:200] + "..." 
                })

        return {
            "answer": final_result,
            "sources": sources_info
        }

    except Exception as e:
        print(f"CRITICAL ERROR IN RAG CHAIN INVOCATION: {e}")
        traceback.print_exc() 
        raise 

def detect_document_request(query: str) -> str:
    """
    Uses an LLM to determine if the query is a request for a document template
    and identifies which document type.
    """
    if llm is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")

    template_names = ", ".join(DOCUMENT_TEMPLATES.keys())
    
    # Document detection prompt
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
    
    # Temporarily adjust temperature for classification task
    original_temperature = llm.temperature
    llm.temperature = 0.3
    try: 
        # Pass only 'query' and 'template_names' to invoke
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
