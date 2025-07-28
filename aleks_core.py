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
qa_chain = None
llm = None
retriever = None 

# Define a comprehensive list of potential tags (MUST MATCH vector_db_creator.py)
ALL_POSSIBLE_TAGS = [
    "labor_law", "employment", "wages", "termination", "employee_rights", "employer_obligations",
    "family_law", "marriage", "divorce", "child_custody", "adoption", "support",
    "contract_law", "agreements", "breach", "enforcement", "nda", "lease", "service_agreement",
    "tax_law", "income_tax", "vat", "business_tax", "tax_filing", "tax_compliance",
    "property_law", "real_estate", "ownership", "land_disputes", "rent", "leasehold",
    "criminal_law", "offenses", "penalties", "arrest", "court_procedures", "rights_of_accused",
    "constitutional_law", "human_rights", "government_structure", "citizenship", "elections",
    "civil_law", "torts", "damages", "obligations", "succession", "persons",
    "corporate_law", "business_registration", "corporate_governance", "mergers", "acquisitions",
    "data_privacy_law", "data_protection", "privacy_rights", "data_breach", "consent",
    "holidays_law", "public_holidays", "special_non_working_days", "holiday_pay",
    "business_registration", "permits", "licenses", "dti", "sec", "bir", "sss", "philhealth", "pagibig",
    "consumer_protection", "product_liability", "consumer_rights",
    "intellectual_property", "copyright", "trademark", "patent",
    "court_procedures", "litigation", "evidence", "appeals",
    "immigration_law", "visa", "citizenship_application", "foreigners_rights"
]

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
        llm = OllamaLLM( 
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
            verbose=True, 
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running, and 'langchain-ollama' is installed.")
        raise 

    # Retriever will be configured dynamically in get_rag_response based on tags
    # We will initialize a basic retriever here for the qa_chain, but the actual filtering
    # will happen when we call retriever.get_relevant_documents directly.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # Default k=1

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


    # The qa_chain will now be set up to use the retriever, but the filtering
    # will be handled by manually calling retriever.get_relevant_documents
    # before passing the context to the LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever, # This retriever will be used for the base similarity search
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_CUSTOM} # FIX: Pass the custom prompt here
    )
    print("Aleks AI components loaded successfully!")

# FIX: Removed unused 'language' parameter from function signature
def detect_tags_from_query(query: str) -> list:
    """
    Uses the LLM to detect relevant tags from the user's query.
    """
    if llm is None:
        raise RuntimeError("LLM not initialized for tag detection.")

    tag_list_str = ", ".join(ALL_POSSIBLE_TAGS)
    tag_detection_prompt = PromptTemplate(
        input_variables=["query", "tag_list"],
        template=f"""Analyze the user's query and identify any relevant legal tags from the following list: {tag_list_str}.
        Return ONLY the tags that are directly relevant to the query, separated by commas. If no tags are relevant, return "NONE".
        Do not include any other text or explanation.

        Examples:
        Query: What are the laws about employment in the Philippines?
        Response: labor_law

        Query: I need help with my marriage contract.
        Response: family_law, contract_law

        Query: How to file taxes?
        Response: tax_law

        Query: General question about the legal system.
        Response: NONE

        Query: {query}
        Response:"""
    )

    llm_chain_tags = LLMChain(prompt=tag_detection_prompt, llm=llm)
    
    try:
        response = llm_chain_tags.invoke({"query": query, "tag_list": tag_list_str})
        detected_tags_raw = response['text'].strip().lower()
        
        if detected_tags_raw == "none" or not detected_tags_raw:
            return []
        
        # Parse detected tags, ensuring they are valid
        generated_tags = [tag.strip() for tag in detected_tags_raw.split(',') if tag.strip() in ALL_POSSIBLE_TAGS]
        return list(set(generated_tags)) # Return unique tags
    except Exception as e:
        print(f"CRITICAL ERROR IN TAG DETECTION LLMCHAIN INVOCATION: {e}")
        traceback.print_exc()
        return [] # Return empty list on error

# FIX: Removed unused 'language' parameter from function signature
def get_rag_response(query: str) -> dict:
    """
    Performs RAG query using the initialized qa_chain, with tag-based filtering.
    """
    global retriever # Ensure we can access the global retriever
    if qa_chain is None or retriever is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")
    
    print(f"DEBUG: Invoking RAG chain with query: '{query}'") 
    
    try: 
        print("DEBUG: Before qa_chain.invoke - attempting RAG process...") 
        
        # --- Step 1.0: Detect Tags from Query ---
        tag_detection_start_time = datetime.now()
        print(f"DEBUG: Starting tag detection... (Time: {tag_detection_start_time.strftime('%H:%M:%S.%f')})")
        detected_tags = detect_tags_from_query(query)
        tag_detection_end_time = datetime.now()
        tag_detection_duration = (tag_detection_end_time - tag_detection_start_time).total_seconds()
        print(f"DEBUG: Tag detection completed. Detected tags: {detected_tags} in {tag_detection_duration:.2f} seconds.")

        # --- Step 1.1: Document Retrieval with Metadata Filtering ---
        retrieval_start_time = datetime.now()
        print(f"DEBUG: Starting document retrieval with tags: {detected_tags}... (Time: {retrieval_start_time.strftime('%H:%M:%S.%f')})") 
        
        # Build the metadata filter (ChromaDB 'where' clause)
        where_clause = {}
        if detected_tags:
            # For multiple tags, use '$or' to match any of the detected tags
            # The 'tags' field in metadata is expected to be a list
            where_clause = {"tags": {"$contains_any": detected_tags}}
            print(f"DEBUG: Applying ChromaDB filter: {where_clause}")
        else:
            print("DEBUG: No specific tags detected, performing general retrieval.")
        
        # Use the retriever with the 'where' clause
        retrieved_docs = retriever.get_relevant_documents(query, where=where_clause)
        
        retrieval_end_time = datetime.now()
        retrieval_duration = (retrieval_end_time - retrieval_start_time).total_seconds()
        print(f"DEBUG: Document retrieval completed. Found {len(retrieved_docs)} documents in {retrieval_duration:.2f} seconds.")

        # Ensure we have at least some documents, even if filtered heavily
        if not retrieved_docs and detected_tags:
            print("WARNING: No documents found with specified tags. Retrying without tag filter.")
            retrieved_docs = retriever.get_relevant_documents(query) # Fallback to no filter
            print(f"DEBUG: Fallback retrieval found {len(retrieved_docs)} documents.")

        # If still no documents, return a specific message
        if not retrieved_docs:
            return {
                "answer": "Sorry, I couldn't find relevant information in the documents to answer that. Please try rephrasing your question.",
                "sources": []
            }

        # --- Step 2: LLM Generation with retrieved context ---
        print(f"DEBUG: Starting LLM generation with context... (Time: {retrieval_end_time.strftime('%H:%M:%S.%f')})") 
        
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
        
        llm_generation_start_time = datetime.now() 
        
        # Pass context and query to the LLM chain
        response_from_llm_chain = llm_chain.invoke({"context": context_text, "question": query})
        
        llm_generation_end_time = datetime.now()
        llm_generation_duration = (llm_generation_end_time - llm_generation_start_time).total_seconds()
        print(f"DEBUG: LLM generation completed in {llm_generation_duration:.2f} seconds.")
        
        final_result = response_from_llm_chain['text'] 
        
        # Format source documents nicely for API response
        sources_info = []
        if retrieved_docs: 
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

# FIX: Removed unused 'language' parameter from function signature
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
