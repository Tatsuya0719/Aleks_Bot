# aleks_core.py
import os
import re
from datetime import datetime
import traceback 

# Core LangChain components for RAG
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
OLLAMA_MODEL_NAME = "phi3:mini" # Using phi3:mini for speed and instruction following

# Global variables for the AI components (will be initialized once)
qa_chain = None
llm = None

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
    global qa_chain, llm
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

    # Create the base retriever here. Its filters will be updated dynamically in get_rag_response.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Custom RAG Prompt Template for Language Instruction
    # Corrected: Use double curly braces for LangChain's placeholders
    rag_template = """You are Aleks, an AI legal assistant specializing in Philippine law.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer in the language specified by the user's language preference.
If the user's language preference is 'fil', respond in Filipino.
If the user's language preference is 'en', respond in English.

Context: {{context}}
Question: {{question}}
User's Language Preference: {{language}}

Helpful Answer:"""
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(rag_template)

    # Initialize RetrievalQA chain with the base retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=base_retriever, # Use the base retriever here
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_CUSTOM} # Apply custom prompt
    )
    print("Aleks AI components loaded successfully!")

def detect_tags_from_query(query: str, language: str = "en") -> list:
    """
    Uses the LLM to detect relevant tags from the user's query.
    """
    if llm is None:
        raise RuntimeError("LLM not initialized for tag detection.")

    tag_list_str = ", ".join(ALL_POSSIBLE_TAGS)
    # Corrected: Use double curly braces for LangChain's placeholders
    tag_detection_prompt = PromptTemplate(
        input_variables=["query", "tag_list", "language"],
        template=f"""Analyze the user's query and identify ALL relevant legal tags from the following list: {tag_list_str}.
Return ONLY the tags that are directly relevant to the query, separated by commas. If no tags are relevant, return "NONE".
Do not include any other text or explanation.
Respond in English if you need to clarify, otherwise just the comma-separated tags or NONE.

Examples:
User: I need an NDA.
Response: nda

User: Can you help me draft a non-disclosure agreement?
Response: non-disclosure agreement

Query: {{query}}
User's Language: {{language}}
Response:"""
    )

    llm_chain_tags = LLMChain(prompt=tag_detection_prompt, llm=llm)
    
    # Temporarily adjust temperature for classification task
    original_temperature = llm.temperature
    llm.temperature = 0.3
    try: 
        response = llm_chain_tags.invoke({"query": query, "tag_list": tag_list_str, "language": language})
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

def get_rag_response(query: str, language: str = "en") -> dict:
    """
    Performs RAG query using the initialized qa_chain, with tag-based filtering and language instruction.
    """
    global qa_chain, llm 
    if llm is None or qa_chain is None: 
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")
    
    print(f"DEBUG: Invoking RAG process with query: '{query}' and language: '{language}'") 
    
    try: 
        # --- Step 1.0: Detect Tags from Query ---
        tag_detection_start_time = datetime.now()
        print(f"DEBUG: Starting query tag detection... (Time: {tag_detection_start_time.strftime('%H:%M:%S.%f')})")
        detected_tags = detect_tags_from_query(query, language) 
        tag_detection_end_time = datetime.now()
        tag_detection_duration = (tag_detection_end_time - tag_detection_start_time).total_seconds()
        print(f"DEBUG: Query tag detection completed. Detected tags: {detected_tags} in {tag_detection_duration:.2f} seconds.")

        # --- Step 1.1: Dynamically update retriever's filter within the qa_chain ---
        # Access the underlying retriever and update its search_kwargs
        if detected_tags:
            where_clause = {"tags": {"$contains_any": detected_tags}}
            qa_chain.retriever.search_kwargs["where"] = where_clause
            print(f"DEBUG: Applying ChromaDB filter to retriever: {where_clause}")
        else:
            # If no tags, ensure no filter is applied (or reset it if previously set)
            if "where" in qa_chain.retriever.search_kwargs:
                del qa_chain.retriever.search_kwargs["where"]
            print("DEBUG: No specific tags detected from query, performing general retrieval.")
        
        retrieval_start_time = datetime.now()
        print(f"DEBUG: Starting RAG chain invocation... (Time: {retrieval_start_time.strftime('%H:%M:%S.%f')})") 

        # Invoke the qa_chain with the query and language
        response = qa_chain.invoke({"query": query, "language": language})
        
        retrieval_end_time = datetime.now()
        full_rag_duration = (retrieval_end_time - retrieval_start_time).total_seconds()
        print(f"DEBUG: Full RAG chain invocation completed in {full_rag_duration:.2f} seconds.")
        print(f"DEBUG: RAG chain returned raw response: {response}")

        # Check if response contains 'result' and 'source_documents'
        if not response.get("result") and not response.get("source_documents") and detected_tags:
            print("WARNING: No results found with specific tags. Retrying RAG without tag filter.")
            # Fallback: Remove the filter and try again
            if "where" in qa_chain.retriever.search_kwargs:
                del qa_chain.retriever.search_kwargs["where"]
            response = qa_chain.invoke({"query": query, "language": language})
            print(f"DEBUG: Fallback RAG chain returned raw response: {response}")

        # If still no documents or answer, return a specific message
        if not response.get("result") and not response.get("source_documents"):
            return {
                "answer": "Sorry, I couldn't find relevant information in the documents to answer that. Please try rephrasing your question.",
                "sources": []
            }
        
        final_answer = response.get("result", "Sorry, I couldn't generate an answer based on the retrieved information.")
        
        # Format source documents nicely for API response
        sources_info = []
        if response.get("source_documents"):
            for i, doc in enumerate(response["source_documents"]):
                source_name = doc.metadata.get('source', 'Unknown Document')
                start_index = doc.metadata.get('start_index', 'N/A')
                sources_info.append({
                    "source": source_name,
                    "startIndex": start_index,
                    "snippet": doc.page_content[:200] + "..." 
                })

        return {
            "answer": final_answer,
            "sources": sources_info
        }

    except Exception as e:
        print(f"CRITICAL ERROR IN RAG CHAIN INVOCATION: {e}")
        traceback.print_exc() 
        raise 

def detect_document_request(query: str, language: str = "en") -> str:
    """
    Uses an LLM to determine if the query is a request for a document template
    and identifies which document type, considering the user's language.
    """
    if llm is None:
        raise RuntimeError("Aleks components not initialized. Call initialize_aleks_components first.")

    template_names = ", ".join(DOCUMENT_TEMPLATES.keys())
    
    # Corrected: Use double curly braces for LangChain's placeholders
    prompt_template = PromptTemplate(
        input_variables=["query", "template_names", "language"],
        template=f"""You are an AI assistant. Analyze the user's query to determine if they are asking for a legal document template.
If they are, identify which specific document they are asking for from the following types: {template_names}.
If you identify a document, respond ONLY with the document type (e.g., "nda", "non-disclosure agreement").
If the query is NOT a document request, respond ONLY with "NONE".
Respond in the language specified by 'language' if you need to clarify, otherwise just the document type or NONE.

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

Query: {{query}}
User's Language: {{language}}
Response:"""
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    # Temporarily adjust temperature for classification task
    original_temperature = llm.temperature
    llm.temperature = 0.3
    try: 
        response = llm_chain.invoke({"query": query, "template_names": template_names, "language": language}) 
    except Exception as e:
        print(f"CRITICAL ERROR IN LLMCHAIN INVOCATION (Document Detection): {e}")
        traceback.print_exc() 
        raise 
    llm.temperature = original_temperature

    detected_type = response['text'].strip().lower()

    if detected_type in DOCUMENT_TEMPLATES:
        return detected_type
    return "NONE"
