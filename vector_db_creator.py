# vector_db_creator.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM # NEW: Import OllamaLLM for tagging
from langchain_core.prompts import PromptTemplate # NEW: For LLM tagging prompt
from langchain.chains import LLMChain # NEW: For LLM tagging chain

# Assuming these constants are defined in document_manager.py
# If not, you might need to define them here or ensure document_manager.py is imported correctly
# from document_manager import LEGAL_DATA_DIR, CHROMA_DB_DIR 
LEGAL_DATA_DIR = "./legal_data_pdfs"
CHROMA_DB_DIR = "./chroma_db"

# --- Configuration ---
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Ollama Configuration for Tagging ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "phi3:mini" # Use phi3:mini for faster tagging

# Define a comprehensive list of potential tags based on your legal documents
# This list will be provided to the LLM to guide its tag generation.
# You can expand this list based on the types of legal documents you have.
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


def create_llm_for_tagging():
    """Initializes a dedicated LLM for generating tags."""
    try:
        llm_tagger = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.0, # Keep temperature low for consistent tagging
            verbose=False,
        )
        return llm_tagger
    except Exception as e:
        print(f"Error initializing LLM for tagging: {e}")
        print("Ensure Ollama is running and model is pulled.")
        return None

def generate_tags_for_chunk(llm_tagger, chunk_content: str) -> list:
    """Uses LLM to generate relevant tags for a given chunk of text."""
    if llm_tagger is None:
        return []

    tag_list_str = ", ".join(ALL_POSSIBLE_TAGS)
    tagging_prompt_template = PromptTemplate(
        input_variables=["chunk_content", "tag_list"],
        template=f"""Analyze the following legal text and identify the most relevant legal topics or categories from this list: {tag_list_str}.
        Return ONLY the relevant tags, separated by commas. If no tags are relevant, return "NONE".
        Do not include any other text or explanation.

        Legal Text:
        ---
        {{chunk_content}}
        ---

        Relevant Tags:"""
    )
    llm_chain = LLMChain(prompt=tagging_prompt_template, llm=llm_tagger)

    try:
        response = llm_chain.invoke({"chunk_content": chunk_content, "tag_list": tag_list_str})
        raw_tags = response['text'].strip().lower()
        if raw_tags == "none" or not raw_tags:
            return []
        
        # Filter to ensure only valid, predefined tags are returned
        generated_tags = [tag.strip() for tag in raw_tags.split(',') if tag.strip() in ALL_POSSIBLE_TAGS]
        return list(set(generated_tags)) # Return unique tags
    except Exception as e:
        print(f"Warning: Error generating tags for chunk: {e}")
        traceback.print_exc()
        return []


def create_vector_db():
    """
    Loads PDF documents, splits them into chunks, assigns LLM-generated tags,
    generates embeddings, and stores them in ChromaDB.
    """
    if not os.path.exists(LEGAL_DATA_DIR):
        print(f"Error: Legal data directory '{LEGAL_DATA_DIR}' not found. Please ensure your PDF documents are in this folder.")
        return

    llm_tagger = create_llm_for_tagging()
    if llm_tagger is None:
        print("Could not initialize LLM for tagging. Aborting DB creation.")
        return

    print(f"Loading documents from {LEGAL_DATA_DIR}...")
    documents = []
    for filename in os.listdir(LEGAL_DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(LEGAL_DATA_DIR, filename)
            try:
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()
                
                for doc in loaded_docs:
                    # Basic cleaning: remove extra newlines and leading/trailing whitespace
                    cleaned_text = os.linesep.join([s for s in doc.page_content.splitlines() if s.strip()])
                    cleaned_text = cleaned_text.strip()
                    doc.page_content = cleaned_text # Update page_content with cleaned text

                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source'] = filename # Keep original source
                
                documents.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} pages from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not documents:
        print("No documents loaded. Please check your 'legal_data_pdfs' directory and PDF files.")
        return

    print(f"Splitting {len(documents)} pages into chunks and generating tags...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    chunks_with_tags = []
    for i, doc in enumerate(documents):
        # Split each document into its own chunks
        doc_chunks = text_splitter.split_documents([doc])
        for chunk in doc_chunks:
            # Generate tags for each chunk using the LLM
            tags = generate_tags_for_chunk(llm_tagger, chunk.page_content)
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata['source'] = doc.metadata.get('source', 'Unknown Document') # Ensure source is carried over
            chunk.metadata['tags'] = tags # Add the LLM-generated tags
            chunks_with_tags.append(chunk)
            print(f"  Processed chunk {len(chunks_with_tags)}: Tags: {tags}, Source: {chunk.metadata['source']}")

    if not chunks_with_tags:
        print("No text chunks created or tagged. Check document content or chunking/tagging parameters.")
        return

    print(f"Created {len(chunks_with_tags)} text chunks with LLM-generated tags.")

    print("Initializing embeddings model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("Embeddings model loaded.")
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        print("Please ensure 'langchain-huggingface' and 'torch' are installed.")
        return

    print(f"Creating ChromaDB at {CHROMA_DB_DIR}...")
    try:
        # Delete existing ChromaDB to ensure fresh build with new metadata
        if os.path.exists(CHROMA_DB_DIR):
            import shutil
            shutil.rmtree(CHROMA_DB_DIR)
            print(f"Removed existing ChromaDB at {CHROMA_DB_DIR}")

        # Add documents with their metadata (including tags) to ChromaDB
        Chroma.from_documents(
            chunks_with_tags, # Use the chunks with LLM-generated tags
            embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("ChromaDB created and populated successfully with LLM-generated tags!")
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        print("Please ensure 'langchain-chroma' is installed.")


if __name__ == "__main__":
    # Ensure the legal_data_pdfs directory and dummy file exist for testing
    dummy_dir = "./legal_data_pdfs"
    dummy_file = os.path.join(dummy_dir, "dummy_law.pdf")
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
    if not os.path.exists(dummy_file):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(dummy_file, pagesize=letter)
        c.drawString(100, 750, "This is a dummy legal document.")
        c.drawString(100, 730, "It talks about some important legal concepts.")
        c.drawString(100, 710, "For example, Article 1, Section 1 states that all citizens are equal.")
        c.drawString(100, 690, "And Section 2 talks about property rights.")
        c.save()
        print(f"Created a dummy PDF at {dummy_file} for testing.")
    else:
        print(f"Using existing dummy PDF at {dummy_file}.")

    print("Starting vector database creation with LLM-generated tags...")
    create_vector_db()

