# vector_db_creator.py
import os
import shutil # For removing directory
import re # For parsing filenames
from datetime import datetime # For timing

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document # Import Document class from langchain_core

# Assuming these constants are defined in document_manager.py
# If not, you might need to define them here or ensure document_manager.py is imported correctly
LEGAL_DATA_DIR = "./legal_data_pdfs"
CHROMA_DB_DIR = "./chroma_db"

# --- Configuration ---
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Define a comprehensive list of potential tags.
# This list MUST be consistent with the tags you use in your filenames.
# This list will also be used in aleks_core.py for query-time tag detection.
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

def extract_tags_from_filename(filename: str) -> list:
    """
    Extracts tags from a filename based on the convention:
    [OriginalFileName]_[tag1]_[tag2]..._tagN.pdf
    """
    # Remove .pdf extension
    base_name = os.path.splitext(filename)[0]
    
    # Split by underscore
    parts = base_name.split('_')
    
    # Tags are expected at the end, after the main filename parts.
    # We'll assume any part that matches one of our ALL_POSSIBLE_TAGS is a tag.
    extracted_tags = []
    for part in parts:
        if part.lower() in ALL_POSSIBLE_TAGS:
            extracted_tags.append(part.lower())
            
    # Return unique tags
    return list(set(extracted_tags))


def create_vector_db():
    """
    Loads PDF documents, extracts text, assigns file-name-based tags,
    splits into chunks, generates embeddings, and stores them in ChromaDB.
    """
    if not os.path.exists(LEGAL_DATA_DIR):
        print(f"Error: Legal data directory '{LEGAL_DATA_DIR}' not found. Please ensure your PDF documents are in this folder.")
        return

    print(f"Loading documents from {LEGAL_DATA_DIR} and extracting tags from filenames...")
    documents = []
    for filename in os.listdir(LEGAL_DATA_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(LEGAL_DATA_DIR, filename)
            
            # Extract tags from the filename
            file_tags = extract_tags_from_filename(filename)
            if not file_tags:
                print(f"Warning: No valid tags found in filename '{filename}'. This document's chunks will not have tags for filtering.")

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
                    doc.metadata['tags'] = file_tags # Assign the extracted tags
                
                documents.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} pages from {filename}. Assigned tags: {file_tags}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not documents:
        print("No documents loaded. Please check your 'legal_data_pdfs' directory and PDF files.")
        return

    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    all_chunks = []
    for doc in documents:
        # Split each document into its own chunks, carrying over metadata (including tags)
        doc_chunks = text_splitter.split_documents([doc])
        for chunk in doc_chunks:
            # Ensure chunk metadata includes source and tags from the parent document
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata['source'] = doc.metadata.get('source', 'Unknown Document')
            chunk.metadata['tags'] = doc.metadata.get('tags', []) # Ensure tags are carried to chunks
            all_chunks.append(chunk)

    if not all_chunks:
        print("No text chunks created. Check document content or chunking parameters.")
        return

    print(f"Created {len(all_chunks)} text chunks with file-name-based tags.")

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
            print(f"Removing existing ChromaDB at {CHROMA_DB_DIR}...")
            shutil.rmtree(CHROMA_DB_DIR)
            print("Existing ChromaDB removed.")

        # Add documents with their metadata (including tags) to ChromaDB
        Chroma.from_documents(
            all_chunks, # Use the chunks with file-name-based tags
            embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("ChromaDB created and populated successfully with file-name-based tags!")
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        print("Please ensure 'langchain-chroma' is installed.")


if __name__ == "__main__":
    # Create a dummy directory and file for demonstration if they don't exist
    dummy_dir = "./legal_data_pdfs"
    # IMPORTANT: Renamed dummy file to include tags for testing the new logic
    dummy_file = os.path.join(dummy_dir, "dummy_law_labor_law_employment.pdf") 
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
    if not os.path.exists(dummy_file):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(dummy_file, pagesize=letter)
        c.drawString(100, 750, "This is a dummy legal document about labor laws.")
        c.drawString(100, 730, "It talks about some important employment concepts.")
        c.drawString(100, 710, "For example, Article 1, Section 1 states that all citizens are equal in employment.")
        c.drawString(100, 690, "And Section 2 talks about property rights, which is irrelevant here.") 
        c.save()
        print(f"Created a dummy PDF at {dummy_file} for testing with tags.")
    else:
        print(f"Using existing dummy PDF at {dummy_file}.")

    print("Starting vector database creation with file-name-based tags...")
    create_vector_db()
