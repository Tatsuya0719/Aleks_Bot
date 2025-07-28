# vector_db_creator.py
import os
import shutil # For removing directory
import json # For parsing LLM's JSON output
from datetime import datetime # For timing
import traceback # For detailed error info

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate 
from langchain.chains import LLMChain 

# Assuming these constants are defined in document_manager.py
# If not, you might need to define them here or ensure document_manager.py is imported correctly
# from document_manager import LEGAL_DATA_DIR, CHROMA_DB_DIR 
LEGAL_DATA_DIR = "./legal_data_pdfs"
CHROMA_DB_DIR = "./chroma_db"

# --- Configuration ---
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 10 # Number of chunks to process in one LLM call for tagging

# --- Ollama Configuration for Tagging ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral" # Changed from "phi3:mini" to "mistral"

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

def generate_tags_for_batch(llm_tagger, chunks: list) -> list:
    """
    Uses LLM to generate relevant tags for a batch of chunks.
    Returns a list of lists, where each inner list contains tags for a chunk.
    """
    if llm_tagger is None or not chunks:
        return [[] for _ in chunks] # Return empty tags for all if LLM not ready or no chunks

    tag_list_str = ", ".join(ALL_POSSIBLE_TAGS)
    
    # Construct the batch prompt
    batch_prompt_content = ""
    for i, chunk in enumerate(chunks):
        batch_prompt_content += f"Chunk {i+1} (ID: chunk_{i}):\n---\n{chunk.page_content}\n---\n\n"

    # MODIFIED PROMPT: Even stronger emphasis on JSON, and a clear instruction to only output JSON
    tagging_prompt_template = PromptTemplate(
        input_variables=["batch_content", "tag_list"],
        template=f"""You are a highly precise AI assistant. Your task is to identify relevant legal topics for each provided text chunk from the given list of tags.
        You MUST return your response as a valid JSON array. Each object in the array MUST have a 'chunk_id' (e.g., 'chunk_0', 'chunk_1') and a 'tags' array.
        If no tags are relevant for a chunk, its 'tags' array MUST be empty.
        DO NOT include any introductory text, conversational phrases, explanations, or anything outside the JSON array.
        Your entire response MUST be a single, valid JSON array.

        Available Tags: {tag_list_str}

        Text Chunks for Tagging:
        {batch_prompt_content}

        JSON Response:"""
    )
    llm_chain = LLMChain(prompt=tagging_prompt_template, llm=llm_tagger)

    try:
        response = llm_chain.invoke({"batch_content": batch_prompt_content, "tag_list": tag_list_str})
        raw_llm_output = response['text'].strip()
        
        # AGGRESSIVE CLEANUP: Try to find the first '[' and last ']' to extract potential JSON
        json_start = raw_llm_output.find('[')
        json_end = raw_llm_output.rfind(']')

        cleaned_json_output = ""
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned_json_output = raw_llm_output[json_start : json_end + 1]
        else:
            print(f"Warning: Could not find valid JSON boundaries in LLM output. Raw output: {raw_llm_output[:500]}...")
            return [[] for _ in chunks] # Return empty tags if no JSON boundaries found


        # Attempt to parse the JSON output
        parsed_results = json.loads(cleaned_json_output)
        
        # Create a dictionary for easy lookup by chunk_id
        tags_map = {item['chunk_id']: item.get('tags', []) for item in parsed_results if 'chunk_id' in item}
        
        # Map parsed tags back to the original order of chunks
        batch_tags = []
        for i in range(len(chunks)):
            chunk_id = f"chunk_{i}"
            # Filter to ensure only valid, predefined tags are returned
            current_chunk_tags = [
                tag.strip() for tag in tags_map.get(chunk_id, []) 
                if isinstance(tag, str) and tag.strip() in ALL_POSSIBLE_TAGS
            ]
            batch_tags.append(list(set(current_chunk_tags))) # Ensure unique tags per chunk
        
        return batch_tags

    except json.JSONDecodeError as e:
        print(f"Warning: LLM returned malformed JSON for tagging batch even after cleanup. Error: {e}")
        print(f"Cleaned LLM output: {cleaned_json_output[:500]}...") # Print cleaned problematic output
        return [[] for _ in chunks] # Return empty tags for all chunks in this batch
    except Exception as e:
        print(f"Warning: Error generating tags for batch: {e}")
        traceback.print_exc()
        return [[] for _ in chunks] # Return empty tags for all chunks in this batch


def create_vector_db():
    """
    Loads PDF documents, splits them into chunks, assigns LLM-generated tags in batches,
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
        if filename.lower().endswith(".pdf"):
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

    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    all_chunks = []
    for doc in documents:
        all_chunks.extend(text_splitter.split_documents([doc]))

    if not all_chunks:
        print("No text chunks created. Check document content or chunking parameters.")
        return

    print(f"Created {len(all_chunks)} text chunks. Starting LLM-based tagging in batches...")

    chunks_with_tags = []
    total_chunks = len(all_chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} chunks)...")
        batch_start_time = datetime.now()
        
        # Generate tags for the entire batch
        batch_tags_list = generate_tags_for_batch(llm_tagger, batch)
        
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        print(f"  Batch processed in {batch_duration:.2f} seconds.")

        # Assign tags back to individual chunks
        for j, chunk in enumerate(batch):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            # Ensure source is carried over, as it might be lost during splitting
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = all_chunks[i+j].metadata.get('source', 'Unknown Document')
            
            # Assign the tags for this specific chunk from the batch_tags_list
            chunk.metadata['tags'] = batch_tags_list[j] if j < len(batch_tags_list) else []
            chunks_with_tags.append(chunk)
            print(f"  Chunk {i+j}: Tags: {chunk.metadata['tags']}, Source: {chunk.metadata['source']}")

    print(f"Finished tagging {len(chunks_with_tags)} text chunks.")

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
            chunks_with_tags, # Use the chunks with LLM-generated tags
            embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("ChromaDB created and populated successfully with LLM-generated tags!")
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")
        print("Please ensure 'langchain-chroma' is installed.")


if __name__ == "__main__":
    # Create a dummy directory and file for demonstration if they don't exist
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
