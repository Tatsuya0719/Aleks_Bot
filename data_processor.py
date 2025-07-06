# data_processor.py
import pypdf
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document # Import Document class

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                # Ensure page can be extracted as text
                if page.extract_text():
                    text += page.extract_text()
                else:
                    print(f"Warning: Could not extract text from page {page_num+1} of {os.path.basename(pdf_path)}")
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def load_and_process_legal_data(pdf_directory="./legal_data_pdfs"):
    """
    Loads PDF documents, extracts text, cleans it, and splits into chunks.
    Returns a list of LangChain Document objects.
    """
    all_raw_text = []
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Basic cleaning: remove extra newlines and leading/trailing whitespace
                cleaned_text = os.linesep.join([s for s in text.splitlines() if s.strip()])
                cleaned_text = cleaned_text.strip()
                all_raw_text.append(Document(page_content=cleaned_text, metadata={"source": filename}))
            else:
                print(f"Skipping {filename} due to no extracted text.")

    if not all_raw_text:
        print("No legal documents found or extracted. Please check your 'legal_data_pdfs' directory.")
        return []

    # Initialize the text splitter
    # RecursiveCharacterTextSplitter tries to split by paragraphs, then sentences, then words
    # You'll need to experiment with chunk_size and chunk_overlap
    # A chunk_size of 1000-2000 characters is a good starting point for legal docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Max characters per chunk
        chunk_overlap=200, # Characters to overlap between chunks (helps maintain context)
        length_function=len, # Use standard Python len() for character count
        add_start_index=True # Adds starting index of chunk to metadata
    )

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(all_raw_text)

    print(f"Created {len(chunks)} text chunks from your legal documents.")
    return chunks

if __name__ == "__main__":
    # Create a dummy directory and file for demonstration if they don't exist
    dummy_dir = "./legal_data_pdfs"
    dummy_file = os.path.join(dummy_dir, "dummy_law.pdf")
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
    if not os.path.exists(dummy_file):
        # Create a very simple dummy PDF for testing
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
        print(f"Using existing dummy PDF at {dummy_file} for testing.")


    legal_chunks = load_and_process_legal_data()
    if legal_chunks:
        print("\nFirst chunk content:")
        print(legal_chunks[0].page_content[:500] + "...")
        print("Metadata:", legal_chunks[0].metadata)