# vector_db_creator.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from data_processor import load_and_process_legal_data # Import your data processing function
import os

def create_and_persist_vector_db(chunks, db_directory="./chroma_db"):
    """
    Creates a ChromaDB vector store from document chunks and persists it to disk.
    """
    print("Loading embedding model...")
    # Choose a multilingual embedding model. These are downloaded on first run.
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" is good for multiple languages.
    # "sentence-transformers/all-MiniLM-L6-v2" is smaller and faster, but less multilingual.
    embeddings_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    print(f"Creating ChromaDB at {db_directory}...")
    # If the directory exists and contains a DB, it will load it.
    # Otherwise, it will create a new one.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )

    # Explicitly persist the database. This saves it to disk.
    vectorstore.persist()
    print(f"Vector database created/updated and persisted successfully at {db_directory}!")
    return vectorstore

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

    legal_chunks = load_and_process_legal_data()
    if legal_chunks:
        # Create and persist the DB
        db = create_and_persist_vector_db(legal_chunks)
        # You can now test retrieval
        print("\nTesting retrieval from the database:")
        query = "What are the rights of citizens?"
        results = db.similarity_search(query, k=2) # Get top 2 most similar documents
        for i, doc in enumerate(results):
            print(f"--- Retrieved Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(doc.page_content[:300] + "...") # Print first 300 chars