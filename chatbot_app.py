import os
# import streamlit as st # Uncomment this if you want to use Streamlit for UI

# Core LangChain components for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Ollama for local LLM inference
from langchain_community.llms import Ollama # <<< THIS MUST BE UNCOMMENTED

# Make sure these are commented out:
# from langchain_community.llms import LlamaCpp
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI


# --- Configuration ---
# Directory where your Chroma vector database is stored
CHROMA_DB_DIR = "./chroma_db"
# Model for generating embeddings (used for semantic search in your vector store)
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API server address
OLLAMA_MODEL_NAME = "mistral" # <<< IMPORTANT: Use the name of the model you pulled with 'ollama pull'

# @st.cache_resource # Uncomment this decorator if you're using Streamlit to cache the RAG chain
def load_rag_chain():
    """
    Loads the embedding model, vector database, and sets up the RAG chain.
    """
    print("Loading embedding model for retrieval...")
    try:
        # Initialize HuggingFace embeddings for text similarity
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure 'sentence-transformers' and 'torch' are installed.")
        exit()

    print(f"Loading vector database from {CHROMA_DB_DIR}...")
    try:
        # Load the Chroma vector store from the persistent directory
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"Error: Chroma DB directory '{CHROMA_DB_DIR}' not found. Please ensure you have run the data ingestion script.")
            exit()

        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        print("Vector database loaded.")
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Please ensure 'chromadb' is installed and your database exists.")
        exit()

    print("Initializing LLM...")
    try:
        # --- LLM INITIALIZATION (FOR LOCAL LLM WITH Ollama) ---
        llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,  # Controls randomness. Keep low for factual answers.
            # Ollama handles n_ctx and n_gpu_layers internally based on its build and your hardware.
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME}")

        # --- IMPORTANT: Ensure other LLM options are commented out ---
        # All LlamaCpp, OpenAI, and Google Gemini related code should be commented out.

    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running.")
        exit()

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG chain loaded successfully!")
    return qa_chain

# --- Main application logic ---
if __name__ == "__main__":
    qa_chain = load_rag_chain()

    print("\nWelcome to LexiBot PH AI Legal Assistant! (Type 'exit' to quit)")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Thank you for using LexiBot. Goodbye!")
            break
        if not user_query.strip():
            print("Please enter a question.")
            continue

        try:
            response = qa_chain.invoke({"query": user_query})

            print("\nLexiBot's Answer:")
            print(response["result"])

            if response["source_documents"]:
                print("\nSources Used:")
                for i, doc in enumerate(response["source_documents"]):
                    source_name = doc.metadata.get('source', 'Unknown Document')
                    start_index = doc.metadata.get('start_index', 'N/A')
                    print(f"  [{i+1}] Source: {source_name}, Start Index: {start_index}")
                    print(f"      Snippet: \"{doc.page_content[:200]}...\"")
            else:
                print("\n(No specific legal sources found for this query in the database.)")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or check your Ollama setup.")