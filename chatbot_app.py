# chatbot_app.py
import os
import re # Still needed for re.sub if not moved
from datetime import datetime

# Core LangChain components for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Ollama for local LLM inference
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import the new function and constants from document_manager.py
from document_manager import handle_document_filling, DOCUMENT_TEMPLATES


# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral"


def load_rag_chain():
    """
    Loads the embedding model, vector database, and sets up the RAG chain.
    """
    print("Loading embedding model for retrieval...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure 'sentence-transformers' and 'torch' are installed.")
        exit()

    print(f"Loading vector database from {CHROMA_DB_DIR}...")
    try:
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
        llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
        )
        print(f"Using local LLM via Ollama: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Local LLM via Ollama: {e}")
        print("Please ensure Ollama is installed, the model is pulled, and the Ollama server is running.")
        exit()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG chain loaded successfully!")
    return qa_chain, llm

def detect_document_request(query, llm):
    """
    Uses an LLM to determine if the query is a request for a document template
    and identifies which document type.
    """
    # Use DOCUMENT_TEMPLATES imported from document_manager
    template_names = ", ".join(DOCUMENT_TEMPLATES.keys())
    
    prompt_template = PromptTemplate(
        input_variables=["query", "template_names"],
        template="""You are an AI assistant. Analyze the user's query to determine if they are asking for a legal document template.
        If they are, identify which specific document they are asking for from the following types: {template_names}.
        If you identify a document, respond ONLY with the document type (e.g., "nda", "non-disclosure agreement").
        If the query is NOT a document request, respond ONLY with "NONE".

        Examples:
        User: I need an NDA.
        Response: nda

        User: Can you help me draft a non-disclosure agreement?
        Response: non-disclosure agreement

        User: What are the tax requirements for a new business?
        Response: NONE

        User: Draft a simple contract.
        Response: NONE

        Query: {query}
        Response:"""
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    llm.temperature = 0.3
    response = llm_chain.invoke({"query": query, "template_names": template_names})
    llm.temperature = 0.1

    detected_type = response['text'].strip().lower()

    if detected_type in DOCUMENT_TEMPLATES:
        return detected_type
    return "NONE"


# --- Main application logic ---
if __name__ == "__main__":
    qa_chain, llm = load_rag_chain()

    print("\nWelcome to LexiBot PH AI Legal Assistant! (Type 'exit' to quit)")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Thank you for using LexiBot. Goodbye!")
            break
        if not user_query.strip():
            print("Please enter a question.")
            continue

        detected_doc_type = detect_document_request(user_query, llm)

        if detected_doc_type != "NONE":
            template_filename = DOCUMENT_TEMPLATES[detected_doc_type]
            print(f"\nLexiBot: It looks like you're asking for a '{detected_doc_type}' template. I have a template for that: {template_filename}.")
            print("LexiBot: Would you like me to help you fill it out? (yes/no)")
            
            confirm_fill = input("Your answer: ").strip().lower()
            if confirm_fill == 'yes':
                # Call the imported handle_document_filling function
                handle_document_filling(detected_doc_type)
            else:
                print("LexiBot: Okay, no problem. What else can I help you with?")
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