import os
import re
from datetime import datetime

# Core LangChain components for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Ollama for local LLM inference
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "mistral"

# --- Document Template Configuration ---
TEMPLATE_DIR = "./document_templates"
DOCUMENT_TEMPLATES = {
    "nda": "simple_nda_template.txt",
    "non-disclosure agreement": "simple_nda_template.txt",
    # Add more mappings as you create more templates (e.g., "lease agreement": "lease_agreement_template.txt")
}

# --- NEW: Placeholder Descriptions ---
# This dictionary maps placeholder names (from your templates) to user-friendly explanations.
# You will need to expand this as you add more templates and placeholders.
PLACEHOLDER_DESCRIPTIONS = {
    "PARTY_ONE_NAME": "The full legal name of the Disclosing Party (the one sharing confidential information)",
    "PARTY_ONE_ADDRESS": "The complete address of the Disclosing Party",
    "PARTY_TWO_NAME": "The full legal name of the Receiving Party (the one receiving confidential information)",
    "PARTY_TWO_ADDRESS": "The complete address of the Receiving Party",
    "CONFIDENTIAL_INFO_DESCRIPTION": "A brief description of the type of confidential information being shared (e.g., business plans, product designs, customer lists)",
    "CONFIDENTIAL_INFO_EXAMPLES": "Specific examples of confidential information (e.g., 'technical data, formulas, marketing strategies')",
    "agreement_term_months": "The duration of the agreement in months (e.g., 12 for one year)",
    # Add descriptions for other placeholders here, if you add them to your templates
    # "CLIENT_NAME": "Full legal name of the client",
    # "SERVICE_DESCRIPTION": "Detailed description of the services to be provided",
}


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

def handle_document_filling(template_key):
    """
    Guides the user through filling out a document template.
    """
    template_filename = DOCUMENT_TEMPLATES[template_key]
    template_path = os.path.join(TEMPLATE_DIR, template_filename)

    if not os.path.exists(template_path):
        print(f"LexiBot: Error: Template file '{template_filename}' not found at '{template_path}'.")
        return

    print(f"\nLexiBot: Okay, let's fill out your '{template_key}' template.")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        print(f"LexiBot: Error reading template file: {e}")
        return

    placeholders = set(re.findall(r'\[(.*?)\]|\{\{(.*?)\}\}', template_content))
    placeholders = {p.strip() for tup in placeholders for p in tup if p.strip()}
    
    if 'current_date' in placeholders:
        placeholders.remove('current_date')

    filled_data = {}
    print("\nLexiBot: Please provide the following details:")

    if 'current_date' in template_content:
        filled_data['current_date'] = datetime.now().strftime("%B %d, %Y")
        print(f"LexiBot: Setting current date to: {filled_data['current_date']}")

    for placeholder in sorted(list(placeholders)):
        # --- NEW: Get and display description for the placeholder ---
        description = PLACEHOLDER_DESCRIPTIONS.get(placeholder, placeholder.replace('_', ' ').title())
        user_input = input(f"LexiBot: {description}: ") # Use description in prompt
        filled_data[placeholder] = user_input

    filled_document = template_content
    for placeholder, value in filled_data.items():
        filled_document = re.sub(rf"\[{re.escape(placeholder)}\]|" + r"\{\{" + re.escape(placeholder) + r"\}\}", value, filled_document)

    print("\n" + "="*50)
    print("LexiBot: Here is your filled document preview:")
    print("="*50)
    print(filled_document)
    print("="*50 + "\n")

    print("LexiBot: Please review the document carefully.")
    review_correct = input("LexiBot: Is everything correct? (yes/no): ").strip().lower()

    if review_correct == 'yes':
        print("LexiBot: Great! The document is finalized.")
        output_filename = f"filled_{template_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_path = os.path.join(TEMPLATE_DIR, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(filled_document)
            print(f"LexiBot: Document saved as '{output_filename}' in the '{TEMPLATE_DIR}' folder.")
            print("LexiBot: (Mock process for sending to government agency complete.)")
        except Exception as e:
            print(f"LexiBot: Error saving document: {e}")
    else:
        print("LexiBot: Okay, please indicate what needs to be changed for future improvements.")
        print("LexiBot: For now, you can manually edit the content from the preview above.")

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