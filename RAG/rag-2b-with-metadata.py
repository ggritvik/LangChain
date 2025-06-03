import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --- SETUP PERSISTENT DIRECTORY ---

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up the database directory and the persistent directory for the Chroma vector store
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# --- INITIALIZE EMBEDDING MODEL ---

# Define the embedding model to use for queries
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

# --- LOAD EXISTING VECTOR STORE ---

# Load the Chroma vector store from disk, using the embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# --- DEFINE USER QUERY ---

# The question you want to ask the vector store
query = "How did Juliet die?"

# --- RETRIEVE RELEVANT DOCUMENTS ---

# Create a retriever object with similarity search and a score threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)

# Retrieve the most relevant documents for the query
relevant_docs = retriever.invoke(query)

# --- DISPLAY RESULTS ---

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")