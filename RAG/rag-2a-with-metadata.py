import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- SETUP DIRECTORIES ---

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory containing the text files (books)
books_dir = os.path.join(current_dir, "books")

# Directory to store the database
db_dir = os.path.join(current_dir, "db")

# Directory where the Chroma vector store will be persisted
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# --- CHECK IF VECTOR STORE EXISTS ---

# If the persistent directory does not exist, we need to create the vector store
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the books directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # --- LOAD DOCUMENTS AND ADD METADATA ---

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")  # Loads the text file
        book_docs = loader.load()       # Returns a list of Document objects
        for doc in book_docs:
            # Add metadata to each document indicating its source file
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # --- SPLIT DOCUMENTS INTO CHUNKS ---

    # Split the documents into smaller chunks for embedding
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # --- CREATE EMBEDDINGS ---

    print("\n--- Creating embeddings ---")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    print("\n--- Finished creating embeddings ---")

    # --- CREATE AND PERSIST VECTOR STORE ---

    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    # If the persistent directory exists, skip initialization
    print("Vector store already exists. No need to initialize.")


'''
Code Explanation
1. Imports:
Imports necessary modules for file handling, text splitting, document loading, vector storage, and embeddings.

2. Directory Setup:

Determines the current script directory.
Sets up paths for the books directory (where your .txt files are), the database directory, and the persistent directory for the Chroma vector store.

3. Check for Existing Vector Store:

If the persistent directory does not exist, it initializes the vector store.
If it does exist, it skips initialization (to avoid overwriting existing data).

4. Load Documents:

Lists all .txt files in the books directory.
Loads each file as a document and attaches metadata indicating the source file name.

5. Split Documents:

Uses CharacterTextSplitter to break each document into chunks of 1000 characters (no overlap).
This is important for embedding, as models often have input size limits.

6. Create Embeddings:

Uses OpenAI’s embedding model (text-embedding-3-small) to convert each chunk into a vector representation.

7. Create and Persist Vector Store:

Stores the embeddings and associated metadata in a Chroma vector store, persisted to disk.


8. Re-Initialization Check:

If the vector store already exists, the script prints a message and does nothing further.
'''