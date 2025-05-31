import os
from dotenv import load_dotenv
# Import necessary modules from LangChain and community extensions
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# from langchain_community.embeddings import OpenAIEmbeddings

# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the text file and the persistent vector database directory
file_path = os.path.join(current_dir, "books", "secret-of-old-oak.txt")
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

# Check if the persistent vector database directory exists
if not os.path.exists(persistent_dir):
    print("persistent_dir does not exist, Initializing Vector database...")

    # Check if the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the text file as documents
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split the documents into chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n------ Doc Chunk info ------")
    print(f"Total number of chunks: {len(docs)}")  
    print(f"Sample Chunk:\n {docs[0].page_content}\n")

    # Create embeddings for the document chunks using OpenAI
    print("\n ----- Creating Embeddings -----")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


    print("Embeddings created successfully.")

    # Create a persistent vector store (Chroma DB) from the document chunks and embeddings
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")