import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir =  os.path.join(db_dir, "chroma_db_apple")

urls = ["https://www.apple.com/in/"]

loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

print("\n------ Doc Chunk info ------")
print(f"Number of documents: {len(docs)}")
print(f"First document chunk: {docs[0].page_content[:100]}...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist, initializing vector database...")

    # Create a persistent vector store (Chroma DB) from the document chunks and embeddings
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    print("\n--- Finished creating vector store ---")
else:
    print("Persistent directory already exists, loading existing vector store...")
    db = Chroma(
        persist_directory=persistent_dir,
        embedding_function=embeddings
    )


retriver = db.as_retriever(
    search_type="similarity",    
    search_kwargs={"k": 3}
)

query = "What is the latest MacBook model?"

relevant_docs = retriver.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n {doc.page_content}...\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")