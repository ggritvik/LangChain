# BLOCK 1: IMPORTS AND DEPENDENCIES
# 1. Import os module for operating system interface functions (file paths, environment variables)
import os
# 2. Import load_dotenv to read environment variables from .env file
from dotenv import load_dotenv

# 3. Import Chroma vector database for storing and retrieving document embeddings
from langchain_chroma import Chroma
# 4. Import HuggingFace embeddings model for converting text to vector representations
from langchain_huggingface import HuggingFaceEmbeddings

# 5. Import ChatOpenAI for language model interface
from langchain_openai import ChatOpenAI
# 6. Import prompt templates for structured conversation handling
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 7. Import message types for chat history management
from langchain_core.messages import HumanMessage, SystemMessage

# 8. Import chain creation functions for document processing and retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# BLOCK 2: ENVIRONMENT AND DIRECTORY SETUP
# 9. Load environment variables from .env file into the current environment
load_dotenv()

# 10. Get the absolute path of the directory containing this Python file
current_dir = os.path.dirname(os.path.abspath(__file__))
# 11. Create path to the persistent Chroma database directory with metadata
persitent_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# BLOCK 3: EMBEDDINGS MODEL INITIALIZATION
# 12. Initialize HuggingFace embeddings using a pre-trained sentence transformer model
# 13. This model converts text into 384-dimensional vectors for semantic similarity
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

# BLOCK 4: VECTOR DATABASE SETUP
# 14. Initialize Chroma vector database instance
# 15. Connect to existing persistent database directory
# 16. Use the HuggingFace embeddings function for text vectorization
db = Chroma(
    persist_directory=persitent_dir, 
    embedding_function=embeddings
)

# BLOCK 5: RETRIEVER CONFIGURATION
# 17. Create a retriever from the vector database for document search
# 18. Set search type to "similarity" for semantic matching
# 19. Configure to return top 3 most similar documents (k=3)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# BLOCK 6: LANGUAGE MODEL SETUP
# 20. Initialize ChatOpenAI with Mistral 7B model via OpenRouter API
# 21. Specify the exact model name for the Mistral 7B Instruct variant
# 22. Get API key from environment variable for authentication
# 23. Set OpenRouter API base URL for model access
model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# BLOCK 7: CONTEXT-AWARE QUESTION REFORMULATION PROMPT
# 24. Define system prompt for contextualizing user questions
# 25. Instructs model to create standalone questions from chat history context
# 26. Ensures questions can be understood without previous conversation
# 27. Explicitly tells model NOT to answer, only reformulate if needed

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)


# BLOCK 8: CONTEXTUALIZATION PROMPT TEMPLATE
# 28. Create chat prompt template for question contextualization
# 29. Include system message with contextualization instructions
# 30. Add placeholder for chat history to maintain conversation context
# 31. Include human input placeholder for the current user question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# BLOCK 9: HISTORY-AWARE RETRIEVER CREATION
# 32. Create a retriever that considers chat history when searching documents
# 33. Combines the language model, document retriever, and contextualization prompt
# 34. Enables retrieval based on reformulated standalone questions
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# BLOCK 10: QUESTION-ANSWERING SYSTEM PROMPT
# 35. Define system prompt for answering questions based on retrieved context
# 36. Instructs AI to use only the provided context for answers
# 37. Sets fallback behavior when answer is unknown ("don't know")
# 38. Limits response length to maximum three sentences for conciseness
# 39. Includes context placeholder for retrieved document content
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# BLOCK 11: QUESTION-ANSWERING PROMPT TEMPLATE
# 40. Create chat prompt template for the Q&A interaction
# 41. Include system message with Q&A instructions and context
# 42. Add human message placeholder for user questions
qa_user_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ]
)

# BLOCK 12: DOCUMENT PROCESSING CHAIN CREATION
# 43. Create a chain that combines retrieved documents with the language model
# 44. Uses the "stuff" strategy to include all retrieved documents in the prompt
# 45. Links the model with the Q&A prompt template for coherent responses
qa_chain = create_stuff_documents_chain(model, qa_user_prompt)

# BLOCK 13: COMPLETE RAG CHAIN ASSEMBLY
# 46. Create the final Retrieval-Augmented Generation (RAG) chain
# 47. Combines history-aware retrieval with document-based question answering
# 48. Enables conversational AI with context-aware document retrieval
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# BLOCK 14: CONTINUOUS CHAT FUNCTION DEFINITION
# 49. Define function to handle ongoing conversation with the AI
def continual_chat():
    # 50. Display welcome message and exit instructions to user
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    # 51. Initialize empty list to store conversation history (messages sequence)
    chat_history = []  # Collect chat history here (a sequence of messages)
    # 52. Start infinite loop for continuous conversation
    while True:
        # 53. Get user input and store in query variable
        query = input("You: ")
        # 54. Check if user wants to exit the conversation
        if query.lower() == "exit":
            # 55. Break out of the loop to end the conversation
            break
        # 56. Process user query through the complete RAG chain
        # 57. Pass both current input and accumulated chat history
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # 58. Display the AI's response from the result dictionary
        print(f"AI: {result['answer']}")
        # 59. Add user's question to chat history as HumanMessage
        chat_history.append(HumanMessage(content=query))
        # 60. Add AI's response to chat history as SystemMessage for context
        chat_history.append(SystemMessage(content=result["answer"]))


# BLOCK 15: MAIN EXECUTION BLOCK
# 61. Check if this script is being run directly (not imported as module)
# 62. Execute the continuous chat function to start the interactive session
if __name__ == "__main__":
    continual_chat()