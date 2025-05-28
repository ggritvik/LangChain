import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory


load_dotenv()


llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)


PROJECT_ID = "langchain-f8192"
SESSION_ID = "langchain-session-2"
COLLECTION_NAME = "chat-history"

# Initialize Firestore client
print("Initializing Firestore client...")

client = firestore.Client(project=PROJECT_ID)


#Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    client=client,
    session_id=SESSION_ID,
    collection=COLLECTION_NAME
)
print("Firestore Chat Message History initialized.")
#print("Current Chat history:", chat_history.messages)




print("Starting chat...")
print("You can type 'exit' to end the chat.")


while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break
    
    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
