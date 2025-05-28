import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))

    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")


print("---Message History---")
print(chat_history)