from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)


messages = [
    SystemMessage(content="You are an expert in biology."),
    HumanMessage(content="What's a cell?")
]

response = llm.invoke(messages)
print(response.content)