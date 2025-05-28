import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

animal_facts__template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}."),
    ("human", "Tell me {fact_count} facts.")
])

translator_template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator who translates text to {language}."),
    ("human", "Translate the following text: {text}")
])

# additional processing steps using runnableLambda
prepare_for_translation = RunnableLambda(lambda output: {
    "text": output,
    "language": "French"
})

chain = animal_facts__template | model | StrOutputParser() | prepare_for_translation | translator_template | model | StrOutputParser()

result = chain.invoke({"animal": "dog","fact_count": 1})

print(result)
