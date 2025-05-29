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

messages = [
    ("system", "You are a facts expert who knows facts about {animal}."),
    ("human", "Tell me {fact_count} facts.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# just an alternative to the LCEL chain in 1__chains.py

format_promt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first = format_promt, 
                         middle = [invoke_model], 
                         last = parse_output)

chain = format_promt | invoke_model | parse_output

result = chain.invoke({"animal": "cheetah","fact_count": 3})

print(result)
