import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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


#create a combined chain using LangChain Expression Language (LCEL)

chain = prompt_template | model | StrOutputParser() # This will create a chain that takes the prompt, passes it to the model, and then parses the output as a string. StrOutputParser is a simple parser that just returns the content of the string.

result = chain.invoke({
    "animal": "cheetah",
    "fact_count": 3
})

print(result)
# This code creates a chain that generates facts about a given animal using the Mistral model from OpenRouter.
# It uses a prompt template to format the input and an output parser to extract the string content from the model's response.
# The result is printed to the console.