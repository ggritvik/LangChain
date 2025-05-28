import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

## From Template
# This example demonstrates how to use a template to generate a prompt for the LLM.

# template = "Write a {tone} email to {company} expressing interest in {postion} internship, mentioning {skills} as key strength and {experience} as experience. keep it 6 lines maximum."

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#     "tone": "formal",
#     "company": "Google",
#     "postion": "software engineering",
#     "skills": "Python, Java, and C++",
#     "experience": "6 month internship at XYZ Corp"
# })

# result = llm.invoke(prompt)  # Call the LLM with the generated prompt

# print(result.content)


### From Messages

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {number} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    "topic": "AI",
    "number": 3
})

result = llm.invoke(prompt)  # Call the LLM with the generated prompt

print(result.content)