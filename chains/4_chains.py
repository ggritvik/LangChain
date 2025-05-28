import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic."),
    ("human", "Provide a brief summary of the movie {movie_name}.")
])

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {plot} and tell me the summary of the movie")
    ])

    return plot_template.format_prompt(plot=plot)


def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the : {characters} and tell me their names along with their real life names")
    ])
    return character_template.format_prompt(characters=characters)


def combine(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# Simplify branches with LCEL

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x).to_messages()) 
    | model 
    | StrOutputParser()
)

characters_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x).to_messages()) 
    | model 
    | StrOutputParser()    
)


chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"plot": plot_branch_chain, "characters": characters_branch_chain})
    | RunnableLambda(lambda x: combine(x["branches"]["plot"], x["branches"]["characters"]))
    )

result = chain.invoke({
    "movie_name": "You have got a mail",
})

print(result)