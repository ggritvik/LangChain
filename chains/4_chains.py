import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

# Initialize the chat model with OpenRouter API and Mistral model
model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Define a prompt template for summarizing a movie
summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic."),
    ("human", "Provide a brief summary of the movie {movie_name}.")
])

# Function to create a prompt for analyzing the plot
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {plot} and tell me the summary of the movie")
    ])
    return plot_template.format_prompt(plot=plot)

# Function to create a prompt for analyzing the characters
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the : {characters} and tell me their names along with their real life names")
    ])
    return character_template.format_prompt(characters=characters)

# Function to combine the plot and character analyses into a single string
def combine(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# Define a chain for analyzing the plot branch:
# 1. Format the plot prompt
# 2. Pass it to the model
# 3. Parse the output as a string
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x).to_messages()) 
    | model 
    | StrOutputParser()
)

# Define a chain for analyzing the characters branch:
# 1. Format the character prompt
# 2. Pass it to the model
# 3. Parse the output as a string
characters_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x).to_messages()) 
    | model 
    | StrOutputParser()    
)

# Define the main chain:
# 1. Use the summary template to get a summary from the model
# 2. Parse the summary output
# 3. In parallel, run plot and character analysis branches
# 4. Combine the results from both branches into a final output
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"plot": plot_branch_chain, "characters": characters_branch_chain})
    | RunnableLambda(lambda x: combine(x["branches"]["plot"], x["branches"]["characters"]))
    )

# Invoke the chain with a specific movie name
result = chain.invoke({
    "movie_name": "Mission Impossible - Dead Reckoning Part One",
})

# Print the final combined analysis
print(result)