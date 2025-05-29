import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

postive_feedback_template = ChatPromptTemplate.from_messages(
    [
    ('system', "You are a helpful assistant."),
    ('human', "Generate a positive thankyou note for the following text: {text}")
])

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
    ('system', "You are a helpful assistant."),
    ('human', "Generate a resposnnse addressing the negative feedback: {text}")
])

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
    ('system', "You are a helpful assistant."),
    ('human', "Generate a resposnnse addressing the neutral feedback:{text}")
])

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
    ('system', "You are a helpful assistant."),
    ('human', "Generate a response for proper escalation to a customer support agent:  {text}")
])


classification_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant"),
    ('human', "classify the  customer feedback into one of the following sentiment categories: positive, negative, neutral, escalate.{text}")
])

branches = RunnableBranch(
    (
        lambda x : 'positive' in x,
            postive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x : 'negative' in x,
            negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: 'neutral' in x,
            neutral_feedback_template | model | StrOutputParser(),
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

review = "The product was great, but the delivery was late."
result = chain.invoke({"text": review})
print(result)