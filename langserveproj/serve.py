from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Define the prompt template
system_template = "translate the following into the {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Create the final chain
parser = StrOutputParser()
chain = prompt_template | model | parser

# Define the FastAPI app
app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple API server using LangChain Runnable interfaces"
)

# Add chain routes
# Add chain routes
add_routes(
    app,
    chain,
    path="/chain"
)


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
