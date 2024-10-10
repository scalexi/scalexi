import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import openai

# Load environment variables, including OpenAI API keys
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Function to load the vector store
def load_chroma_vectorstore(persist_directory):
    # Load the embedding model (use the correct import and model)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Adjust this to match the correct model

    # Load the Chroma vector store from disk
    chroma_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    return chroma_store

# Function to create a retrieval chain and perform retrieval
def perform_retrieval(query, vectorstore_retriever):
    # Ensure the query is a string
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    # Retrieve relevant documents from the vector store
    retrieved_docs = vectorstore_retriever.get_relevant_documents(query)
    
    # Combine the retrieved documents' content into a single string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Setup the LLM (ChatOpenAI) with your desired parameters
    llm = ChatOpenAI(temperature=0.6, max_tokens=128, model_name="gpt-4")

    # Define the custom template for the retrieval chain
    template = """
    <|system|>>
    You are a helpful AI Assistant that answers questions based on the uploaded documents.

    CONTEXT: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """

    # Setup the prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Fill the template with context and query
    formatted_prompt = prompt.format(context=context, query=query)

    # Run the language model with the formatted prompt
    response = llm(formatted_prompt)
    
    return response

# Path to the persisted Chroma vector store
persist_directory = "./Chroma_vectorstore"

# Load the vector store
chroma_store = load_chroma_vectorstore(persist_directory)

# Convert the vector store into a retriever
vectorstore_retriever = chroma_store.as_retriever()

# Example query
query = "What is the definition of Islam?"

# Perform the retrieval
try:
    result = perform_retrieval(query, vectorstore_retriever)
    # Print the result of the query
    print(f"Query: {query}")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {e}")
