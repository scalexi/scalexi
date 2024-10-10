from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pprint import pprint
import getpass
import os
from dotenv import load_dotenv


if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

load_dotenv()

# Access the API key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")


# Load Chroma vectorstore

def load_chroma_vectorstore(persist_directory):
    # Load the embedding model (use the correct import and model)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Adjust this to match the correct model

    # Load the Chroma vector store from disk
    chroma_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    return chroma_store

# Path to the persisted Chroma vector store
persist_directory = "./Chroma_vectorstore"

# Load the vector store
chroma_store = load_chroma_vectorstore(persist_directory)

# Convert the vector store into a retriever
vectorstore_retriever = chroma_store.as_retriever()

# Data model for routing queries
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., description="Given a user question choose to route it to web search or a vectorstore."
    )


# Set up LLM and routing prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to AlJumuah magazine's teachings, focusing on understanding and living by the Final Heavenly Message from God, and following the Last Messenger, Muhammad ﷺ.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# Define Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Web search tool setup
web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    include_domains=[
        "https://www.aljumuah.com",
        "https://www.aljumuah.com/category/fiqh/",
        "https://www.aljumuah.com/category/advice/",
        "https://www.aljumuah.com/discover-islam/"
    ],
    name="Aljumuah_Search"
)

# Graph nodes

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = vectorstore_retriever.invoke(question)
    return {"documents": documents, "question": question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": [web_results], "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    prompt = hub.pull("rlm/rag-prompt")
    llm_rag = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens= 512)
    rag_chain = prompt | llm_rag | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Routing logic

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"

# Compile Graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_conditional_edges(
    START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Example usage
inputs = {"question": "What are the rights of the poor in Islam?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])