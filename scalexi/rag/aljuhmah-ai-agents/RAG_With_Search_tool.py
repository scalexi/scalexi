from langchain_chroma import Chroma
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
from langchain_core.runnables import RunnableConfig, chain

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
        ..., description="Given a user question, determine whether to route it to a web search or a vectorstore based on the topic of the question."
    )

# Set up LLM and routing prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
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
    include_images=True,
    include_domains=[
        "https://www.aljumuah.com",
        "https://www.aljumuah.com/category/fiqh/",
        "https://www.aljumuah.com/category/advice/",
        "https://www.aljumuah.com/discover-islam/"
    ],
    name="Aljumuah_Search"
)

# Graph nodes

def both_retrieve_and_search(state):
    print("---RETRIEVE AND SEARCH---")
    retrieve_result = retrieve(state)
    print(f"Vector Store Result: {retrieve_result.page_content}")  # Print vector store result
    web_search_result = web_search(state)
    print(f"Web Search Result: {web_search_result['documents'][0].page_content}")  # Print web search result
    return {"documents": [retrieve_result, *web_search_result['documents']], "question": state["question"]}

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    response = perform_retrieval(question, vectorstore_retriever)
    return Document(page_content=response)

# Function to create a retrieval chain and perform retrieval
def perform_retrieval(query, vectorstore_retriever):
    # Ensure the query is a string
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    # Retrieve relevant documents from the vector store
    retrieved_docs = vectorstore_retriever.invoke(query)
    
    # Combine the retrieved documents' content into a single string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Setup the LLM (ChatOpenAI) with your desired parameters
    llm = ChatOpenAI(temperature=0.6, max_tokens=128, model_name="gpt-4o-mini")

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
    response = llm.invoke(formatted_prompt).content if hasattr(llm.invoke(formatted_prompt), 'content') else str(llm.invoke(formatted_prompt))
    
    return response

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        include_domains=[
            "https://www.aljumuah.com",
            "https://www.aljumuah.com/category/fiqh/",
            "https://www.aljumuah.com/category/advice/",
            "https://www.aljumuah.com/discover-islam/"
        ],
        name="Aljumuah_Search"
    )
    result = tool.invoke({"query": question})
    print(f"Full Web Search Result: {result}")  # Debug: Print the entire web search result
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    prompt = ChatPromptTemplate([
        ("system", f"You are a knowledgeable assistant who provides accurate and insightful answers based on the search tool results. Your scope includes understanding and clarifying teachings from AlJumuah magazine, including Islamic perspectives, guidance on Fiqh, advice, and understanding the Quran and Sunnah."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ])
    llm_with_tools = llm.bind_tools([tool])
    llm_chain = prompt | llm_with_tools

    @chain
    def tool_chain(user_input: str, config: RunnableConfig):
        input_ = {"user_input": user_input}
        ai_msg = llm_chain.invoke(input_, config=config)
        tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
        return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

    result = tool_chain.invoke(question)
    if hasattr(result, 'content') and result.content:
        page_content = result.content
    else:
        page_content = "No relevant content found in web search."
    return {"documents": [Document(page_content=page_content)], "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    vector_store_content = "\n\n".join([doc.page_content for doc in documents if 'Vector Store' in doc.page_content])
    web_search_content = "\n\n".join([doc.page_content for doc in documents if 'Web Search' in doc.page_content])
    combined_documents = f"[Vector Store Result]\n{vector_store_content}\n\n[Web Search Result]\n{web_search_content}"
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": combined_documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Routing logic

def route_question(state):
    print("---ROUTE QUESTION TO BOTH WEB SEARCH AND VECTORSTORE---")
    return "both"

# Compile Graph
workflow = StateGraph(GraphState)
workflow.add_node("both", both_retrieve_and_search)
workflow.add_node("generate", generate)
workflow.add_edge(START, "both")
workflow.add_edge("both", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Example usage
inputs = {"question": "What is the definition of Islam?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        pprint(value)
    pprint("\n---\n")