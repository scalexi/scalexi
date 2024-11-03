import functools
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain import hub
from langchain_core.runnables import RunnableConfig, chain
from pprint import pprint

# ===============================
# Constants and Configurations
# ===============================

# Path to the persisted Chroma vector store
VECTOR_STORE_DIRECTORY = "./Chroma_vectorstore"

# Domains to include in web search
ALJUMUAH_DOMAINS = [
    "https://www.aljumuah.com",
    "https://www.aljumuah.com/category/fiqh/",
    "https://www.aljumuah.com/category/advice/",
    "https://www.aljumuah.com/discover-islam/",
]

# LLM configurations
LLM_MODEL_NAME = "gpt-4o-mini"  # Adjust as necessary

# ===============================
# Data Models
# ===============================

class RouteQuery(BaseModel):
    """Data model for routing queries."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description=(
            "Given a user question, determine whether to route it to a web search "
            "or a vector store based on the topic of the question."
        ),
    )

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# ===============================
# Handler Classes
# ===============================

class VectorStoreHandler:
    """Handles loading and querying the vector store."""

    def __init__(self, persist_directory: str):
        # Load the embedding model (adjust the model as needed)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        # Load the Chroma vector store from disk
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
        )
        # Convert the vector store into a retriever
        self.retriever = self.vector_store.as_retriever()

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from the vector store based on the query."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        # Retrieve relevant documents
        retrieved_docs = self.retriever.get_relevant_documents(query)
        return retrieved_docs

class WebSearchHandler:
    """Handles performing web searches."""

    def __init__(self, domains: List[str], max_results: int = 5, search_depth: str = "advanced"):
        self.tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            include_domains=domains,
            name="Aljumuah_Search",
        )
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, max_tokens=1024)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a knowledgeable assistant who provides accurate and insightful answers based on the search tool results. "
                "Your scope includes understanding and clarifying teachings from AlJumuah magazine, including Islamic perspectives, "
                "guidance on Fiqh, advice, and understanding the Quran and Sunnah."
            )),
            ("human", "{user_input}"),
            ("placeholder", "{messages}"),
        ])
        self.llm_with_tools = self.llm.bind_tools([self.tool])
        self.llm_chain = self.prompt | self.llm_with_tools

    def perform_search(self, question: str) -> List[Document]:
        """Perform a web search and return the results as documents."""
        @chain
        def tool_chain(user_input: str, config: RunnableConfig):
            input_ = {"user_input": user_input}
            ai_msg = self.llm_chain.invoke(input_, config=config)
            tool_msgs = self.tool.batch(ai_msg.tool_calls, config=config)
            return self.llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
        
        result = tool_chain.invoke(question)
        page_content = result.content if hasattr(result, 'content') else str(result)
        return [Document(page_content=page_content)]

class QueryRouter:
    """Determines whether to use the vector store or web search based on the question."""

    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.system_prompt = (
            "You are an expert at routing a user question to a vector store or web search.\n"
            "The vector store contains documents related to AlJumuah magazine's teachings, focusing on understanding and living by the Final Heavenly Message from God, "
            "and following the Last Messenger, Muhammad ﷺ.\n"
            "Use the vector store for questions on these topics. Otherwise, use web search."
        )
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}"),
        ])
        self.question_router = self.route_prompt | self.structured_llm_router

    def route_question(self, question: str) -> str:
        """Route the question to the appropriate data source."""
        source = self.question_router.invoke({"question": question})
        if source.datasource == "web_search":
            return "web_search"
        elif source.datasource == "vectorstore":
            return "vector_store"
        else:
            raise ValueError(f"Invalid data source: {source.datasource}")

class AnswerGenerator:
    """Generates the final answer based on retrieved documents."""

    def __init__(self, llm_model_name: str = LLM_MODEL_NAME):
        self.llm = ChatOpenAI(model=llm_model_name, temperature=0.6, max_tokens=128)
        self.prompt_template = """
<|system|>>
You are a helpful AI Assistant that answers questions based on the uploaded documents.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generate an answer using the initial prompt and LLM."""
        # Combine documents into context
        context = "\n\n".join([doc.page_content for doc in documents])
        # Format the prompt
        formatted_prompt = self.prompt.format(context=context, query=question)
        # Generate response
        response = self.llm(formatted_prompt).content if hasattr(self.llm(formatted_prompt), 'content') else str(self.llm(formatted_prompt))
        return response

    def generate_final_answer(self, question: str, documents: List[Document]) -> str:
        """Generate the final answer using RAG chain."""
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return generation

# ===============================
# Workflow Functions
# ===============================

def retrieve_from_vector_store(state, vector_store_handler: VectorStoreHandler):
    """Retrieve documents from the vector store."""
    print("--- Retrieving from Vector Store ---")
    question = state["question"]
    documents = vector_store_handler.retrieve_documents(question)
    return {"documents": documents, "question": question}

def perform_web_search(state, web_search_handler: WebSearchHandler):
    """Perform a web search and retrieve documents."""
    print("--- Performing Web Search ---")
    question = state["question"]
    documents = web_search_handler.perform_search(question)
    return {"documents": documents, "question": question}

def generate_final_answer(state, answer_generator: AnswerGenerator):
    """Generate the final answer based on retrieved documents."""
    print("--- Generating Final Answer ---")
    question = state["question"]
    documents = state["documents"]
    generation = answer_generator.generate_final_answer(question, documents)
    return {"documents": documents, "question": question, "generation": generation}

def route_question(state, query_router: QueryRouter):
    """Route the question to the appropriate data source."""
    print("--- Routing Question ---")
    question = state["question"]
    source = query_router.route_question(question)
    if source == "web_search":
        print("--- Routing to Web Search ---")
        return "web_search"
    elif source == "vector_store":
        print("--- Routing to Vector Store ---")
        return "retrieve"
    else:
        raise ValueError("Routing failed.")

# ===============================
# Workflow Setup
# ===============================

# Create handler instances
vector_store_handler = VectorStoreHandler(VECTOR_STORE_DIRECTORY)
web_search_handler = WebSearchHandler(ALJUMUAH_DOMAINS)
query_router = QueryRouter()
answer_generator = AnswerGenerator()

# Create partial functions with handlers bound
from functools import partial

retrieve_node = partial(retrieve_from_vector_store, vector_store_handler=vector_store_handler)
web_search_node = partial(perform_web_search, web_search_handler=web_search_handler)
generate_node = partial(generate_final_answer, answer_generator=answer_generator)
route_question_node = partial(route_question, query_router=query_router)

# Compile Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_conditional_edges(
    START, route_question_node, {"web_search": "web_search", "vector_store": "retrieve"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the application
app = workflow.compile()

# ===============================
# Application Execution
# ===============================

def run_app(question: str):
    """Run the application with the given question and display outputs."""
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key in output:
            pprint(f"Node '{key}':")
            pprint(output[key])
        pprint("\n---\n")
    final_output = output.get("generation", "")
    print("Final Answer:")
    print(final_output)

# Example usage
if __name__ == "__main__":
    user_question = "What is the definition of Islam?"
    run_app(user_question)
