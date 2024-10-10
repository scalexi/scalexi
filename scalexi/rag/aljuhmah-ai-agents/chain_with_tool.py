import getpass
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain


if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

load_dotenv()

# Access the API key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")



tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    include_domains=["https://www.aljumuah.com" , "https://www.aljumuah.com/category/fiqh/" , "https://www.aljumuah.com/category/advice/" , "https://www.aljumuah.com/discover-islam/"],
    # exclude_domains=[...],
    name="Aljumuah_Search",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

#result = tool.invoke({"query": "What is the Islamic perspective on DNA testing for paternity?"})
#print(result)

llm = ChatOpenAI(model="gpt-4o-mini" , max_tokens= 1024)


prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant That answers related question from the search tool"),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# specifying tool_choice will force the model to call this tool.
llm_with_tools = llm.bind_tools([tool])

llm_chain = prompt | llm_with_tools

@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


result = tool_chain.invoke("What is the definition of Islam?")

print(result)