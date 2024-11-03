import getpass
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

from langchain_community.tools import TavilySearchResults

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

result = tool.invoke({"query": "What is Salafiya?"})

print(result)
