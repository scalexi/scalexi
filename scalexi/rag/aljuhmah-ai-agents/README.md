# Aljuhmah-AI-Agents
Different code implementations for Aljumah Project (RAG + SEARCH TOOL)

# Repository Overview

This repository contains several Python scripts designed to handle various operations related to PDF document processing, vector store creation, and retrieval-based search tools using large language models (LLMs). Below is an overview of each script along with explanations of their purpose and functionality.

## 1. `Create_Vectorstore_From_PDFs.py`

### Description:
This script is responsible for extracting data from PDF documents and creating a vector store using ChromaDB. The generated vector store allows for efficient document retrieval. The process is optimized for a large collection of PDFs, with all the vectors being saved in a .zip file.

### Output:
- **Chroma Vectorstore**: The generated vector store for all Aljumah resources is saved in a file named `Chroma_vectorstore.zip`.
- **Statistics**: After processing, the script generates a stats file containing PDF extraction statistics:
    ```
    --- PDF Extraction Statistics ---
    Total Documents Processed: 191
    Successful Documents: 191
    Failed Documents: 0
    Total Pages Processed: 11841
    Successful Pages: 11569
    Failed Pages: 0
    ----------------------------------
    ```

---

## 2. `Load_Vectorestore_and_perform_retrieval_with_llm.py`

### Description:
This script loads the vector store previously created and uses it to perform retrieval operations through an LLM (Language Learning Model) retrieval chain. It allows customization of the OpenAI model using a template prompt provided in the code.

### Key Features:
- Loads saved vector store for document retrieval.
- Executes Retrieval-Augmented Generation (RAG) using a customizable OpenAI model prompt.

---

## 3. `search_tool.py`

### Description:
This script implements a search tool using the `TavilySearchResults()` function. It takes a user query and performs a search optimized for specific domains. The tool efficiently handles domain-specific searches for relevant information.

### Key Features:
- Specialized search on specific domains.
- Efficient retrieval based on user questions.

---

## 4. `chain_with_tool.py`

### Description:
This script combines a domain-specific search tool with an LLM chain. It integrates the search results with the LLM to produce better responses. The chain uses templates and enforces a maximum token limit for optimized responses.

### Key Features:
- Domain-specific search with LLM integration.
- Custom LLM chain using prompt templates for better results.
- Supports a maximum token setup to manage output length.

---

## 5. `router.py`

### Description:
This script implements a router using LangGraph that combines both the search tool and RAG. Based on the user query, the router decides whether to retrieve data from the vector store or perform a search using the search tool.

### Key Features:
- Decision-based routing between vector store retrieval and search tool.
- Combines RAG and search for optimal answers based on the user's query.

---

## 6. `Enhanced_rag_with_search_tool_router.py`

### Description:
This script enhances the LLM chain by incorporating both the search tool and retrieval-based RAG. It uses advanced methods, including routing algorithms, to combine results from the search tool and vector store retrieval for better performance.

### Key Features:
- Enhanced integration of LLM chain with both search tool and retrieval RAG.
- Improved methods for combining chains, router algorithms, and prompt templates.

---

## 7. `RAG_With_Search_tool.py`

### Description:
This script provides dual answers to a user query: one from vector store retrieval and one from the search tool. It allows comparison between the two results and can also generate a combined answer using both retrieval and search results.

### Key Features:
- Dual answers from vector store retrieval and search tool.
- Option to combine the results for a more comprehensive response.

---

## Environment Variables

Make sure to configure the following API keys in a `.env` file before running the scripts:
- OPENAI_API_KEY="" 
- TAVILY_API_KEY="" 
- LANGCHAIN_API_KEY=""


These API keys are essential for interacting with the respective LLMs and search services in the scripts.

---

## Conclusion

This repository provides a flexible framework for working with PDF documents, creating vector stores, and leveraging search tools integrated with large language models. By using the provided scripts, you can handle retrieval-based tasks efficiently with customizable prompts and domain-specific search options.
