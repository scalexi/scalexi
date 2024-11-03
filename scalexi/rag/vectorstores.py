import os
import openai
from PyPDF2 import PdfReader
from scalexi.utilities.logger import Logger
from dotenv import load_dotenv
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS



class VectorstoreManager:
    """
    Class to build a vector store from PDF documents.

    This class loads PDF files from a specified directory, extracts text, splits the text into smaller chunks,
    generates embeddings using OpenAI's embedding model, and saves the embeddings into a vector store such as Chroma.

    Attributes:
        config (dict): Configuration dictionary loaded from a YAML file.
        directory_path (str): Path to the directory containing PDF files.
        persist_directory (str): Path to the directory where the vector store will be persisted.
        api_key (str): OpenAI API key loaded from environment variables.
        embedding_model (OpenAIEmbeddings): OpenAI model used to generate text embeddings.
        logger (Logger): Logger for logging progress and errors.
    """

    def __init__(self, config):
        """
        Initializes the VectorstoreManager class.

        Args:
            config (dict): Configuration dictionary loaded from a YAML file.
        """
        load_dotenv()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
        

        # Extract parameters from config
        vectorstore_config = config.get('vectorstore', {})
        embedding_model_config = config.get('embedding_model', {})
        retrieval_config = config.get('retrieval', {})

        # Set directory paths
        #base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script resides
        #data_dir = config.get('vectorstore', {}).get('directory_path', 'data')
        #self.directory_path = os.path.join(base_path, data_dir)
        #persist_dir = config.get('vectorstore', {}).get('persist_directory', 'data/vectorstores')
        #self.persist_directory = os.path.join(base_path, persist_dir)
        # Set directory paths to absolute
        self.directory_path = os.path.abspath(vectorstore_config.get('directory_path', 'data'))
        self.persist_directory = os.path.abspath(vectorstore_config.get('persist_directory', 'data/vectorstores'))
        self.vectorstore_filename = vectorstore_config.get('vectorstore_filename', 'vectorstore')

        #self.directory_path = vectorstore_config.get('directory_path', 'data')
        #self.persist_directory = vectorstore_config.get('persist_directory', 'data/vectorstores')
        #self.vectorstore_filename = vectorstore_config.get('vectorstore_filename', 'vectorstore')

        # Set chunk size and overlap
        self.chunk_size = vectorstore_config.get('chunk_size', 1000)
        self.chunk_overlap = vectorstore_config.get('chunk_overlap', 200)

        # Set text splitter
        text_splitter_config = vectorstore_config.get('text_splitter', {})
        self.text_splitter_type = text_splitter_config.get('type', 'semantic')
        self.text_splitter_model_name = text_splitter_config.get('model', 'text-embedding-3-small')

        # Set embedding model
        self.embedding_provider = embedding_model_config.get('provider', 'openai')
        self.embedding_model_name = embedding_model_config.get('model', 'text-embedding-3-small')

        # Initialize embedding model
        if self.embedding_provider == 'openai':
            self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
        else:
            # Handle other providers
            raise NotImplementedError(f"Embedding provider '{self.embedding_provider}' is not implemented.")

        self.logger = Logger().get_logger()
        self.logger.info("VectorstoreManager initialized")
        self.vectorstore = None
        self.stats = None
        self.stats_filename = "extraction_stats.txt"
        self.retriever = None

        # Set retrieval parameters
        self.top_k_docs = retrieval_config.get('top_k_docs', 4)
        self.search_type = retrieval_config.get('search_type', 'similarity')
        self.score_threshold = retrieval_config.get('score_threshold', 0.5)
        self.fetch_k = retrieval_config.get('fetch_k', 10)
        self.lambda_mult = retrieval_config.get('lambda_mult', 0.6)
        # Initialize text splitter
        self.vectorstore_type = vectorstore_config.get('type', 'chroma')
        self.initialize_text_splitter()
        self.configure_vector_store()

    def load_pdfs(self):
        """
        Loads PDF documents from the specified directory, extracting text from each page.

        Returns:
            tuple: A list of extracted documents and a statistics dictionary that tracks total and successful
                   document and page extraction counts.
        """
        self.stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_pages": 0,
            "successful_pages": 0,
            "failed_pages": 0,
        }
        documents = []
        self.logger.info(f"Starting to load PDFs from {self.directory_path}")
        for root, dirs, files in os.walk(self.directory_path):
            for filename in files:
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(root, filename)
                    self.logger.info(f"Loading PDF: {filename}")
                    self.stats, doc = self.process_pdf(pdf_path, filename, self.stats)
                    if doc:
                        documents.append(doc)
        self.logger.info("PDF loading completed")
        return documents

    def process_pdf(self, pdf_path, filename, stats):
        """
        Processes a single PDF, extracting text from each page.

        Args:
            pdf_path (str): Path to the PDF file.
            filename (str): Name of the PDF file.
            stats (dict): Dictionary to track statistics.

        Returns:
            tuple: Updated statistics dictionary and extracted Document object.
        """
        try:
            self.logger.info(f"Processing PDF: {filename}")
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            stats["total_documents"] += 1
            stats["total_pages"] += num_pages
            document_text = []
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        document_text.append(text)
                        stats["successful_pages"] += 1
                    else:
                        stats["failed_pages"] += 1
                        self.logger.debug(f"No text on page {page_num} of {filename}")
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num} of {filename}: {e}")
                    stats["failed_pages"] += 1
            if document_text:
                stats["successful_documents"] += 1
                return stats, Document(page_content="\n".join(document_text))
            else:
                stats["failed_documents"] += 1
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            stats["failed_documents"] += 1
        return stats, None

    def split_documents(self, documents):
        """
        Splits documents into smaller chunks for efficient processing.

        Args:
            documents (list): List of Document objects to be split.

        Returns:
            list: List of split Document objects.
        """
        self.logger.info("Starting to split documents")
        # Split documents into smaller chunks
        split_docs = self.text_splitter.split_documents(documents)
        self.logger.info(f"Documents split into {len(split_docs)} chunks")
        return split_docs

    
    def configure_vector_store(self):
        """
        Configures the vector store based on the type specified in the configuration.
        """
        if self.vectorstore_type == 'chroma':
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
            self.logger.info("Chroma vector store configured.")
        elif self.vectorstore_type == 'faiss':
            # Initialize FAISS index
            index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))

            # Initialize vector store
            self.vectorstore = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            self.logger.info("FAISS vector store configured.")
        else:
            raise NotImplementedError(f"Vector store type '{self.vectorstore_type}' is not implemented.")

    
    
    def initialize_text_splitter(self):
        """
        Initializes the text splitter based on the configuration parameters.
        """
        if self.text_splitter_type == 'semantic':
            # Initialize the embedding model for the text splitter
            if self.embedding_provider == 'openai':
                self.text_splitter_embedding_model = OpenAIEmbeddings(model=self.text_splitter_model_name)
            else:
                # Handle other providers
                raise NotImplementedError(f"Embedding provider '{self.embedding_provider}' is not implemented.")
            self.text_splitter = SemanticChunker(self.text_splitter_embedding_model)
        elif self.text_splitter_type == 'recursive':
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        elif self.text_splitter_type == 'spacy':
            self.text_splitter = SpacyTextSplitter(chunk_size=self.chunk_size)
        # Add other text splitters as needed
        else:
            raise NotImplementedError(f"Text splitter type '{self.text_splitter_type}' is not implemented.")
        self.logger.info("Text splitter initialized")
    
    
    def save(self, documents):
        self.logger.info("Starting to save embeddings")
        if self.vectorstore_type == 'chroma':
            self.vectorstore = Chroma.from_documents(
                documents, self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.vectorstore_filename
            )
            self.logger.info("Embeddings saved to Chroma vector store")
        elif self.vectorstore_type == 'faiss':
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
            self.vectorstore.save_local(self.persist_directory+"/"+self.vectorstore_filename)
            self.logger.info("Embeddings saved to FAISS vector store")
            

    def display_and_save_stats(self, is_save=False):
        """
        Displays extraction statistics and saves them to a file.
        """
        self.logger.info("Displaying and saving extraction statistics")
        with open(self.stats_filename, "w") as f:
            for key, value in self.stats.items():
                log_message = f"{key.replace('_', ' ').title()}: {value}"
                self.logger.info(log_message)
                if is_save:
                    f.write(f"{log_message}\n")

    def initialize_vectorstore(self):
        """
        Method to automatically load PDFs, process them, split into chunks, and save embeddings into a vector store.
        """
        # Load PDFs and get stats
        documents = self.load_pdfs()

        # Save stats to a file
        self.display_and_save_stats()

        # Split documents into chunks
        split_docs = self.split_documents(documents)

        # Save embeddings to vector store
        self.save(split_docs)
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.top_k_docs, "score_threshold": self.score_threshold},
        )

    def load_vectorstore(self):
        """
        Method to load the saved vectorstore from the disk.

        Returns:
            vectorstore: The loaded vectorstore.
        """
        try:
            # Load the Chroma vector store from the disk
            if self.vectorstore_type == 'chroma':
                self.vectorstore = Chroma(
                    collection_name=self.vectorstore_filename,
                    persist_directory=self.persist_directory, 
                    embedding_function=self.embedding_model
                )
            elif self.vectorstore_type == 'faiss':
                self.vectorstore = FAISS.load_local(self.persist_directory+"/"+self.vectorstore_filename, 
                                                    self.embedding_model, 
                                                    allow_dangerous_deserialization=True)  # Set this to True)
            
            if self.search_type == "similarity":
                self.logger.info("Using similarity search")
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={"k": self.top_k_docs},
                )
            elif self.search_type == "mmr":
                self.logger.info("Using MMR search")
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={'k': self.top_k_docs, 'fetch_k': 50, 'lambda_mult': 0.25}
                )
            elif self.search_type == "similarity_score_threshold":
                self.logger.info("Using similarity search with score threshold")
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={'score_threshold': self.score_threshold}
                )
            else:
                self.logger.info("Using default similarity search") 
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={"k": self.top_k_docs},
                )
        
            self.logger.info("Vectorstore loaded from disk")

            return self.vectorstore
        except Exception as e:
            self.logger.error(f"Error loading vectorstore: {e}")
            return None
    

    def search(self, query):
        """
        Method to search the vector store for the most similar documents to a given query.

        Args:
            query (str): The query to search for.

        Returns:
            list: A list of the most relevant documents.
        """
        return self.retriever.invoke(query)




    


