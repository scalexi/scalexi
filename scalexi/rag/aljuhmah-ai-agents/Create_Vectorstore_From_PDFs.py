import os
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv  # type: ignore
import openai

# Load environment variables, including API keys
load_dotenv()
# Access the API key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI API client
openai.api_key = api_key

# Function to load all PDFs from a directory and its subdirectories
def load_pdfs_from_directory(directory_path):
    stats = {
        "total_documents": 0,
        "successful_documents": 0,
        "failed_documents": 0,
        "total_pages": 0,
        "successful_pages": 0,
        "failed_pages": 0,
    }

    documents = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".pdf"):
                stats["total_documents"] += 1  # Increment document count
                pdf_path = os.path.join(root, filename)
                try:
                    reader = PdfReader(pdf_path)
                    num_pages = len(reader.pages)
                    stats["total_pages"] += num_pages  # Count total pages

                    document_text = []
                    for page_num, page in enumerate(reader.pages):
                        try:
                            text = page.extract_text()  # Extract text from each page
                            if text:
                                print(f"Extracted text from {filename}, page {page_num}")
                                document_text.append(text)
                                stats["successful_pages"] += 1  # Count successfully processed pages
                        except ValueError as ve:
                            print(f"Skipping malformed data on page {page_num} of {filename}: {ve}")
                            stats["failed_pages"] += 1  # Count failed pages

                    if document_text:
                        # Convert the document text to a single string and wrap it in a Document object
                        documents.append(Document(page_content="\n".join(document_text)))
                        stats["successful_documents"] += 1  # Count successful document
                    else:
                        stats["failed_documents"] += 1  # If no text was extracted from the document
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    stats["failed_documents"] += 1

    return documents, stats

# Function to split documents into manageable chunks for embeddings
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)  # Pass documents list directly
    print(f"Total chunks created: {len(split_docs)}")
    return split_docs

# Function to display statistics
def display_stats(stats):
    print("\n--- PDF Extraction Statistics ---")
    print(f"Total Documents Processed: {stats['total_documents']}")
    print(f"Successful Documents: {stats['successful_documents']}")
    print(f"Failed Documents: {stats['failed_documents']}")
    print(f"Total Pages Processed: {stats['total_pages']}")
    print(f"Successful Pages: {stats['successful_pages']}")
    print(f"Failed Pages: {stats['failed_pages']}")
    print("----------------------------------\n")

# Function to save stats to a .txt file
def save_stats_to_file(stats, file_path):
    with open(file_path, "w") as f:
        f.write("--- PDF Extraction Statistics ---\n")
        f.write(f"Total Documents Processed: {stats['total_documents']}\n")
        f.write(f"Successful Documents: {stats['successful_documents']}\n")
        f.write(f"Failed Documents: {stats['failed_documents']}\n")
        f.write(f"Total Pages Processed: {stats['total_pages']}\n")
        f.write(f"Successful Pages: {stats['successful_pages']}\n")
        f.write(f"Failed Pages: {stats['failed_pages']}\n")
        f.write("----------------------------------\n")

# Start the process
print("Loading Data Start")

# Load and prepare documents
directory_path = "./Data"  # Set your directory path here
documents, stats = load_pdfs_from_directory(directory_path)

# Display stats for documents and pages processed
display_stats(stats)

print("Loading Data Finished")
print("Chunking Start")

# Split documents into smaller chunks
split_docs = split_documents(documents)

print("Chunking Finished")
print(f"Number of chunks created: {len(split_docs)}")

# Create embeddings using OpenAI
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Adjust this to match the correct model

# Option 1: Save and load with Chroma Vector Store
def save_embeddings_chroma(documents, embeddings, persist_directory):
    chroma_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    chroma_store.persist()
    return chroma_store

def load_embeddings_chroma(persist_directory, embeddings):
    chroma_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return chroma_store

# Save the embeddings to Chroma
persist_directory = "./Chroma_vectorstore"
chroma_store = save_embeddings_chroma(split_docs, embedding_model, persist_directory)

print("Embeddings saved to Chroma vector store")

# Save stats to a file
stats_file_path = "pdf_extraction_stats.txt"  # Path where the stats will be saved
save_stats_to_file(stats, stats_file_path)

# Output message for the user
print(f"Statistics saved to {stats_file_path}")
print("Process Completed Successfully")
