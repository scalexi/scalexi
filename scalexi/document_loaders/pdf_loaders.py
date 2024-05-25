import os
import json
import pkgutil
from langchain_community.document_loaders import PyPDFLoader
from scalexi.llm.openai_gpt import GPT
from scalexi.llm.google_gemini import Gemini
from scalexi.openai.pricing import OpenAIPricing
import PyPDF2

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        
    def load_pdf(self):
        pages = self.loader.load_and_split()
        all_pages_text = [document.page_content for document in pages]
        return "\n".join(all_pages_text)
    
    def is_pdf_readable2(self):
        try:
            with open(self.pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                first_page_text = reader.pages[0].extract_text()
                if first_page_text and len(first_page_text.strip()) > 50:  # Check if there is substantial text
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return False
    
    def is_pdf_readable(self):
        try:
            # Attempt to load the first page
            pages = self.loader.load_and_split()
            if pages:
                first_page_text = pages[0].page_content
                # Check if there is substantial text (more than 50 characters)
                return len(first_page_text.strip()) > 50
            return False
        except Exception as e:
            print(f"Error checking PDF readability: {e}")
            return False


def main():
    # Example PDF path, replace 'example.pdf' with your actual PDF file path
    pdf_path = "pdf/TGRS.pdf"
    # Define the path for the output text file
    output_text_file = 'output_text.txt'
    
    pdf_loader = PDFLoader(pdf_path)
    
    # Load the PDF and get all pages text
    pdf_text = pdf_loader.load_pdf()
    
    
    
    # Save the extracted text to a text file
    with open(output_text_file, 'w', encoding='utf-8') as file:
        file.write(pdf_text)
    
    print(f"PDF text has been saved to {output_text_file}")

if __name__ == "__main__":
    main()
