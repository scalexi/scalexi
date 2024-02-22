import os
import json
import pkgutil
from langchain_community.document_loaders import PyPDFLoader
from scalexi.llm.openai_gpt import GPT
from scalexi.llm.google_gemini import Gemini
from scalexi.openai.pricing import OpenAIPricing

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        
    def load_pdf(self):
        pages = self.loader.load_and_split()
        all_pages_text = [document.page_content for document in pages]
        return "\n".join(all_pages_text)


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
