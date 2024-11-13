import os
import json
import pkgutil
from langchain_community.document_loaders import PyPDFLoader
from scalexi.llm.google_gemini import Gemini
from scalexi.openai.pricing import OpenAIPricing
from scalexi.utilities.text_processing import classify_text_binary, remove_code_blocks, remove_code_block_markers, get_text_statistics
from scalexi.generators.generator import Generator
from scalexi.utilities.data_formatter import DataFormatter
import re
from scalexi.utilities.logger import Logger
import PyPDF2
import pdfplumber
from scalexi.openai.pricing import OpenAIPricing
from scalexi.utilities.logger import Logger
from collections import Counter
import pkgutil
from scalexi.openai.utilities import estimate_inference_cost
import time
import fitz  # PyMuPDF

# Create a logger file
logger = Logger().get_logger()

data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
pricing_info = json.loads(data)
#print(dfm.json_to_yaml(pricing_info))
pricing = OpenAIPricing(pricing_info)

class PDFLoader:
    """
    A class for loading, splitting, and extracting information from PDF files.
    
    Attributes:
        pdf_path (str): Path to the PDF file.
        model_name (str): Name of the model used for information extraction.
        loader (PyPDFLoader): Instance of PyPDFLoader for loading PDF.
        llm (Generator): Instance of Generator for making LLM requests.
        system_prompt: system prompts.
        logger (Logger): Logger for logging information.
    """

    def __init__(self, pdf_path=None, model_name=None, loader_type = "pdfplumber", openai_key=None, system_prompt = None, 
                replace_unicode=True):
        """
        Initializes the PDFLoader with a path to the PDF and an optional model name.
        
        Parameters:
            pdf_path (str): Path to the PDF file to be loaded.
            model_name (str): Optional; name of the model for extraction. Default is 'gpt-4o'.
        """
        t0= time.time()
        logger.info('[PDFLoader] Initializing PDFLoader.')
        #EnvironmentConfigLoader().load_environment_config()#must be delcared before logger
        self.logger = Logger().get_logger()
        if pdf_path is not None:
            self.pdf_path = pdf_path
        if openai_key is not None:
            self.llm = Generator(openai_key=openai_key)
        else:
            logger.warning("OpenAI key not provided. Using default key.")
            self.llm = Generator()
        if model_name is not None:
            self.model_name = model_name
            logger.info(f"Model Name set : {model_name} in constructor of PDFLoader")
            
        if system_prompt is None:
            self.system_prompt = "Extract the information from the PDF and return a JSON structured string"
        else:
            self.system_prompt = system_prompt
        #EnvironmentConfigLoader().display_all_env_variables()
        t1= time.time()
        self.loader_type = loader_type
        self.text = self.load_pdf(loader_type=loader_type, replace_unicode = replace_unicode)
        self.num_tokens = None
        if self.text is not None:
            self.pdf_loding_execution_time = time.time()-t1
            logger.info(f"[PDFLoader] PDF text extracted in {self.pdf_loding_execution_time} seconds")
            #self.logger.info('[PDFLoader] Initialized with model: %s', model_name)
            t1= time.time()
            self.num_tokens = pricing.calculate_token_usage_for_text(self.text, model_name)
            self.calculate_pricing_execution_time = time.time()-t1
            logger.info(f"[PDFLoader] Token usage calculated in {self.calculate_pricing_execution_time} seconds")
            self.total_execution_time = time.time()-t0
            logger.info(f"[PDFLoader] Completed initialization with {model_name} in {self.total_execution_time} seconds")
            #self.stats = self.get_stats(model_name = model_name)
        
        #exit()


    def load_pdf(self, pdf_path=None, loader_type = "pdfplumber", replace_unicode=True):
        
        """
        Loads the PDF file and extracts text from it.
        
        Returns:
            str: The extracted text from the PDF.
        """
        if pdf_path is not None:
            self.pdf_path = pdf_path
        if self.pdf_path is None:
            self.logger.error('[PDFLoader] PDF path is not provided. Please provide a valid PDF path.')
            raise ValueError("[PDFLoader] PDF path is not provided. Please provide a valid PDF path.")
        self.logger.info('[PDFLoader] Loading PDF.')
        try:
            if loader_type.lower() == "pdfplumber":
                text = self.extract_text_pdfplumber(replace_unicode = replace_unicode).replace('“', '').replace('”', '').replace('"', '').replace('’', "'")
            elif loader_type.lower() == "fitz":
                text = self.extract_text_with_fitz(replace_unicode = replace_unicode).replace('“', '').replace('”', '').replace('"', '').replace('’', "'")
            elif loader_type.lower() == "pypdf2":
                text = self.extract_text_from_pdf_with_PyPDF2(replace_unicode = replace_unicode).replace('“', '').replace('”', '').replace('"', '').replace('’', "'")
            elif loader_type.lower() == "pypdf":
                text = self.extract_text_with_PyPDFLoader(replace_unicode = replace_unicode).replace('“', '').replace('”', '').replace('"', '').replace('’', "'")
            else:
                text = self.extract_text_from_pdf(replace_unicode = replace_unicode).replace('“', '').replace('”', '').replace('"', '').replace('’', "'")
            
            if text and len(text) > 50: # Check if there is substantial text
                return text
            else:
                self.logger.error('[PDFLoader-load_pdf] Text is not readable. Returning None')
                return None
        except Exception as e:
            self.logger.error('[PDFLoader-load_pdf] Failed to extract text: %s', str(e))
            return None
            #raise  ValueError("[PDFLoader] Failed to extract text from PDF. Upload a Valid PDF")
            
    def is_pdf_readable(self):
        try:
            # Attempt to load the first page
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load_and_split()
            if pages:
                first_page_text = pages[0].page_content
                # Check if there is substantial text (more than 50 characters)
                return len(first_page_text.strip()) > 50
            return False
        except Exception as e:
            print(f"Error checking PDF readability: {e}")
            return False
    
    

    def extract_text_with_fitz(self, replace_unicode=True):
        """
        Loads and extracts text from the PDF using PyMuPDF (fitz).
        
        Returns:
            str: The complete text extracted from all pages of the PDF.
        """
        self.logger.info('[PDFLoader] Extracting text using fitz (PyMuPDF).')
        
        # Check file extension
        if not self.pdf_path.lower().endswith('.pdf'):
            self.logger.error('[PDFLoader] The file is not a PDF.')
            return None
        
        try:
            # Open the PDF file
            document = fitz.open(self.pdf_path)
            all_pages_text = []
            
            # Iterate through all the pages
            for page_num in range(len(document)):
                page = document.load_page(page_num)  # Load each page
                text = page.get_text()  # Extract text from the page
                all_pages_text.append(text)
            
            self.logger.info('[PDFLoader] PDF Loaded and text extracted from all pages.')
            return "\n".join(all_pages_text)
        
        except Exception as e:
            self.logger.error(f'[PDFLoader] Error extracting text using fitz: {str(e)}')
            return None

    
    
    def extract_text_with_PyPDFLoader(self, replace_unicode=True):
        """
        Loads and splits the PDF into pages.
        
        Returns:
            str: The complete text extracted from all pages of the PDF.
        """
        self.logger.info('[PDFLoader] Extracting text using PyPDFLoader.')
        # Check file extension
        if not self.pdf_path.lower().endswith('.pdf'):
            return None
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load_and_split()
        all_pages_text = [document.page_content for document in pages]
        self.logger.info('[PDFLoader] PDF Loaded and split into pages.')
        return "\n".join(all_pages_text)
    
    def clean_text(self, text, replace_unicode=True):
        """
        Cleans the text by replacing special characters with their correct counterparts.
        
        Parameters:
            text (str): The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        self.logger.debug('[PDFLoader] Cleaning text.')
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€”': '-', 'â€“': '-', 'â€¢': '*', 'â€¢': '*', 'â€¢': '*','“': '', '”': '', '’': "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Replace non-ASCII characters with '?'
        if replace_unicode:
            text = text.encode('ascii', 'replace').decode('ascii')
        
        return text
    
    def extract_text_from_pdf_with_PyPDF2(self,replace_unicode=True):
        self.logger.info('[PDFLoader] Extracting text using PyPDF2.')
        all_text = ""
        if not self.pdf_path.lower().endswith('.pdf'):
            return None
        try:
            with open(self.pdf_path , 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""  # Handle None
                    all_text += self.clean_text(page_text, replace_unicode = replace_unicode) + "\n"
        
        except FileNotFoundError:
            print("[scalexi-pdf-loader] PDF file not found.")
            all_text = None
        except IOError as io_error:
            all_text = None
            print(f"[scalexi-pdf-loader] An I/O error occurred: {str(io_error)}")
        except PyPDF2.PdfReadError as pdf_error:
            all_text = None
            print(f"[scalexi-pdf-loader] An error occurred while reading the PDF file: {str(pdf_error)}")
        except Exception as e:
            all_text = None
            print(f"[scalexi-pdf-loader] An unexpected error occurred: {str(e)}")
        except Exception as e:
            print(f"Failed to read PDF with PyPDF2: {e}")
            all_text = None
        return all_text

    def is_spacing_anomalous(self, text, max_expected_ratio=0.2):
        """
        Check if the spacing in the text is anomalous.
        
        Parameters:
            text (str): The text to check.
            max_expected_ratio (float): Maximum expected ratio of spaces to total characters.
            
        Returns:
            bool: True if the spacing is anomalous, False otherwise.
        """
        self.logger.info('[PDFLoader] Checking spacing in text.')
        if not text:
            return True
        total_chars = len(text)
        space_count = text.count(' ')
        space_ratio = space_count / total_chars
        return space_ratio > max_expected_ratio

    def extract_text_pdfplumber(self, replace_unicode=True):
        """
        Extracts text from the PDF using pdfplumber.
        
        Returns:
            str: The extracted text.
        """
        self.logger.info('[PDFLoader] Extracting text using pdfplumber.')
        text = ''
        if not self.pdf_path.lower().endswith('.pdf'):
            return None
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += self.clean_text(page_text, replace_unicode = replace_unicode) + "\n"
        except Exception as e:
            print(f"Failed to read PDF with pdfplumber: {e}")
            text = None
        return text

            
    def extract_text_from_pdf(self, replace_unicode=True):
        """
        Extracts text from the PDF using PyPDF2 and pdfplumber.
        
        Returns:
            str: The extracted text.
        """
        self.logger.info('[PDFLoader] Extracting text from PDF.')
        
        text = self.extract_text_from_pdf_with_PyPDF2(replace_unicode = replace_unicode)
        if text and len(text) > 50 and not any(ord(char) > 128 for char in text) and not self.is_spacing_anomalous(text):
            return text
        else:
            text = self.extract_text_pdfplumber(replace_unicode = replace_unicode)
            if text and len(text) >= 50: # Check if there is substantial text
                return text
            else:
                self.logger.error("Text is not readable after trying with pdfplumber. Returning None")
                raise ValueError("Text is not readable after trying with pdfplumber and PyPDF2. Upload a Valid PDF")

    
    def extract_information(self, text:str, 
                            stream=False, 
                            max_tokens=4096, 
                            system_prompt = None, 
                            model_name=None):
        """
        Extracts information from the given text using a specific prompt ID.
        
        Parameters:
            complete_text (str): The text to extract information from.
        
        Returns:
            tuple: Extracted information, price, token usage, and execution time.
        
        Raises:
            ValueError: If the prompt ID does not exist in the templates.
            Exception: For other errors that may occur during processing.
        """
        
        if model_name is not None:
            self.model_name = model_name
            logger.info(f"Model Name changed to: {model_name}")
        
        self.logger.info('[extract_information] Extracting information.')
        
        if self.model_name is None:
            self.logger.error("[PDFLoader:extract_information] LLM Model Name is None. Error in extracting information")
            raise ValueError("[PDFLoader:extract_information] Model Name is None. Using default model name")
        
        if system_prompt is None:
            if self.system_prompt is not None:
                system_prompt = self.system_prompt
            else:
                logger.error("System Prompt is None. Error in extracting information")
                raise ValueError("System Prompt is None. Using default system prompt")

        try:
            logger.debug(f"System Prompt:\n{system_prompt}")
            response, price, token_usage, execution_time = self.llm.ask_llm(
                model_name=self.model_name,
                max_tokens=max_tokens,
                temperature=0.1,
                system_prompt=system_prompt,
                user_prompt=text,
                stream=stream
            )
            clean_response = remove_code_block_markers(response)

            self.logger.debug('[extract_information] Response: %s', clean_response)
            self.logger.debug('[extract_information] Token usage: %s', token_usage)
            
            return clean_response, price, token_usage, execution_time

        except Exception as e:
            self.logger.error('[extract_information] Failed to extract information: %s', str(e))
            #raise e
        
    def get_stats(self, model_name="gpt-4"):
        """
        Provides descriptive statistics about the extracted text from a PDF.

        :param pdf_path: The path to the PDF file to analyze.
        :param model_name: The name of the model to use for token calculation.

        :return: A dictionary containing descriptive statistics about the text.
        """
        # Initialize variables for collecting text and word counts per page
        all_text = ""
        words_per_page = []
        num_pages = 0
        

        #with pdfplumber.open(self.pdf_path) as pdf:

                
        # Open the PDF with fitz
        with fitz.open(self.pdf_path) as pdf:
            num_pages = len(pdf)
            for page in pdf:
                page_text = page.get_text()
                all_text += page_text
                
                # Count words per page
                words = re.findall(r'\w+', page_text)
                words_per_page.append(len(words))

        # Number of characters
        num_chars = len(all_text)
        
        # Number of words
        words = re.findall(r'\w+', all_text)
        num_words = len(words)
        
        # Number of sentences
        sentences = re.split(r'[.!?]+', all_text)
        num_sentences = len(sentences) - 1  # Adjust for possible trailing empty string
        
        # Most common words
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(10)
        #print('most_common_words:', most_common_words)
        # Token count
        num_tokens = pricing.calculate_token_usage_for_text(all_text, model_name)
        
        # Get file size in bytes
        try:
            file_size = os.path.getsize(self.pdf_path)
        except Exception as e:
            logger.error(f"An error occurred while getting file size: {e}")
            file_size = None
        
        # Generate statistics dictionary
        stats = {
            "num_chars": num_chars,
            "num_words": num_words,
            "num_sentences": num_sentences,
            "most_common_words": str(most_common_words),
            "num_tokens": num_tokens,
            "num_pages": num_pages,
            "words_per_page": str(words_per_page),
            "file_size": file_size,
            "file_path": self.pdf_path,
            "model_name": model_name
        }   
        
        
        return stats
    
    
    def get_first_page(self):
        try:
            # Open the PDF file using pdfplumber
            with pdfplumber.open(self.pdf_path) as pdf:
                
                # Check if the PDF has pages
                if len(pdf.pages) < 1:
                    return {"success": False, "message": "The PDF file is empty."}
                
                # Get the first page
                first_page = pdf.pages[0]
                
                # Extract text from the first page
                first_page_text = first_page.extract_text()
                
                # Return the text of the first page
                return {"success": True, "page_text": first_page_text}
        
        except Exception as e:
            logger.error(f"An error occurred while extracting the first page: {e}")
            raise ValueError("Failed to extract the first page. Upload a Valid PDF")
            
    
    def calculate_pricing(self, token_usage):
        """
        Calculates the pricing based on token usage.
        
        Parameters:
            token_usage (dict): A dictionary containing 'prompt_tokens' and 'completion_tokens'.
        
        Returns:
            float: Estimated cost based on token usage.
        """
        self.logger.info('[PDFLoader] calculate_pricing')
        try:
            
            return estimate_inference_cost(
                token_usage,
                self.model_name
            )
        except Exception as e:
            self.logger.error('An error occurred while calculating pricing: %s', str(e))
            raise ValueError("[calculate_pricing] Failed to calculate pricing. Upload a Valid PDF")

    def structure_document(self, system_prompt=None, 
                           stream=False, 
                           max_tokens=4096, 
                           model_name=None):
        """
        Structures the document based on a given prompt ID. This method
        loads the PDF, extracts information using a specified prompt, calculates
        pricing based on token usage, and logs the entire process.

        Parameters:
            system_prompt (str): The system prompt to use for structuring the document.
            stream (bool): Whether to stream the response or not.
            max_tokens (int): The maximum number of tokens to use for the operation.

        Returns:
            tuple: A tuple containing the structured document response, the price
                for the operation, token usage, and execution time. In case of an
                error, returns None for the response and zeros for all numerical values.
        """
        if model_name is not None:
            self.model_name = model_name
            logger.info(f"Model Name changed to: {model_name}")
        self.logger.info('[PDFLoader] structure_document')
        try:
            complete_text = self.load_pdf(loader_type=self.loader_type).replace("%", " percent ").replace('"', '')   # Load the PDF and extract text
            complete_text = complete_text.replace("%", " percent ").replace('"', '')
            if complete_text is None:
                self.logger.error("[structure_document] Failed to load the PDF. Upload a Valid PDF")
                raise ValueError("[structure_document] Failed to load the PDF. Upload a Valid PDF")
            self.logger.debug('complete_text: %s', complete_text)
            if system_prompt is None:
                system_prompt = self.system_prompt
            response, price, token_usage, execution_time = self.extract_information(complete_text, 
                                                                                    system_prompt=system_prompt, 
                                                                                    stream=stream, 
                                                                                    max_tokens=max_tokens, model_name = self.model_name)
            #print('response:', response)
            price = self.calculate_pricing(token_usage)
            return response, price, token_usage, execution_time
        except Exception as e:
            self.logger.error('An error occurred while structuring the document: %s', str(e))
            raise ValueError("[structure_document] Failed to structure the document. Upload a Valid PDF")

    def extract_json_from_text(self, text):
        """
        Extracts a JSON object from the given text. This method searches the text
        for a JSON structure, parses it, and returns the JSON object as a string
        with proper formatting. Logs the process and errors if any.

        Parameters:
            text (str): The text from which to extract a JSON object.

        Returns:
            str: A string representation of the JSON object if found, otherwise None.
        """
        self.logger.info('[PDFLoader] extract_json_from_text')
        try:
            start_index = text.index('{')
            end_index = text.rindex('}')
            json_str = text[start_index:end_index + 1]
            json_object = json.loads(json_str)
            json_string = json.dumps(json_object, indent=4)
            self.logger.debug('json_string: %s', json_string)
            return json_string
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.error('An error occurred while extracting JSON from text: %s', str(e))
            raise ValueError("[extract_json_from_text] Failed to extract JSON from text. Upload a Valid PDF")



############################################################################################################
### CV Extractor: Extracts information from CVs using a specific extraction prompt. ########################
############################################################################################################

class CVExtractor(PDFLoader):
    """
    A class derived from PDFLoader to specialize in extracting information
    from CVs. It adds functionality to handle different types of CVs by
    setting a specific extraction prompt based on the CV type.

    Attributes:
        cv_type (str): The type of CV for which the class will extract information.
        extraction_prompt (str): The system prompt used for extraction, derived from the CV type.
    """

    def __init__(self, pdf_path, cv_type: str = "Software Engineer", model_name="gpt-4-turbo-preview", loader_type = "pdfplumber", system_prompt = None):
        """
        Initializes the CVExtractor with a path to the CV, the CV type, and an
        optional model name.

        Parameters:
            pdf_path (str): Path to the PDF file (CV) to be loaded.
            cv_type (str): Type of CV to set the specific extraction prompt.
            model_name (str): Optional; name of the model for extraction. Default is 'gpt-4-turbo-preview'.
        """
        super().__init__(pdf_path, model_name ,loader_type=loader_type, system_prompt=system_prompt)
        self.set_cv_type(cv_type)
        self.logger.info('[CVExtractor] Initialized with CV type: %s', cv_type)

    def set_cv_type(self, cv_type):
        """
        Sets the CV type and adjusts the system prompt for extraction based on the CV type.

        Parameters:
            cv_type (str): The type of CV for which information will be extracted.
        
        Raises:
            ValueError: If an invalid CV type is provided.
        """
        self.logger.info('[CVExtractor] set_cv_type to %s', cv_type)
        try:
            self.cv_type = cv_type
            #self.extraction_prompt = """extract the CV from the PDF"""
        except Exception as e:
            self.logger.error('An error occurred while setting the CV type: %s', str(e))
            raise ValueError("Failed to set CV type. Upload a Valid PDF")

    def extract_cv_information(self, text=None, system_prompt=None, stream=False, max_tokens=4096):
        """
        Extracts information from the CV using the specified extraction prompt.

        Returns:
            tuple: A tuple containing the extracted CV information, the price
                   for the operation, token usage, and execution time. In case of an
                   error, returns None for the response and zeros for all numerical values.
        """
        
        self.logger.info('[extract_cv_information] extract_cv_information')
        try:
            
            
            if text is None:
                text = self.text
            
            return super().extract_information(text, max_tokens=max_tokens, stream=stream)

        except Exception as e:
            self.logger.error('[extract_cv_information] An error occurred while extracting CV information: %s', str(e))
            #raise ValueError("[extract_cv_information] Failed to extract CV information. Upload a Valid PDF")
            
            




# Assuming the necessary imports and class definitions are already in place
def test_cv_extraction():
    # Define the path to the CV PDF file
    pdf_path = "cv/cv1.pdf"
    # Initialize the JobOfferExtractor with the specified job offer type
    #model_name="gpt-4-turbo-preview"
    #model_name="gemini-1.0-pro"
    #model_name="command"
    #model_name="gpt-3.5-turbo"
    model_name="gpt-4o"
    
    # Initialize the CVExtractor with the specified CV type
    cv_extractor = CVExtractor(pdf_path,  model_name=model_name, system_prompt = "Return a JSON string that Extract the CV from the PDF and return a JSON structured string")
    
    # Extract CV information
    try:
        structured_data, price, token_usage, execution_time = cv_extractor.extract_cv_information()
        print("Structured Document:", remove_code_blocks(structured_data))
        print("Cost of Extraction:", price)
        print("Token Usage:", token_usage)
        print("Execution Time:", execution_time)
        print("model_name:", cv_extractor.model_name)
    except Exception as e:
        print("An error occurred during CV extraction:", str(e))

import time
# Call the test functions
if __name__ == "__main__":
    test_cv_extraction()



   



