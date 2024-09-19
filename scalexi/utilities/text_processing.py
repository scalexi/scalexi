import re
import PyPDF2
import time
import json
from openai import OpenAI
from scalexi.utilities.logger import Logger
from scalexi.openai.pricing import OpenAIPricing
import pkgutil
import os
import pdfplumber
from scalexi.openai.pricing import OpenAIPricing
from scalexi.utilities.logger import Logger
import re
from collections import Counter
import pdfplumber
import tiktoken

import pkgutil

# Create a logger file
logger = Logger().get_logger()

data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
pricing_info = json.loads(data)
#print(dfm.json_to_yaml(pricing_info))
pricing = OpenAIPricing(pricing_info)

context_lengths = {
                "gpt-4o": 128000,
                "gpt-4o-mini": 128000,
                "gpt-4o-2024-05-13": 128000,
                "gpt-4-turbo": 128000,
                "gpt-4-turbo-2024-04-09": 128000,
                "gpt-4-turbo-preview": 128000,
                "gpt-4-0125-preview": 128000,
                "gpt-4-1106-preview": 128000,
                "gpt-4-vision-preview": 128000,
                "gpt-4-1106-vision-preview": 128000,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-32k-0613": 32768,
                "gpt-3.5-turbo-0125": 16385,
                "gpt-3.5-turbo": 16385,
                "gpt-3.5-turbo-1106": 16385,
                "gpt-3.5-turbo-instruct": 4096,
                "gpt-3.5-turbo-16k": 16385,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-16k-0613": 16385
            }

def remove_latex_constructs(context: str) -> str:
    # Remove LaTeX comment lines
    context = re.sub(r'%.*', '', context)
    
    # Replace LaTeX citation commands with [Citation]
    context = re.sub(r'~\\cite\{.*?\}', '[Citation]', context)

    # Replace other LaTeX commands like \textbf, \textcolor, etc.
    context = re.sub(r'\\text[a-zA-Z]+\{.*?\}', '', context)
    
    # Replace LaTeX commands for sections and subsections
    context = re.sub(r'\\(sub)*section\{.*?\}', '', context)

    # Handle itemize environment
    context = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', context, flags=re.DOTALL)

    # Replace individual items in lists
    context = re.sub(r'\\item', '', context)
    
    # Remove any extra spaces
    context = ' '.join(context.split())

    # Add more preprocessing steps as required

    return context



def convert_latex_to_plain_text(context: str) -> str:
    # Remove LaTeX comment lines
    context = re.sub(r'%.*', '', context)
    
    # Remove LaTeX citation commands
    context = re.sub(r'~?\\cite\{.*?\}', '', context)

    # Remove LaTeX commands with arguments, e.g., \textbf{...}, \textcolor{red}{...}
    context = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', context)
    
    # Remove LaTeX commands without arguments, e.g., \noindent
    context = re.sub(r'\\[a-zA-Z]+', '', context)
    
    # Remove LaTeX environment declarations, e.g., \begin{itemize} ... \end{itemize}
    context = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', context, flags=re.DOTALL)

    # Remove individual items in lists
    context = re.sub(r'\\item', '', context)

    # Remove any extra spaces
    context = ' '.join(context.split())

    return context

import re

def extract_plain_text(context: str) -> str:
    # Remove LaTeX comment lines
    context = re.sub(r'%.*', '', context)
    
    # Replace LaTeX citation commands with [cite citation_ref]
    context = re.sub(r'~?\\cite\{(.*?)\}', r'[cite \1]', context)

    # Replace _{ with [ and }_ with ]
    context = re.sub(r'_{', '[', context)
    context = re.sub(r'}_', ']', context)
    
    # Remove inline math: $...$
    context = re.sub(r'\$.*?\$', '', context)
    
    # Remove LaTeX commands with arguments, e.g., \textbf{...}, \textcolor{red}{...}
    context = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', context)
    
    # Remove LaTeX commands without arguments, e.g., \noindent
    context = re.sub(r'\\[a-zA-Z]+', '', context)
    
    # Remove LaTeX environment declarations, e.g., \begin{itemize} ... \end{itemize}
    context = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '[itemize]', context, flags=re.DOTALL)

    # Remove individual items in lists
    context = re.sub(r'\\item', '[item]', context)

    # Remove special characters that might be problematic in langchain
    context = re.sub(r'[\{\}\\&]', '', context)

    # Clean up extra spaces without removing newlines
    context = '\n'.join([' '.join(line.split()) for line in context.split('\n')])

    return context

import re

def remove_code_blocks(text):
# This regex pattern specifically looks for the sequence ```json followed by any characters (including newlines),
    # and then the closing ```. It captures the content between ```json and the closing ``` to keep the JSON content.
    pattern = r'```json\n?(.*?)```'
    
    # The re.DOTALL flag allows the dot (.) to match newlines, and re.MULTILINE treats each line as a separate string.
    # re.sub is used with a lambda function that returns only the captured group (the JSON content) for each match,
    # effectively removing the ```json and ``` delimiters but keeping the JSON content.
    cleaned_text = re.sub(pattern, lambda match: match.group(1), text, flags=re.DOTALL | re.MULTILINE)
    return cleaned_text

def remove_code_block_markers(text):
    # This regex pattern looks for the start of a code block (``` followed by
    # an optional language identifier like json, html, xml, etc.), and the end of a code block (```)
    # It replaces only these markers with an empty string, leaving the content in between intact.
    # The pattern explicitly captures the start and end markers of code blocks and uses
    # non-capturing groups (?:...) for optional language identifiers.
    pattern = r'(```[a-zA-Z]*\n)|(```$)'
    # The sub function replaces these markers with an empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return cleaned_text

def is_pdf_readable(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            first_page_text = reader.pages[0].extract_text()
            if first_page_text and len(first_page_text.strip()) > 20:  # Check if there is substantial text
                return True
            else:
                return False
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False
    
def extract_gpt_token_usage(response):
        """
        Extracts the token usage from a ChatCompletion response object and returns it in a dictionary.

        :param response: The ChatCompletion response object.
        :return: A dictionary containing the number of tokens used for the prompt, completion, and total.
        """
        if hasattr(response, 'usage'):
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            token_usage = {"error": "No token usage information available"}

        return token_usage
    
def classify_text_binary(text, category_name, criteria="", model_name="gpt-3.5-turbo-1106", 
                         openai_api_key=None, 
                         temperature=0.0, 
                         max_tokens=128, 
                         top_p=1.0, 
                         frequency_penalty=0.0, 
                         presence_penalty=0.0):
    """
    Classifies text using a specified model from OpenAI and checks if it matches the given category,
    also calculates the execution time, cost, and token usage.
    
    Args:
    text (str): The text to classify.
    category_name (str): The expected category name to validate against.
    criteria: The criteria for the classification.
    model_name (str): Model identifier for OpenAI API.
    openai_api_key (str, optional): API key for OpenAI. If None, it fetches from environment variable.
    temperature (float): Sampling temperature.
    max_tokens (int): Maximum number of tokens to generate.
    top_p (float): Nucleus sampling parameter.
    frequency_penalty (float): Frequency penalty parameter.
    presence_penalty (float): Presence penalty parameter.
    
    Returns:
    dict: A dictionary with the classification results, execution time, cost, and token usage.
    """
    logger.info(f"[SCALEXI-classify_text_binary] Classifying text using model: {model_name}")
    # Define the system prompt
    system_prompt = f"""You are a precise and strict text classifier for the "{category_name}" category. 
Below are the specific criteria for the "{category_name}" category: {criteria}
Provide your classification response in a valid JSON structure format, ensuring no use of code markers (e.g., no ``` markers). The response format should include:
- "category": "{category_name}"
- "reason": "<str>" (explanation for the classification decision)
- "category_match": <bool> (True if the text clearly and explicitly belongs to the "{category_name}" category, otherwise False)
    """
    logger.debug(f"[SCALEXI-classify_text_binary] System prompt: {system_prompt}")
    
    # Define the prompt that contains the text
    prompt = f"Here is the text: {text}"
    
    # Measure the start time
    start_time = time.time()

    # Initialize your OpenAI client with your API key
    client = OpenAI(api_key=openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY"), max_retries=3)

    # Use the combined prompt in open chat completion
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    except Exception as e:
        print(f"Failed to get response from model: {e}")
        return {}


    # Extract the classification result
    logger.debug(f"[SCALEXI-classify_text_binary] Response: {response.choices[0].message.content}")
    try:
        logger.info(f"[SCALEXI-classify_text_binary] Decoding JSON from model response.")
        completion = json.loads(remove_code_block_markers(response.choices[0].message.content))
    except json.JSONDecodeError:
        logger.info(f"Failed to decode JSON from model response. Set completion without decoding.")
        logger.info(type(remove_code_block_markers(response.choices[0].message.content)))
        completion = remove_code_block_markers(response.choices[0].message.content)
        #return {}
    
    end_time = time.time()  # End time
    execution_time = end_time - start_time  # Execution time
    token_usage = pricing.extract_gpt_token_usage(response)
    price = pricing.estimate_inference_cost(token_usage['prompt_tokens'], token_usage['completion_tokens'], model_name)
    #print(response)
    logger.debug(f'[SCALEXI-classify_text_binary] Response: {response}')
    
    #return response.choices[0].message.content, price, token_usage, execution_time

    # Prepare the final result
    final_result = {
        "category": completion['category'],
        "category_match": completion['category_match'],
        "reason": completion['reason'],  # "reason": "The text contains the word 'sports' multiple times.
        "execution_time": execution_time,
        "price": price,
        "token_usage": token_usage, 
        "model_name": model_name
    }
    logger.info(f'[SCALEXI-classify_text_binary] returning response, pricing, token_usage, execution_time')
    return final_result


def classify_text(text, categories, model_name="gpt-3.5-turbo-1106", openai_api_key=None, 
                  temperature=0.0, max_tokens=128, top_p=1.0, 
                  frequency_penalty=0.0, presence_penalty=0.0):
    """
    Classifies text into a list of categories, returning a JSON structure of categories and their confidence scores.
    
    Args:
    text (str): The text to classify.
    categories (list): A list of category names to classify against.
    model_name (str): Model identifier for OpenAI API.
    openai_api_key (str, optional): API key for OpenAI. If None, it fetches from environment variable.
    temperature (float): Sampling temperature.
    max_tokens (int): Maximum number of tokens to generate.
    top_p (float): Nucleus sampling parameter.
    frequency_penalty (float): Frequency penalty parameter.
    presence_penalty (float): Presence penalty parameter.
    
    Returns:
    str: A JSON formatted string representing the categories sorted by confidence scores.
    """
    logger.info(f"[SCALEXI-classify_text] Classifying text using model: {model_name}")
    # Fetch API key if not provided
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key is not provided and not found in environment variables.")
            return json.dumps({"error": "API key not provided"})

    client = OpenAI(api_key=openai_api_key, max_retries=3)

    categories_json = """
    {
        "<category_name_1>": "confidence_level_1",
        "<category_name_2>": "confidence_level_2",
        ...
        "<category_name_n>": "confidence_level_n"
    }

    """
    system_prompt = f"""You are a text classifier. 
    Given the following text, classify it into the provided categories and r
    eturn a JSON structure of the list of categories with each category associated 
    with a confidence score. Categories: {categories}
    Never add any markers like ```json
    The JSON is strictly the following:
     """  + categories_json
    
    logger.debug(f"[SCALEXI-classify_text] System prompt: {system_prompt}")
    logger.info(f"[SCALEXI-classify_text] Starting classification process.")
    prompt = f"Classify the following text into categories: {text}"
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None
        )
        
                
        try:
            logger.info(f"[SCALEXI-classify_text] Decoding JSON from model response.")
            completion = json.loads(remove_code_block_markers(response.choices[0].message.content))
        except json.JSONDecodeError:
            logger.info(f"Failed to decode JSON from model response. Set completion without decoding.")
            logger.info(type(remove_code_block_markers(response.choices[0].message.content)))
            completion = remove_code_block_markers(response.choices[0].message.content)
        
        logger.info("Classification successful.")
        end_time = time.time()  # End time
        execution_time = end_time - start_time  # Execution time
        token_usage = pricing.extract_gpt_token_usage(response)
        price = pricing.estimate_inference_cost(token_usage['prompt_tokens'], token_usage['completion_tokens'], model_name)
        #print(response)
        logger.debug(f'[SCALEXI-classify_text] Response: {response}')

        # Prepare the final result
        final_result = {
            "confidence_scores": completion,
            "execution_time": execution_time,
            "price": price,
            "token_usage": token_usage, 
            "model_name": model_name
        }
        
        logger.info(f'[SCALEXI-classify_text] returning response, pricing, token_usage, execution_time')
        return final_result
    except Exception as e:
        logger.error(f"Error during text classification: {str(e)}")
        return json.dumps({"error": str(e)})

def get_text_statistics_basic(pdf_path, model_name="gpt-4"):
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

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
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
    
    # Generate statistics dictionary
    stats = {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "most_common_words": str(most_common_words),
        "num_tokens": num_tokens,
        "num_pages": num_pages,
        "words_per_page": str(words_per_page)
    }
    
    return stats


def get_openai_pricing():
    """
    Returns the pricing information for OpenAI models.

    :return: A dictionary containing the pricing information for OpenAI models.
    """
    data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
    pricing_info = json.loads(data)
    pricing = OpenAIPricing(pricing_info)
    return pricing
  
def get_openai_pricing_info():
    """
    Returns the pricing information for OpenAI models.

    :return: A dictionary containing the pricing information for OpenAI models.
    """
    data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
    pricing_info = json.loads(data)
    return pricing_info

def get_text_statistics_from_text(extracted_text, model_name="gpt-4"):
    """
    Provides descriptive statistics about the given text.

    :param extracted_text: The text extracted from a PDF.
    :param model_name: The name of the model to use for token calculation.

    :return: A dictionary containing descriptive statistics about the text.
    """
    # Initialize variables for collecting text and word counts per page
    all_text = extracted_text
    words_per_page = []

    # Split text into pages assuming each page has approximately the same number of characters
    # This is a simplification; you might need a more complex method to split pages
    words = re.findall(r'\w+', all_text)
    num_words = len(words)

    # Number of characters
    num_chars = len(all_text)
    
    # Number of sentences
    sentences = re.split(r'[.!?]+', all_text)
    num_sentences = len(sentences) - 1  # Adjust for possible trailing empty string
    
    # Most common words
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)
    #print('most_common_words:', most_common_words)

    # Token count
    num_tokens = calculate_token_usage_for_text(all_text, model_name)
    
    # Generate statistics dictionary
    stats = {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "most_common_words": str(most_common_words),
        "num_tokens": num_tokens,
        "num_pages": None,  # Page count is not applicable here
        "words_per_page": None,  # Words per page is not applicable here
        "file_size": None,  # File size is not applicable here
        "file_path": None,  # File path is not applicable here
        "model_name": model_name
    }   
    
    return stats
  

def get_text_statistics(pdf_path, model_name="gpt-4"):
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

        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
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
        print('most_common_words:', most_common_words)
        # Token count
        num_tokens = pricing.calculate_token_usage_for_text(all_text, model_name)
        
        # Get file size in bytes
        try:
            file_size = os.path.getsize(pdf_path)
        except Exception as e:
            logger.error(f"An error occurred while getting file size: {e}")
            stats["file_size"] = None
        
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
            "file_path": pdf_path,
            "model_name": model_name
        }   
        
        
        return stats
    
def can_model_process_text(text_statistics: dict, model_name: str) -> bool:
        """
        Determines if the text can be processed by the specified model based on the model's token capacity.

        Args:
            text_statistics (dict): A dictionary containing statistics about the text, including the number of tokens.
            context_length (dict): A dictionary containing the maximum token context length for the model.
            model_name (str): The name of the model to check for processing capacity.

        Returns:
            bool: True if the model can process the text within its token limit, False otherwise.
        """
        num_tokens = text_statistics.get("num_tokens", 0)
        max_tokens = context_lengths.get(model_name, 0)
        return num_tokens <= max_tokens
    
def calculate_token_usage_for_text(text, model="gpt-3.5-turbo-0613"):
    """
    Calculates the total number of tokens used by an input text, considering the specified model's tokenization scheme.

    :method calculate_token_usage_for_text: Determine the total token count for a given text based on the model's encoding.
    :type calculate_token_usage_for_text: method

    :param text: A string representing the input text.
    :type text: str

    :param model: Identifier of the model for estimating token count. Defaults to "gpt-3.5-turbo-0613".
    :type model: str, optional

    :return: The total token count for the provided text as encoded by the specified model.
    :rtype: int

    :raises KeyError: If the token encoding for the specified model is not found in the encoding data.
    :raises NotImplementedError: If the function does not support token counting for the given model.

    :example:

    ::

        >>> text = "Hello! How can I assist you today?"
        >>> calculate_token_usage_for_text(text)
        # Assuming the model 'gpt-3.5-turbo-0613', this returns the total token count for the text.
    """

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Token allocation per model
    tokens_allocation = {
        "gpt-3.5-turbo-0613": (3, 1),
        "gpt-3.5-turbo-16k-0613": (3, 1),
        "gpt-4-0314": (3, 1),
        "gpt-4-32k-0314": (3, 1),
        "gpt-4-0613": (3, 1),
        "gpt-4-32k-0613": (3, 1),
        "gpt-3.5-turbo-0301": (4, -1)  # every message follows {role/name}\n{content}\n
    }

    # Default tokens per message and name
    tokens_per_message, tokens_per_name = tokens_allocation.get(
        model, 
        (3, 1)  # Default values
    )

    # Handling specific model updates
    if "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Assuming gpt-3.5-turbo-0613.")
        tokens_per_message, tokens_per_name = tokens_allocation["gpt-3.5-turbo-0613"]
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Assuming gpt-4-0613.")
        tokens_per_message, tokens_per_name = tokens_allocation["gpt-4-0613"]
    else:
        raise NotImplementedError(
            f"Token counting not implemented for model {model}. "
            "See the OpenAI Python library documentation for details."
        )

    # Token counting
    num_tokens = tokens_per_message  # start with the base tokens per message
    num_tokens += len(encoding.encode(text))  # add the tokens for the text content

    return num_tokens


def remove_field_from_dict(data: dict, field: str) -> dict:
    """
    Removes a specified field from a dictionary, including nested dictionaries within lists.

    Parameters:
    data (dict): The dictionary from which the field should be removed.
    field (str): The field to remove from the dictionary.

    Returns:
    dict: The dictionary with the specified field removed.
    """
    if not isinstance(data, dict):
        raise ValueError("The data parameter must be a dictionary.")
    if not isinstance(field, str):
        raise ValueError("The field parameter must be a string.")

    def remove_field_from_dict_recursively(d: dict, field: str):
        if field in d:
            del d[field]
        for key, value in d.items():
            if isinstance(value, dict):
                remove_field_from_dict(value, field)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        remove_field_from_dict(item, field)

    remove_field_from_dict_recursively(data, field)
    return data