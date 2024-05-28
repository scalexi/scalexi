from scalexi.openai.pricing import OpenAIPricing
from scalexi.utilities.logger import Logger
import re
from collections import Counter
import pdfplumber
import tiktoken
import json
from scalexi.openai.utilities import get_openai_pricing_info, get_context_length
import pkgutil

# Create a logger file
logger = Logger().get_logger()



def test_get_context_length():
    """
    Test function to retrieve and print the context length for each model in the OpenAI pricing JSON.
    """
    # Retrieve the full pricing information
    pricing_info = get_openai_pricing_info()

    # Extract all model names from the pricing info
    all_models = []
    for category in pricing_info['pricing']['language_models'].values():
        for model in category['models'].keys():
            all_models.append(model)

    # Test get_context_length for each model and print the results
    for model_name in all_models:
        context_length = get_context_length(model_name)
        print(f"Model: {model_name}, Context Length: {context_length}")

# Call the test function
test_get_context_length()

