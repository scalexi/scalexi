from scalexi.openai.pricing import OpenAIPricing
from scalexi.utilities.logger import Logger
import json
import pkgutil
import tiktoken

# Create a logger file
logger = Logger().get_logger()

data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
pricing_info = json.loads(data)
#print(dfm.json_to_yaml(pricing_info))
pricing = OpenAIPricing(pricing_info)

#print('princing_info:', pricing_info)


def get_data():
    """
    Returns the pricing information for OpenAI models.

    :return: A dictionary containing the pricing information for OpenAI models.
    """
    data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
    pricing_info = json.loads(data)
    return pricing_info

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


def get_context_length(model_name):
    """
    Returns the context length of the specified model using the pricing info from OpenAI.

    :param model_name: The name of the model whose context length is needed.
    :return: The context length of the model if found, otherwise returns None.
    """
    # Retrieve the pricing information from the pre-defined method
    pricing_info = get_openai_pricing_info()

    # Traverse the JSON structure to find the context length
    for category in pricing_info['pricing']['language_models'].values():
        for model, details in category['models'].items():
            if model == model_name:
                return details.get('context_length', None)

    # If the model is not found, return None
    return None

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

def get_language_model_pricing(model_name=None):
        """
        Retrieves pricing information for language models based on the provided model name.

        This method accesses the stored pricing data to return details for a specific language model or for all language models. 
        If a model name is specified, it fetches pricing for that particular model. 
        If no model name is given, it returns comprehensive pricing details for all language models.

        :param model_name: The name of a specific language model for which pricing information is required. 
                        If None, pricing information for all language models is returned.
        :type model_name: Optional[str]

        :return: A dictionary containing the pricing information. If a specific model name is provided, 
                it returns a dictionary with 'input' and 'output' costs. Otherwise, it returns a dictionary 
                with each model category as keys and their respective pricing information as values.
        :rtype: dict

        :raises ValueError: If a specified model_name is not found in the stored language models pricing data.

        :example:

        ::

            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_language_model_pricing('GPT-4')
            {'input': 0.03, 'output': 0.06}
        """
        #print(data)
        language_models = pricing_info["pricing"]["language_models"]
        #print('language_models:', language_models)

        if model_name:
            for category_name, category in language_models.items():
                #print('category_name:', category_name)
                #print('category:', category)
                
                if 'models' in category and model_name in category["models"]:
                    #print(f"Found {model_name} in {category_name}, pricing:", category["models"][model_name])
                    return category["models"][model_name]
                #else:
                    #print(f"{model_name} not found in {category_name}")
            
            raise ValueError(f"Model '{model_name}' not found in language models pricing.")
        else:
            all_models = {}
            for category in language_models.values():
                if 'models' in category:
                    all_models.update(category["models"])
            return all_models

 
def estimate_inference_cost_by_tokens(input_tokens: int, output_tokens: int, model_name: str = "gpt-3.5-turbo")-> float:
        """
        Estimates the cost of inference operations based on the number of input and output tokens, as well as the chosen model.

        :method estimate_inference_cost: Calculate the estimated cost for inference operations.
        :type estimate_inference_cost: method

        :param input_tokens: The number of tokens to be processed as input during the inference.
        :type input_tokens: int

        :param output_tokens: The number of tokens expected to be generated as output during the inference.
        :type output_tokens: int

        :param model_name: The name of the AI model used for the inference operation. Defaults to 'gpt-3.5-turbo' if not specified.
        :type model_name: str, optional

        :return: The calculated cost of inference for the specified number of input and output tokens with the given model.
        :rtype: float

        :raises ValueError: If there is no pricing information available for the specified model, thus hindering the cost estimation.

        :example:

        ::

            >>> estimate_inference_cost(100, 50, "gpt-3.5-turbo")
            # Assuming hypothetical cost rates, this will return a float representing the estimated cost for 100 input tokens and 50 output tokens using 'gpt-3.5-turbo'.
        """

        logger.debug(f"Starting Cost Estimation for {input_tokens} input tokens and {output_tokens} output tokens using {model_name}")
        try:
            input_cost_per_token = get_language_model_pricing(model_name)['input'] / 1000.0
            output_cost_per_token = get_language_model_pricing(model_name)['output'] / 1000.0
            logger.debug(f"Estimated cost for {input_tokens} input tokens and {output_tokens} output tokens using {model_name}: ${input_cost_per_token * input_tokens + output_cost_per_token * output_tokens:.2f}")
            return input_cost_per_token * input_tokens + output_cost_per_token * output_tokens
        except Exception as e:
            logger.error(f"[OpenAIPricing] Pricing information for model {model_name} not found. ERROR: {e}\n", exc_info=False)
            raise ValueError(f"Pricing information for model {model_name} not found.")

def estimate_inference_cost(token_usage: dict, model_name: str = "gpt-3.5-turbo-1106")-> float:
    """
    Estimates the cost of inference operations based on the token usage statistics and the chosen model.

    :method estimate_inference_cost: Calculate the estimated cost for inference operations.
    :type estimate_inference_cost: method

    :param token_usage: Dictionary containing token usage counts with keys 'prompt_tokens' and 'completion_tokens'.
    :type token_usage: dict

    :param model_name: The name of the AI model used for the inference operation. Defaults to 'gpt-3.5-turbo' if not specified.
    :type model_name: str, optional

    :return: The calculated cost of inference for the specified token usage with the given model.
    :rtype: float

    :raises ValueError: If there is no pricing information available for the specified model, thus hindering the cost estimation.

    :example:

    ::

        >>> token_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        >>> estimate_inference_cost(token_usage, "gpt-3.5-turbo")
        # This will return a float representing the estimated cost for 100 prompt tokens and 50 completion tokens using 'gpt-3.5-turbo'.
    """

    logger.debug(f"Starting Cost Estimation for {token_usage['prompt_tokens']} prompt tokens and {token_usage['completion_tokens']} completion tokens using {model_name}")
    try:
        input_cost_per_token = get_language_model_pricing(model_name)['input'] / 1000.0
        output_cost_per_token = get_language_model_pricing(model_name)['output'] / 1000.0
        total_cost = (input_cost_per_token * token_usage['prompt_tokens'] + output_cost_per_token * token_usage['completion_tokens'])
        logger.debug(f"Estimated cost for {token_usage['prompt_tokens']} prompt tokens and {token_usage['completion_tokens']} completion tokens using {model_name}: ${total_cost:.2f}")
        return total_cost
    except Exception as e:  
        error_message = f"Pricing information for model {model_name} not found. {e}"
        logger.error(f"[OpenAIPricing] {error_message}")
        raise ValueError(error_message)


    
def extract_response_and_token_usage_and_cost(response, model_name):
        """
        Extracts the content of the response, token usage, and estimates the inference cost.

        Parameters:
        - response (dict): The response message received.

        Returns:
        - tuple: A tuple containing the content of the response, the token usage dictionary, and the estimated cost.
        """
        # Extract the content and token usage
        content = response.choices[0].message.content
        print('content:', content)
        print('response.usage:', response.usage)
        token_usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
        #print(token_usage)
        
        # Estimate the cost
        print('estimate_inference_cost  token_usage:', token_usage)
        cost = estimate_inference_cost_by_tokens(token_usage['prompt_tokens'], token_usage['completion_tokens'], model_name)
        
        return content, token_usage, cost