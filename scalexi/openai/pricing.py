import logging
import os 
import tiktoken
import pandas as pd
import json

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

class OpenAIPricing:

    """
    A class dedicated to handling and accessing OpenAI's pricing data for various models and services.

    This class simplifies the process of accessing detailed pricing information for OpenAI's diverse range of models, 
    including language models, assistants API, fine-tuning models, embedding models, base models, image models, and audio models.

    The pricing data is structured as follows:

        - Language Models: Pricing for various language models like GPT-4, GPT-3.5, etc., including input and output token costs.
        
        - Assistants API: Pricing for tools such as Code Interpreter and Retrieval, including cost per session and any special notes.
        
        - Fine Tuning Models: Costs associated with training and input/output usage for models like gpt-3.5-turbo and davinci-002.
        
        - Embedding Models: Usage costs for models such as ada v2.
        
        - Base Models: Token usage costs for base models like davinci-002 and babbage-002.
        
        - Image Models: Pricing for different resolutions and quality levels in image models like DALL·E 3 and DALL·E 2.
        
        - Audio Models: Usage costs for models like Whisper and Text-To-Speech (TTS) models.

    This information provides a comprehensive overview of the pricing structure, aiding users in making informed decisions based on their specific needs.

    :method __init__: Initialize the OpenAIPricing instance with JSON-formatted pricing data.
    :type __init__: constructor

    :param json_data: The JSON data containing the pricing information of OpenAI models and services.
    :type json_data: dict

    :return: An instance of OpenAIPricing with parsed pricing data.
    :rtype: OpenAIPricing

    :example:

    ::

        >>> json_data = {
                "release_date": "2023-11-15",
                "pricing": {
                    "language_models": {
                        "GPT-4 Turbo": {
                            "context": "128k context, fresher knowledge ...",
                            "models": {
                                "gpt-4-1106-preview": {"input": 0.01, "output": 0.03}
                            }
                        }
                    }
                }
            }
        >>> pricing = OpenAIPricing(json_data=json_data)
        >>> print(type(pricing))
        <class 'OpenAIPricing'>
    """

    def __init__(self, json_data: dict):
        """
        Initializes the OpenAIPricing with JSON data.

        Parameters:
        
            json_data (dict): A dictionary containing the pricing data.
        """
        self.data = json_data

    def get_release_date(self):
        """
        Retrieves the release date of the pricing data.

        Returns:
            str: The release date.
        """
        return self.data.get("release_date")

    def get_language_model_pricing(self, model_name=None):
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
        language_models = self.data["pricing"]["language_models"]
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
        
    def get_inference_pricing(self, model_name=None):
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
        return self.get_language_model_pricing(model_name)

    def get_assistants_api_pricing(self, tool_name=None):
        """
        Retrieves pricing information for the Assistants API tools from the stored pricing data.

        This method provides the ability to access pricing details for specific tools or all tools within the Assistants API category. 
        When a specific tool name is provided, it returns the pricing for that tool. If no tool name is specified, 
        the method returns pricing information for all tools under the Assistants API.

        Parameters:
            tool_name (str, optional): The name of the specific Assistants API tool for which pricing 
                                    information is required. If None, returns pricing for all tools.

        Returns:
            dict: A dictionary containing the pricing information. If a specific tool name is given, 
                the dictionary includes the tool's cost and any additional notes. If no tool name 
                is specified, it returns a dictionary with tool names as keys and their respective 
                pricing information as values.

        Raises:
            ValueError: If a specific tool_name is provided but not found in the Assistants API pricing data.

        Example:
            >>> pricing_data.get_assistants_api_pricing('Code interpreter')
            {'input': 0.03, 'note': 'Free until 11/17/2023'}
        """

        tools = self.data["pricing"]["assistants_api"]
        if tool_name:
            if tool_name in tools:
                return tools[tool_name]
            else:
                raise ValueError(f"Tool '{tool_name}' not found in Assistants API pricing.")
        else:
            return tools


    def get_fine_tuning_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for fine-tuning models from the stored pricing data.

        This method is designed to access the cost associated with training, input usage, and output usage for fine-tuning models.
        Users can specify a particular model to obtain its specific pricing details. If no model name is provided, 
        the method returns comprehensive pricing information for all available fine-tuning models.

        Parameters:
            model_name (str, optional): The name of the specific fine-tuning model for which pricing 
                                        information is desired. If None, pricing information for all 
                                        fine-tuning models is returned. The default value is None.

        Returns:
            dict: A dictionary containing the pricing information. If a specific model name is given, 
                the dictionary includes keys 'training', 'input_usage', and 'output_usage' with their 
                respective costs. If no model name is specified, it returns a dictionary with model 
                names as keys and their respective pricing details as values.

        Raises:
            ValueError: If the provided model_name does not exist in the fine-tuning models pricing data.

        Example:
            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_fine_tuning_model_pricing('gpt-3.5-turbo')
            {'training': 0.008, 'input_usage': 0.003, 'output_usage': 0.006}
        """

        models = self.data["pricing"]["fine_tuning_models"]
        if model_name:
            # Check if the model_name is in the models dictionary
            if model_name in models:
                logger.debug(f"Retrieved pricing for model '{model_name}': {models[model_name]}")
                return models[model_name]
            else:
                error_message = f"Model '{model_name}' not found in available fine-tuning models."
                logger.error(f"[OpenAIPricingData] {error_message}\nAvailable models: {models.keys()}")
                raise ValueError(f"[OpenAIPricingData] {error_message}")
        else:
            # If no model_name is provided, return all the models
            return models



    def get_embedding_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for embedding models from the stored pricing data.

        This method allows for fetching pricing details of specific embedding models or all embedding models, based on the provided model name.
        If a model name is specified, it returns the pricing for that particular embedding model. If no model name is given, 
        it returns pricing details for all embedding models in a comprehensive dictionary.

        Parameters:
            model_name (str, optional): The name of the specific embedding model for which pricing information is desired. 
                                        If None, pricing information for all embedding models is returned.

        Returns:
            dict: A dictionary containing the pricing information. If a specific model name is provided, 
                the dictionary includes the 'usage' cost for that model. If no model name is specified, 
                it returns a dictionary with model names as keys and their respective pricing information as values.

        Raises:
            ValueError: If a model_name is specified but not found in the embedding models pricing data.

        Example:
            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_embedding_model_pricing('ada v2')
            {'usage': 0.0001}
        """

        models = self.data["pricing"]["embedding_models"]
        if model_name:
            if model_name in models:
                return models[model_name]
            else:
                raise ValueError(f"Model '{model_name}' not found in embedding models pricing.")
        else:
            return models

    def get_base_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for base models from the stored pricing data.

        This method facilitates access to cost details associated with the use of base language models. Users can obtain pricing information 
        for a specific model if the model name is provided. If no model name is given, the method returns pricing details for all base models.

        Parameters:
            model_name (str, optional): The name of the specific base model for which pricing information is sought. 
                                        If None, the method returns pricing information for all base models.

        Returns:
            dict: A dictionary containing pricing details. If a specific model name is provided, 
                it returns a dictionary with details about the usage cost for that model. 
                If no model name is specified, it returns a dictionary with each model name as a key 
                and their respective pricing details as values.

        Raises:
            ValueError: If a model_name is provided but not found in the base models pricing data.

        Example:
            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_base_model_pricing('davinci-002')
            {'usage': 0.002}

            >>> pricing_data.get_base_model_pricing()
            {'davinci-002': {'usage': 0.002}, 'babbage-002': {'usage': 0.0004}}
        """

        models = self.data["pricing"]["base_models"]
        if model_name:
            if model_name in models:
                return models[model_name]
            else:
                raise ValueError(f"Model '{model_name}' not found in base models pricing.")
        else:
            return models

    def extract_response_and_token_usage(self, response):
        """
        Extracts the content of the response and token usage from the response message.

        Parameters:
        - response (dict): The response message received.

        Returns:
        - tuple: A tuple containing the content of the response and the token usage dictionary.
        """
        # Extract the content of the response
        content = response.choices[0].message.content
        
        # Extract the token usage
        token_usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return content, token_usage
    
    def extract_gpt_token_usage(self, response):
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


    def get_image_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for image models from the stored pricing data.

        This method is designed to provide users with pricing details for various image generation models offered by OpenAI. 
        If a specific model or category name is given, it returns the pricing specific to that model or category. 
        If no model name is specified, the method returns pricing information for all available image models.

        Parameters:
            model_name (str, optional): The name of the specific image model or category for which pricing information is sought. 
                                        If None, pricing information for all image models is returned.

        Returns:
            dict or None: A dictionary containing the pricing information for the requested image model(s). 
                        Each key in the dictionary represents a model name or category, with corresponding 
                        pricing details in a nested dictionary. Returns None if the requested model name 
                        does not exist in the pricing data.

        Raises:
            ValueError: If the specified model_name is not found in the image models pricing data.

        Example:
            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_image_model_pricing('DALL·E 3')
            {
                'Standard': {
                    '1024x1024': 0.04,
                    '1024x1792_1792x1024': 0.08
                },
                'HD': {
                    '1024x1024': 0.08,
                    '1024x1792_1792x1024': 0.12
                }
            }

            >>> pricing_data.get_image_model_pricing()
            {
                'DALL·E 3': { ... },
                'DALL·E 2': { ... }
            }
        """

        models = self.data["pricing"]["image_models"]
        if model_name:
            model_pricing = models.get(model_name)
            if model_pricing:
                return model_pricing
            else:
                raise ValueError(f"Image model '{model_name}' not found in the pricing data.")
        else:
            return models


    def get_audio_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for audio models from the stored pricing data.

        This method is intended to provide detailed pricing information for audio-related models, including services like transcription 
        and text-to-speech. Users can specify a particular audio model to get its pricing details. If no model name is provided, 
        the method returns pricing information for all available audio models.

        Parameters:
            model_name (str, optional): The name of the specific audio model for which pricing information is sought. 
                                        If None, the method returns pricing information for all audio models.

        Returns:
            dict or None: A dictionary containing the pricing information for the requested audio model(s). 
                        Each key in the dictionary represents a model name, with corresponding pricing details. 
                        Returns None if the requested model name does not exist in the pricing data.

        Raises:
            ValueError: If the specified model_name is not found in the audio models pricing data.

        Example:
            >>> pricing_data = OpenAIPricing(json_data)
            >>> pricing_data.get_audio_model_pricing('Whisper')
            {'usage': 0.006}

            >>> pricing_data.get_audio_model_pricing()
            {
                'Whisper': {'usage': 0.006},
                'TTS': {'usage': 0.015},
                'TTS HD': {'usage': 0.03}
            }
        """

        models = self.data["pricing"]["audio_models"]
        if model_name:
            model_pricing = models.get(model_name)
            if model_pricing is not None:
                return model_pricing
            else:
                raise ValueError(f"Audio model '{model_name}' not found in the pricing data.")
        else:
            return models
    
    def estimate_finetune_training_cost(self, number_of_tokens: int, model_name: str = "gpt-3.5-turbo")-> float:
        """
        Estimates the cost of training or fine-tuning a model based on token count and model selection.

        :method estimate_finetune_training_cost: Calculate the estimated cost for training or fine-tuning.
        :type estimate_finetune_training_cost: method

        :param number_of_tokens: The total number of tokens that will be processed during training or fine-tuning.
        :type number_of_tokens: int

        :param model_name: The name of the AI model for which the training or fine-tuning is being estimated. Defaults to 'gpt-3.5-turbo' if not specified.
        :type model_name: str, optional

        :return: The calculated cost of training or fine-tuning for the given number of tokens and the specified model.
        :rtype: float

        :raises ValueError: If the pricing data for the specified model is unavailable, resulting in an inability to estimate the cost.

        :example:

        ::

            >>> estimate_finetune_training_cost(10000, "gpt-3.5-turbo")
            # Assuming a hypothetical cost calculation, this could return a float representing the cost.
        """

        logger.debug(f"Starting Cost Estimation for {number_of_tokens} tokens using {model_name}")
        try:
            cost_per_token = self.get_fine_tuning_model_pricing(model_name)['training'] / 1000.0
            logger.debug(f"Estimated cost for {number_of_tokens}")
            estimated_cost = cost_per_token * number_of_tokens
            logger.debug(f"Estimated cost for {number_of_tokens} tokens using {model_name}: ${estimated_cost:.2f}")
            return estimated_cost
        except Exception as e:
            logger.error(f"[OpenAIPricing] Pricing information for model {model_name} not found.\n", exc_info=False)
            raise ValueError(f"Pricing information for model {model_name} not found.")

    def estimate_inference_cost(self, input_tokens: int, output_tokens: int, model_name: str = "gpt-3.5-turbo")-> float:
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
            input_cost_per_token = self.get_language_model_pricing(model_name)['input'] / 1000.0
            output_cost_per_token = self.get_language_model_pricing(model_name)['output'] / 1000.0
            logger.debug(f"Estimated cost for {input_tokens} input tokens and {output_tokens} output tokens using {model_name}: ${input_cost_per_token * input_tokens + output_cost_per_token * output_tokens:.2f}")
            return input_cost_per_token * input_tokens + output_cost_per_token * output_tokens
        except Exception as e:
            logger.error(f"[OpenAIPricing] Pricing information for model {model_name} not found. ERROR: {e}\n", exc_info=False)
            raise ValueError(f"Pricing information for model {model_name} not found.")
    


    def calculate_token_usage_for_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """
        Calculates the total number of tokens used by a list of messages, considering the specified model's tokenization scheme.

        :method calculate_token_usage_for_messages: Determine the total token count for a given set of messages based on the model's encoding.
        :type calculate_token_usage_for_messages: method

        :param messages: A list of dictionaries representing messages, with keys such as 'role', 'name', and 'content'.
        :type messages: list of dict

        :param model: Identifier of the model for estimating token count. Defaults to "gpt-3.5-turbo-0613".
        :type model: str, optional

        :return: The total token count for the provided messages as encoded by the specified model.
        :rtype: int

        :raises KeyError: If the token encoding for the specified model is not found in the encoding data.
        :raises NotImplementedError: If the function does not support token counting for the given model.

        :example:

        ::

            >>> messages = [{"role": "user", "content": "Hello!"}, 
            ...             {"role": "assistant", "content": "Hi there!"}]
            >>> calculate_token_usage_for_messages(messages)
            # Assuming the model 'gpt-3.5-turbo-0613', this returns the total token count for the messages.
        """

            # Convert string input to the required dictionary format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

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
        num_tokens = 3  # every reply is primed with 'assistant'
        for message in messages:
            num_tokens += tokens_per_message
           #ensure that we encode only a string. Enforce str(value) this to avoid encoding errors
            num_tokens += sum(len(encoding.encode(str(value))) for key, value in message.items())
            if "name" in message:
                num_tokens += tokens_per_name

        return num_tokens

    def load_dataset(self, file_path):
        """
        Loads a dataset from a specified file, supporting CSV, JSON, or JSONL formats.

        :method load_dataset: Read and load data from a file into an appropriate data structure.
        :type load_dataset: method

        :param file_path: The path to the file containing the dataset.
        :type file_path: str

        :return: A list of dictionaries where each dictionary represents a record in the dataset.
        :rtype: list of dict

        :raises ValueError: If the file format is not supported. Only CSV, JSON, or JSONL formats are acceptable.

        :example:

        ::

            >>> load_dataset("data.csv")
            # This will return the contents of 'data.csv' as a list of dictionaries.
        """

        if file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_dict(orient='records')
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                return json.load(file)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r') as file:
                return [json.loads(line) for line in file]
        else:
            raise ValueError("Unsupported file format. Please use CSV, JSON, or JSONL.")

    def calculate_token_usage_for_dataset(self, dataset_path, model="gpt-3.5-turbo-0613"):
        """
        Calculates the total number of tokens used by a dataset, based on the provided model's tokenization scheme.

        :method calculate_token_usage_for_dataset: Estimate the total token count for a dataset using a specified model.
        :type calculate_token_usage_for_dataset: method

        :param dataset_path: The file path of the dataset for which token usage is to be calculated.
        :type dataset_path: str

        :param model: Identifier of the model for estimating token count. Defaults to "gpt-3.5-turbo-0613".
        :type model: str, optional

        :return: The total token count for the dataset as per the model's encoding scheme.
        :rtype: int

        :example:

        ::

            >>> calculate_token_usage_for_dataset("dataset.jsonl")
            # Assuming the model 'gpt-3.5-turbo-0613', this returns the total token count for the dataset.
        """

        messages = self.load_dataset(dataset_path)
        return self.calculate_token_usage_for_messages(messages, model=model)
    

    def extract_response_and_token_usage_and_cost(self, response, model_name):
        """
        Extracts the content of the response, token usage, and estimates the inference cost.

        Parameters:
        - response (dict): The response message received.

        Returns:
        - tuple: A tuple containing the content of the response, the token usage dictionary, and the estimated cost.
        """
        # Extract the content and token usage
        content = response.choices[0].message.content
        #print('content:', content)
        token_usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
        #print(token_usage)
        
        # Estimate the cost
        cost = self.estimate_inference_cost(token_usage['prompt_tokens'], token_usage['completion_tokens'], model_name)
        
        return content, token_usage, cost
    
    
    def calculate_token_usage_for_text(self, text, model="gpt-3.5-turbo-0613"):
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
    

