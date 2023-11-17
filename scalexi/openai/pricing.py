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
    A class for handling and accessing pricing data for various OpenAI models.

    This class allows easy access to pricing information for different categories and models
    as provided by OpenAI. The data is expected to be a dictionary obtained from parsing
    a JSON structure with specific keys representing different aspects of the pricing.

    The expected JSON structure is as follows:
    - release_date: A string representing the release date of the pricing data.
    - pricing: A nested dictionary containing various categories of pricing information:
        - language_models: Contains pricing information for different language models, structured as:
            - Model Category (e.g., "GPT-4 Turbo"): A dictionary containing:
                - context: A description of the model's capabilities.
                - models: A nested dictionary of specific model configurations, each with:
                    - Model Name (e.g., "gpt-4-1106-preview"): A dictionary containing:
                        - input: Cost per 1K tokens for inputs.
                        - output: Cost per 1K tokens for outputs.
        - assistants_api: Contains pricing information for assistants API tools, structured as:
            - Tool Name (e.g., "Code interpreter"): A dictionary containing:
                - input: Cost per session or usage.
                - note: Additional notes, such as free usage periods.
        - fine_tuning_models: Contains pricing for fine-tuning models, structured as:
            - Model Name (e.g., "gpt-3.5-turbo"): A dictionary containing:
                - training: Cost per 1K tokens for training.
                - input_usage: Cost per 1K tokens for input during inference.
                - output_usage: Cost per 1K tokens for output during inference.
        - embedding_models: Contains pricing for embedding models, structured as:
            - Model Name (e.g., "ada v2"): A dictionary containing:
                - usage: Cost per 1K tokens for usage.
        - base_models: Contains pricing for base models, structured as:
            - Model Name (e.g., "davinci-002"): A dictionary containing:
                - usage: Cost per 1K tokens for usage.
        - image_models: Contains pricing for image models, structured as:
            - Model Category (e.g., "DALL·E 3"): A dictionary containing:
                - Quality Level (e.g., "Standard"): A dictionary of resolutions and their prices:
                    - Resolution (e.g., "1024x1024"): Price per image.
        - audio_models: Contains pricing for audio models, structured as:
            - Model Name (e.g., "Whisper"): A dictionary containing:
                - usage: Cost per minute or per character for usage.

    Parameters:
    - json_data (dict): The pricing data in JSON format, parsed into a dictionary.

    Attributes:
    - data (dict): The parsed JSON data as a dictionary accessible to the instance.

    Methods:
    - get_release_date(): Returns the release date of the pricing data.
    - get_language_model_pricing(model_name=None): Returns pricing information for specified language models.
    - get_assistants_api_pricing(tool_name=None): Returns pricing information for the assistants API tools.
    - get_fine_tuning_model_pricing(model_name=None): Returns pricing information for fine-tuning models.
    - get_embedding_model_pricing(model_name=None): Returns pricing information for embedding models.
    - get_base_model_pricing(model_name=None): Returns pricing information for base models.
    - get_image_model_pricing(model_name=None): Returns pricing information for image models.
    - get_audio_model_pricing(model_name=None): Returns pricing information for audio models.

    Example JSON data input:
    {
        "release_date": "2023-11-15",
        "pricing": {
            "language_models": {
                "GPT-4 Turbo": {
                    "context": "128k context, fresher knowledge ...",
                    "models": {
                        "gpt-4-1106-preview": { "input": 0.01, "output": 0.03 },
                        ...
                    }
                },
                ...
            },
            ...
        }
    }

    Each 'get' method can optionally take a model or tool name as an argument. If provided,
    it will return the specific pricing data for that name. If not provided, it returns the
    entire category of pricing data.
    """
    # Class implementation...

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
        Retrieves the pricing information for language models from the stored pricing data.

        If a specific model name is provided, this method returns the pricing details for that
        particular language model. If no model name is given, it returns the pricing information
        for all language models in a dictionary.

        Parameters:
            model_name (str, optional): The name of the specific language model to retrieve pricing
                                        for. If None, returns all language models' pricing info.

        Returns:
            dict: A dictionary containing the pricing information. If a specific model name is given,
                the dictionary contains keys 'input' and 'output' with respective costs. If no model
                name is provided, returns a dictionary with model categories as keys and their
                respective pricing dictionaries as values.

        Raises:
            ValueError: If a specific model_name is provided but not found in the pricing data.

        Example:
            >>> pricing_data.get_language_model_pricing('GPT-4')
            {'input': 0.03, 'output': 0.06}
        """
        models = self.data["pricing"]["language_models"]
        if model_name:
            if model_name in models:
                return models[model_name]
            else:
                raise ValueError(f"Model '{model_name}' not found in language models pricing.")
        else:
            return models

    def get_assistants_api_pricing(self, tool_name=None):
        """
        Retrieves pricing information for the Assistants API tools from the stored pricing data.

        If a specific tool name is provided, this method returns the pricing details for that
        particular tool. If no tool name is given, it returns the pricing information for all
        tools under the Assistants API in a dictionary.

        Parameters:
            tool_name (str, optional): The name of the specific Assistants API tool to retrieve
                                    pricing for. If None, returns all tools' pricing info.

        Returns:
            dict: A dictionary containing the pricing information. If a specific tool name is given,
                the dictionary contains the tool's cost and any additional notes. If no tool name
                is provided, returns a dictionary with tool names as keys and their respective
                pricing dictionaries as values.

        Raises:
            ValueError: If a specific tool_name is provided but not found in the pricing data.

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

        This method can be used to retrieve the cost of training, input usage, and output usage
        for a specific fine-tuning model if the model name is provided. If no model name is
        provided, it returns a dictionary of all available fine-tuning models with their
        respective pricing information.

        Parameters:
            model_name (str, optional): The name of the specific fine-tuning model for which
                                        to retrieve pricing information. If None, pricing
                                        information for all fine-tuning models is returned.
                                        Default is None.

        Returns:
            dict: A dictionary containing pricing information. If a model name is provided,
                it returns a dictionary with keys 'training', 'input_usage', and 'output_usage'.
                If no model name is provided, it returns a dictionary with model names as keys
                and their respective pricing information as values.

        Raises:
            ValueError: If a model_name is provided but does not exist in the pricing data.

        Example:
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

        If a specific model name is provided, this method returns the pricing details for that
        particular embedding model. If no model name is given, it returns the pricing information
        for all embedding models in a dictionary.

        Parameters:
            model_name (str, optional): The name of the specific embedding model to retrieve pricing
                                        for. If None, returns all embedding models' pricing info.

        Returns:
            dict: A dictionary containing the pricing information. If a specific model name is given,
                the dictionary contains keys 'usage' with respective costs. If no model
                name is provided, returns a dictionary with model names as keys and their
                respective pricing dictionaries as values.

        Raises:
            ValueError: If a specific model_name is provided but not found in the pricing data.

        Example:
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
        Retrieves pricing information for base models from the provided pricing data.

        This method returns the cost associated with the use of base language models as specified in the
        pricing data. If a specific model name is provided, it returns the pricing for that model.
        If no model name is provided, it returns the pricing for all base models.

        Parameters:
            model_name (str, optional): The name of the base model for which pricing information is requested.
                                        If None, the method returns pricing information for all base models.

        Returns:
            dict: A dictionary containing the pricing information for the requested base model(s).
                Each key in the dictionary is a model name, and the value is another dictionary
                with details about the usage cost.

        Raises:
            ValueError: If the model_name is provided but not found in the base models pricing data.

        Example:
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


    def get_image_model_pricing(self, model_name=None):
        """
        Retrieves pricing information for image models from the pricing data.

        This method provides pricing details for image generation models. If a specific
        model name is provided, it returns the pricing for that model. If no model name
        is provided, it returns the pricing information for all available image models.

        Parameters:
        ----------
        model_name : str, optional
            The name of the specific image model for which pricing information is requested.
            If None, the method returns pricing information for all image models.

        Returns:
        -------
        dict or None
            A dictionary containing the pricing information for the requested image model(s).
            Each key in the dictionary represents a model name or category, with corresponding
            pricing details as a nested dictionary.

        Raises:
        ------
        ValueError
            If the model_name is provided but not found in the image models pricing data.

        Example:
        --------
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
        Retrieves the pricing information for audio models from the stored data.

        This method fetches the pricing details for audio-related models, such as transcription
        and text-to-speech services. If a specific model name is provided, it returns the pricing
        for that model. If no model name is provided, it returns the pricing information for all
        available audio models.

        Parameters:
        ----------
        model_name : str, optional
            The name of the specific audio model for which pricing information is requested.
            If None, the method returns pricing information for all audio models.

        Returns:
        -------
        dict or None
            A dictionary containing the pricing information for the requested audio model(s).
            Each key in the dictionary represents a model name, with the corresponding pricing
            details.

        Raises:
        ------
        ValueError
            If the model_name is provided but not found in the audio models pricing data.

        Example:
        --------
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
            Estimates the cost of training or fine-tuning based on token count and model.

            Parameters
            ----------
            number_of_tokens : int
                The number of tokens that will be processed.
            model_name : str, optional
                The name of the model to be used, defaulting to 'gpt-3.5-turbo'.

            Returns
            -------
            float
                The estimated cost for the specified number of tokens and operation.

            Raises
            ------
            ValueError
                If pricing for the specified model is not found in the pricing data.
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
            Estimates the cost of inference based on token count and model.

            Parameters
            ----------
            input_tokens : int
                The number of tokens that will be processed as input.
            output_tokens : int
                The number of tokens that will be processed as output.
            model_name : str, optional
                The name of the model to be used, defaulting to 'gpt-3.5-turbo'.

            Returns
            -------
            float
                The estimated cost for the specified number of tokens and operation.

            Raises
            ------
            ValueError
                If pricing for the specified model is not found in the pricing data.
            """
        logger.debug(f"Starting Cost Estimation for {input_tokens} input tokens and {output_tokens} output tokens using {model_name}")
        try:
            input_cost_per_token = self.get_fine_tuning_model_pricing(model_name)['input_usage'] / 1000.0
            output_cost_per_token = self.get_fine_tuning_model_pricing(model_name)['output_usage'] / 1000.0
            logger.debug(f"Estimated cost for {input_tokens} input tokens and {output_tokens} output tokens using {model_name}: ${input_cost_per_token * input_tokens + output_cost_per_token * output_tokens:.2f}")
            return input_cost_per_token * input_tokens + output_cost_per_token * output_tokens
        except Exception as e:
            logger.error(f"[OpenAIPricing] Pricing information for model {model_name} not found.\n", exc_info=False)
            raise ValueError(f"Pricing information for model {model_name} not found.")
    


    def calculate_token_usage_for_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """
        Calculate the total number of tokens used by a list of messages.

        This function estimates the token usage for messages based on the model's
        tokenization scheme. It supports different versions of GPT-3.5 Turbo and
        GPT-4 models. For unsupported models, a NotImplementedError is raised.
        This is used to estimate the cost of interactions with OpenAI's API based
        on message lengths.

        Parameters
        ----------
        messages : list of dict
            List of message dictionaries with keys like 'role', 'name', and 'content'.
        model : str, optional
            Identifier of the model to estimate token count. Default is "gpt-3.5-turbo-0613".

        Returns
        -------
        int
            Total number of tokens for the messages as per the model's encoding scheme.

        Raises
        ------
        KeyError
            If the model's encoding is not found.
        NotImplementedError
            If token counting is not implemented for the model.

        Examples
        --------
        >>> messages = [{"role": "user", "content": "Hello!"}, 
        ...             {"role": "assistant", "content": "Hi there!"}]
        >>> calculate_token_usage_for_messages(messages)
        14  # Example token count for "gpt-3.5-turbo-0613" model.
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
        Load a dataset from a file which can be in CSV, JSON or JSONL format.
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
        Calculate the total number of tokens used by a dataset.
        """
        messages = self.load_dataset(dataset_path)
        return self.calculate_token_usage_for_messages(messages, model=model)
    
