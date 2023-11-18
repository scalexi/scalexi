import os
import openai
import time
import csv
import json
import pandas as pd
import logging
from openai import OpenAI
import httpx
from typing import List, Dict, Optional
import scalexi.utilities.data_formatter as dfm

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

RETRY_SLEEP_SECONDS = 0.5

DEFAULT_SYSTEM_PROMPT = """You are an assistant to create a JSON Array of prompt and completions from a context. 
                                Return prompt and completion as a JSON ARRAY structure:\n"
                                [{"prompt": "question1", "completion": "answer1"},{"prompt": "question2", "completion": "answer2"}]"
                                """

class PromptCompletionGenerator:
    def __init__(self, openai_key: Optional[str] = None):
        # Set the OpenAI API key
        self.openai_api_key = openai_key if openai_key else os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key.")

        # Set the API key for the OpenAI client
        openai.api_key = self.openai_api_key

        # Set the default retry sleep seconds
        self.retry_sleep_seconds = 0.5

        # Set the default system prompt
        self.default_system_prompt = """You are an assistant to create prompt and completions from a context. 
                                        Return prompt and completion as a JSON ARRAY structure:\n"
                                        [{"prompt": "question1", "completion": "answer1"},
                                         {"prompt": "question2", "completion": "answer2"}]"""
        
        self.data_formatter = dfm.DataFormatter()


    def parse_and_save_json(self, json_string, output_file, context=None):
        """
        Parses a JSON-formatted string and persists it to a file with optional context.
        -------------------------------------------------------------------------------

        This function offers a two-fold utility: it first parses a given JSON-formatted string, transforming it into a structured data format, and then saves it into a JSON file. Additionally, users have the option to provide contextual data, which, if given, will be incorporated into the saved data, enriching the content.

        Parameters:
        ----------
            json_string (str): 
                A string formatted in JSON, representing structured data to be parsed.
                
            output_file (str): 
                The destination file path where the parsed JSON data will be saved. 
                
            context (str, optional): 
                If provided, this context will be added to the parsed data, augmenting the information. Defaults to None.

        Note:
        -----
            If the `context` is provided, it is essential that it aligns with the structure or schema of the JSON string for consistency in the output file.

        """


        try:
            # Load the JSON array into a list of dictionaries
            data_list = json.loads(json_string)

            # Transform the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(data_list)

            # Initialize a DataFrame to hold the newly formatted data
            formatted_data = pd.DataFrame(columns=['formatted_prompt', 'formatted_completion'])

            # Loop through each row in the DataFrame and format the prompts and completions
            for index, row in df.iterrows():
                prompt = row['prompt']
                completion = row['completion']
                formatted_data = pd.concat([formatted_data, self.format_prompt_completion_df(prompt, completion)], ignore_index=True)

            # Save the formatted data as JSON
            self.data_formatter.df_to_json(formatted_data, output_file)

        except json.decoder.JSONDecodeError as e:
            logger.error("\n\n", e, "\n")
            logger.error("Error parsing JSON. Skipping context ... \n\n", context)

    def generate_system_prompt(self, num_questions: int, question_type: str, detailed_explanation: bool = True):
        """
        Generates a system prompt that includes instructions for the number of questions and their type.
        
        Parameters:
        ----------
        num_questions : int
            The number of questions to be included in the prompt.
        question_type : str
            The type of questions, such as "open-ended", "yes/no", etc.
        detailed_explanation : bool, optional
            Flag indicating whether to include instructions for detailed explanations and arguments. Defaults to True.

        Returns:
        -------
        str
            A string containing the generated system prompt.
        """
    

    

        # Define static questions for different question types
        static_questions = {
            "open-ended": ["What is the capital of France", "How does photosynthesis work", "Where is the Eiffel Tower located", "Why do birds migrate", "When did World War II end"],
            "yes-no": ["Is the sky blue", "Can you swim", "Do cats have tails", "Will it rain tomorrow", "Did you eat breakfast"],
            "multiple-choice": ["Which of the following fruits is red", "Select the option that is a prime number", "Pick the choice that represents a mammal", "Choose the correct answer to 2 + 2", "Which planet is closest to the Sun"],
            "closed-ended": ["On a scale of 1 to 5, how satisfied are you with our service", "Rate your agreement from 1 to 5 with the statement", "How likely are you to recommend our product from 1 to 5", "How would you rate your experience from 1 to 5", "Rate your knowledge level from 1 to 5"],
            "ranking": ["Please rank the following movies in order of preference", "Rank the cities by population size from largest to smallest", "Order the items by importance from most to least", "Rank the books in the order you read them", "Rank the colors by your favorite to least favorite"],
            "hypothetical": ["What would you do if you won the lottery", "In a hypothetical scenario, how would you react if you met a celebrity", "Imagine a situation where you find a wallet on the street", "What would be your response if you saw a UFO", "If you could time travel, where and when would you go"],
            "clarification": ["Could you please explain the concept of blockchain", "I need clarification on the fourth step of the process", "Can you provide more details about the theory of relativity", "Please explain the main idea of the book", "What is the meaning behind the artwork"],
            "leading": ["Don't you agree that exercise is important for a healthy lifestyle", "Isn't it true that honesty is the best policy", "Wouldn't you say that education is valuable", "Aren't you excited about the upcoming event", "Don't you think chocolate is delicious"],
            "critical-thinking": ["How would you solve the problem of climate change", "What are your thoughts on the impact of technology on society", "Can you critically analyze the economic implications of the policy", "What strategies would you use to improve customer satisfaction", "How do you propose to address the issue of poverty"],
            "reflective": ["How do you feel about your recent achievements", "Share your reflections on the past year", "What are your sentiments regarding the current political situation", "Reflect on your experiences during the trip", "How do you perceive the concept of success"]
        }

        # Check if the question_type is valid
        if question_type not in static_questions:
            raise ValueError("Invalid question_type. Supported values are: {}".format(", ".join(static_questions.keys())))

        # Initialize the initial prompt
        system_prompt = "Given the context below, generate a JSON array with {} precisely crafted pairs of prompts as {} questions and their corresponding completions as JSON Array, following these guidelines for the context below:\n".format(num_questions, question_type)

        # Generate example of questions based on the specified type to be added an a few-shot learning example
        for i in range(1, num_questions + 1):
            questions = static_questions[question_type]
            if i <= len(questions):
                question = questions[i - 1] + "?"
            else:
                question = "Example {}: {} question {}".format(i, question_type.capitalize(), i)
            
            #system_prompt += "Example {}: {}\n".format(i, question)

        # Add the remaining context
        #system_prompt += "Each prompt is inherently correctly answerable with an in-depth and justified response.\n"
        # Include detailed instructions based on the flag
        if detailed_explanation:
            system_prompt += "Each response to a prompt should be meticulously crafted to offer a detailed explanation along with a robust argument to substantiate the response.\n"
            system_prompt += "Each completion must be developed offering a sufficient explanation and ample arguments to justify the given response.\n"

        #system_prompt += "The returned response of all prompts should be formatted within a JSON ARRAY structure.\n"
        #system_prompt += "Each individual JSON record should encapsulate one distinct prompt and its corresponding in-depth completion.\n"
        json_prefix = """```json"""
        #system_prompt += "[LIFE CRITICAL REQUIREMENT] Output Format: prompt and completion as a JSON ARRAY structure:\n"
        system_prompt += 'EXACT JSON ARRAY  STRUCTURE FORMAT:\n[{\"prompt\": \"question1\", \"completion\": \"answer1\"}, {\"prompt\": \"question1\", \"completion\": \"answer1\"}]'
        #system_prompt += "\ndo NOT add "+json_prefix+" as prefix. \n\n"
        #system_prompt += "```json\n"
        #system_prompt += "[\n"
        #system_prompt += '    {"prompt": "question1", "completion": "answer1"},\n'
        #system_prompt += '    {"prompt": "question2", "completion": "answer2"}\n'
        #system_prompt += "]\n"
        #system_prompt += "```\n"
        

        return system_prompt


    def generate_prompt_completions(self, context_text: str, output_csv: str,
                                    user_prompt: str = "",
                                    openai_key: Optional[str] = None,
                                    temperature: float = 0.1, 
                                    model: str = "gpt-3.5-turbo-1106",
                                    max_tokens: int = 1054, 
                                    top_p: float = 1.0,
                                    frequency_penalty: float = 0.0, 
                                    presence_penalty: float = 0.0,
                                    retry_limit: int = 3,
                                    num_questions: int = 3, 
                                    question_type: str = "open-ended",
                                    detailed_explanation: bool = True) -> List[Dict[str, str]]:
        """
        Generates prompt completions using the OpenAI API and records them to a CSV file.

        This function utilizes the specified OpenAI model to generate responses based on provided context. 
        It records the prompt-completion pairs to a CSV file for storage and further use. Customization options 
        include the model, generation parameters, and the type of questions generated.

        Parameters:
        ----------
        context_text : str
            The context based on which prompts are generated.
        output_csv : str
            The file path for saving generated completions in CSV format.
        user_prompt : str, optional
            A user-defined initial prompt. Defaults to an empty string.
        openai_key : str, optional
            The API key for authenticating requests to the OpenAI service. Defaults to None, using the environment variable "OPENAI_API_KEY".
        temperature : float, optional
            The level of randomness in the output. Higher values lead to more varied outputs. Defaults to 1.
        model : str, optional
            The OpenAI model used for generation. Defaults to "gpt-3.5-turbo-1106".
        max_tokens : int, optional
            The maximum length of the generated output. Defaults to 1054.
        top_p : float, optional
            The proportion of most likely tokens considered for sampling. Defaults to 1.
        frequency_penalty : float, optional
            The decrease in likelihood for frequently used tokens. Defaults to 0.
        presence_penalty : float, optional
            The decrease in likelihood for already used tokens. Defaults to 0.
        retry_limit : int, optional
            The maximum number of retries for API call failures. Defaults to 3.
        num_questions : int, optional
            The number of questions to generate. Defaults to 3.
        question_type : str, optional
            The type of questions to generate, such as "open-ended", "yes/no", etc. Defaults to "open-ended".
        detailed_explanation : bool, optional
            Flag indicating whether to include instructions for detailed explanations and arguments. Defaults to True.

        Returns:
        -------
        List[Dict[str, str]]:
            A list of dictionaries, each containing 'prompt' and 'completion' keys.

        Raises:
        ------
        ValueError:
            If the OpenAI API key is invalid or not provided.
        Exception:
            If the function fails after the specified number of retries.

        Notes:
        -----
        It is crucial to have proper API key authorization for successful API requests. Ensure the OpenAI key is valid and has the necessary permissions.
        """

        # Attempt to get the API key from the function parameter or the environment variable
        openai_api_key = openai_key if openai_key else os.environ["OPENAI_API_KEY"]

        # Now check if the obtained key is valid
        if not openai_api_key or not openai_api_key.startswith("sk-"):
            logger.error("Invalid OpenAI API key. The key must start with 'sk-' and cannot be None.")
            raise ValueError("Invalid OpenAI API key. The key must start with 'sk-' and cannot be None.")

        # If the key is valid, set it
        openai.api_key = openai_api_key

    
        # Customize the initial prompt based on the number and type of questions
        OUTPUT_FORMAT='[{"prompt": "question1", "completion": "answer1"}, {"prompt": "question1", "completion": "answer1"}]'
        system_prompt = self.generate_system_prompt(num_questions, question_type, detailed_explanation)
        #user_prompt = 'requirement:  The output must be a JSON ARRAY structure exactly like: '+ OUTPUT_FORMAT +"\ncontext: \n" + context_text
        user_prompt = "context: \n" + context_text
        # Log the function call
        redacted_key = openai.api_key[:10]+"..." 
        logger.debug(
            f"Called generate_prompt_completions with params: \n"
            f"context_text={context_text}, \noutput_csv={output_csv}, "
            f"system_prompt={system_prompt}, \nuser_prompt={user_prompt}, \n"
            f"openai_key={redacted_key}, \ntemperature={temperature}, \n"
            f"model={model}, \nmax_tokens={max_tokens}, \ntop_p={top_p}, \n"
            f"frequency_penalty={frequency_penalty}, \npresence_penalty={presence_penalty}, \n"
            f"retry_limit={retry_limit}, \nnum_questions={num_questions}, \n"
            f"question_type={question_type}"
        )


        retry_count = 0

        # Configure the OpenAI client with retries and timeout settings
        client = OpenAI(api_key=openai_api_key, max_retries=retry_limit)
        client = client.with_options(timeout=httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0))

        while retry_count < retry_limit:
            try:
                logger.debug(f"Attempting to generate prompt completions with params: \n")

                response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=["END"]
                )
                
                logger.debug(f"[generate_prompt_completions] Successfully generated prompt completions with params: \n")
                # Process the response
                #if 'choices' in response and len(response.choices) > 0:
                json_string = response.choices[0].message.content #.strip()
                logger.debug(f"[generate_prompt_completions] Successfully processed response: \n{json_string} ...")
                # Parse the JSON string

                logger.info(f"[generate_prompt_completions] Attempting to parse JSON response: \n\n{json_string} ...")
                prompt_completion_pairs = self.data_formatter.extract_json_array(json_string) # remove ```json and ``` from the string if exists and extract the array as list of dict
                logger.info(f"[generate_prompt_completions] Successfully parsed JSON response")
                
                # Save to CSV
                list_to_save = [{"prompt": pair["prompt"], "completion": pair["completion"]} for pair in prompt_completion_pairs]
                self.data_formatter.list_to_csv(list_to_save, output_csv)
                return self.data_formatter.remove_json_markers(json_string) # remove ```json and ``` from the string and return the json string
                #else:
                #    retry_count += 1
                #    logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n")
                #    logger.error(f"[generate_prompt_completions] retry_count:Error processing response: \n{response}")
                #    time.sleep(RETRY_SLEEP_SECONDS)
                #    raise Exception(f"[generate_prompt_completions] Error processing OpenAI response: \n{response}")
            except json.JSONDecodeError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Error decoding JSON response: {e}. \njson_string: {json_string}")
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
                time.sleep(RETRY_SLEEP_SECONDS)

            except openai.APIConnectionError as e:
                retry_count += 1            
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an APIConnectionError error:\n{e}")
                logger.error(f"[generate_prompt_completions] Retrying in 0.5 seconds...")
                time.sleep(RETRY_SLEEP_SECONDS)

            except openai.RateLimitError as e:
                # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} failed due to RateLimit Error {e}. Max tokens = {max_tokens}. Trying again in 0.5 seconds...")
                max_tokens = int(max_tokens * 0.8)
                time.sleep(RETRY_SLEEP_SECONDS)  # Pause for 0.5 seconds

            except openai.APIStatusError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an APIStatusError:\n{e}")
                # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter
                logger.error(f"\n\n[generate_prompt_completions] Service is currently unavailable. Waiting for 10 seconds before retrying...\n\n")
                time.sleep(10)  # Pause for 10 seconds

            except AttributeError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an AttributeError:\n{e}")
                # Handle the exception and log the error message
                logging.error("[generate_prompt_completions] An AttributeError occurred: %s", e)
                # You can also add additional error handling code here if needed

            except Exception as e:
                logger.error(f"\n\n[generate_prompt_completions] Exception: {e}\n\n")
                retry_count += 1
                time.sleep(20)
                if retry_count > 3:
                    logger.error(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text}. Exit.")
                    raise Exception(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")

        if retry_count >=3:
            logger.error(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")
            raise Exception("[generate_prompt_completions] Gave up after max retry limit attempts ...\n Exit.")

        return self.data_formatter.remove_json_markers(json_string) # remove ```json and ``` from the string and return the json string
 

