import os
import openai
import time
import csv
import json
from typing import Optional, Dict, List
import pandas as pd
import logging

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

DEFAULT_SYSTEM_RPOMPT = """You are an assistant to create prompt and completions from a context. 
                                Return prompt and completion as a JSON ARRAY structure:\n"
                                [{"prompt": "question1", "completion": "answer1"},{"prompt": "question2", "completion": "answer2"}]"
                                """

def format_prompt_completion(prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
    """
    Formats a given prompt and its corresponding completion into a structured dictionary format.
    --------------------------------------------------------------------------------------------

    This function is designed to take a user-defined prompt and its associated completion and format them into a structured sequence. By default, the function uses specific start and end sequences to encapsulate the prompt-completion pair, but these can be customized as needed. Such structured formatting is essential when working with datasets for large language models, ensuring uniformity and ease of processing.

    Parameters
    ----------
        prompt (str): 
            The primary text or question that requires a completion. This serves as the context for the model.
            
        completion (str): 
            The model's response or answer to the given prompt. It completes the information or context provided by the prompt.
            
        start_sequence (str, optional): 
            A sequence of characters that signifies the beginning of a new prompt-completion pair. It helps in demarcating different entries in a dataset. Defaults to "\\n\\n###\\n\\n".
            
        end_sequence (str, optional): 
            A sequence of characters that marks the end of a prompt-completion pair. Like the start sequence, it aids in distinguishing between different dataset entries. Defaults to "END".

    Returns
    -------
        dict: 
            A dictionary containing the formatted prompt and completion. This structured output can be directly used for dataset creation or other related tasks.
    """

    logger.info("starting format_prompt_completion ...")
    formatted_prompt = prompt.strip() + start_sequence
    formatted_completion = completion.strip() + end_sequence
    return {"prompt": formatted_prompt, "completion": formatted_completion}

def format_prompt_completion_df(prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
    """
        Formats and structures a given prompt and its completion into a DataFrame.
        --------------------------------------------------------------------------

        This function facilitates the organization of a user-defined prompt and its corresponding completion into a structured pandas DataFrame. Such organization is especially beneficial when handling, analyzing, or storing large datasets for machine learning tasks, ensuring a uniform and easily accessible data structure.

        Parameters:
        ----------
            prompt (str): 
                The initial text or question that is presented to a model or system. It provides context or a scenario which requires a subsequent completion or answer.
                
            completion (str): 
                The subsequent text or response that corresponds to the prompt. It provides additional information or a resolution to the context given by the prompt.
                
            start_sequence (str, optional): 
                A sequence of characters that denotes the beginning of the prompt. It helps segment and recognize the start of a data entry in larger datasets. Defaults to "\\n\\n###\\n\\n".
                
            end_sequence (str, optional): 
                A sequence of characters that indicates the conclusion of the completion. It assists in segmenting and marking the end of a data entry in larger datasets. Defaults to "END".

        Returns:
        -------
            pandas.DataFrame: 
                A DataFrame containing two columns: 'Prompt' and 'Completion'. This structured output facilitates easier data manipulation, analysis, and storage.
        """
    logger.info("starting format_prompt_completion_df ...")
    formatted_prompt = prompt.strip() + start_sequence
    formatted_completion = completion.strip() + end_sequence
    return pd.DataFrame({"formatted_prompt": [formatted_prompt], "formatted_completion": [formatted_completion]})

def df_to_json(df, json_output):
    """
    Converts a DataFrame to JSON and saves it to a file.

    Args:
        df (pandas.DataFrame): The DataFrame to convert and save.
        json_output (str): The path to the output JSON file.
    """
    mode = 'a' if os.path.exists(json_output) else 'w'
    try:
        with open(json_output, mode) as f:
            # Convert the DataFrame to a list of dictionaries (records)
            records = df.to_dict(orient='records')
            for response in records:
                f.write(json.dumps(response) + '\n')
    except Exception as e:
        logger.error(f"Error while saving responses to JSON: {e}")
        logger.error(df)

def list_to_csv(data_list, output_file):
    """
    Saves a list of dictionaries to a CSV file.
    -------------------------------------------


    This function provides a streamlined method to persistently store structured data, specifically a list of dictionaries, into a CSV format. Such serialization is instrumental when aiming to maintain data consistency, ensure data portability, or perform data interchange tasks.

    Parameters:
    ----------`
        data_list (list of dict): 
            A list containing dictionaries. Each dictionary in the list represents a record, where keys correspond to column names and values to the data entries.
            
        output_file (str): 
            The file path where the resulting CSV will be saved. If a file already exists at this path, it will be overwritten.

    """


    if not data_list:
        logger.error("Data list is empty. Nothing to save.")
        return
    
    mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_list[0].keys())
        
        if mode == 'w':
            writer.writeheader()
        
        writer.writerows(data_list)

def df_to_csv(df, output_file):
    """
    Append a DataFrame to a CSV file or creates a new one if it doesn't exist.
    ---------------------------------------------------------------------------

    This function is specifically designed to handle the persistence of data in pandas DataFrame format. If the specified CSV file does not exist, it will create a new one and add a header. If the file does exist, it will simply append the data from the DataFrame without adding another header, ensuring data continuity and avoiding redundancy.

    Parameters:
    ----------
        output_file (str): 
            The target file path where the DataFrame data should be written or appended. 
            
        df (pandas.DataFrame): 
            The data in DataFrame format that needs to be persisted to the CSV file.

    Note:
    -----
        Existing data in the CSV file will not be overwritten. Instead, new data will be appended to ensure no data loss.

    """

    if not os.path.isfile(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

def parse_and_save_json(json_string, output_file, context=None):
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
            formatted_data = pd.concat([formatted_data, format_prompt_completion_df(prompt, completion)], ignore_index=True)

        # Save the formatted data as JSON
        df_to_json(formatted_data, output_file)

    except json.decoder.JSONDecodeError as e:
        logger.error("\n\n", e, "\n")
        logger.error("Error parsing JSON. Skipping context ... \n\n", context)

def generate_system_prompt(num_questions: int, question_type: str, detailed_explanation: bool = True):
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
    system_prompt = "Given the context below, craft precisely {} pairs of prompts as {} questions and their corresponding completions, following these guidelines for the context below:\n".format(num_questions, question_type)

    # Generate example of questions based on the specified type to be added an a few-shot learning example
    for i in range(1, num_questions + 1):
        questions = static_questions[question_type]
        if i <= len(questions):
            question = questions[i - 1] + "?"
        else:
            question = "Example {}: {} question {}".format(i, question_type.capitalize(), i)
        
        system_prompt += "Example {}: {}\n".format(i, question)

    # Add the remaining context
    #system_prompt += "Each prompt is inherently correctly answerable with an in-depth and justified response.\n"
     # Include detailed instructions based on the flag
    if detailed_explanation:
        system_prompt += "Each response to a prompt should be meticulously crafted to offer a detailed explanation along with a robust argument to substantiate the response.\n"
        system_prompt += "Each completion must be developed offering a sufficient explanation and ample arguments to justify the given response.\n"

    #system_prompt += "The returned response of all prompts should be formatted within a JSON ARRAY structure.\n"
    #system_prompt += "Each individual JSON record should encapsulate one distinct prompt and its corresponding in-depth completion.\n"
    json_prefix = """```json"""
    system_prompt += "[LIFE CRITICAL REQUIREMENT] Output Format: prompt and completion as a JSON ARRAY structure:\n do NOT add "+json_prefix+" as prefix. Sample output format:\n"
    system_prompt += '[{"prompt": "question1", "completion": "answer1"}, {"prompt": "question1", "completion": "answer1"}]'
    #system_prompt += "```json\n"
    #system_prompt += "[\n"
    #system_prompt += '    {"prompt": "question1", "completion": "answer1"},\n'
    #system_prompt += '    {"prompt": "question2", "completion": "answer2"}\n'
    #system_prompt += "]\n"
    #system_prompt += "```\n"
    

    return system_prompt


def generate_prompt_completions(context_text: str, output_csv: str,
                                system_prompt: str = DEFAULT_SYSTEM_RPOMPT, 
                                user_prompt: str = "",
                                openai_key: Optional[str] = None,
                                temperature: float = 1, 
                                model: str = "gpt-3.5-turbo-1106",
                                max_tokens: int = 1054, 
                                top_p: float = 1,
                                frequency_penalty: float = 0, 
                                presence_penalty: float = 0,
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
    system_prompt : str, optional
        A system-level prompt that guides the generation. Defaults to DEFAULT_SYSTEM_RPOMPT.
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
    system_prompt = generate_system_prompt(num_questions, question_type, detailed_explanation)
    user_prompt = 'requirement:  The output must be a JSON ARRAY structure exactly like: '+ OUTPUT_FORMAT +"\ncontext: " + context_text
    
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

    while retry_count < retry_limit:
        try:
            logger.debug(f"Attempting to generate prompt completions with params: \n")
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            logger.debug(f"[generate_prompt_completions] Successfully generated prompt completions with params: \n")
            # Process the response
            #if 'choices' in response and len(response.choices) > 0:
            json_string = response.choices[0].message['content'].strip()
            logger.debug(f"[generate_prompt_completions] Successfully processed response: \n{json_string[:100]} ...")
            # Parse the JSON string

            logger.info(f"[generate_prompt_completions] Attempting to parse JSON response: \n\n{json_string[:100]} ...")
            prompt_completion_pairs = json.loads(json_string)
            logger.info(f"[generate_prompt_completions] Successfully parsed JSON response")
            
            # Save to CSV
            list_to_save = [{"Prompt": pair["prompt"], "Completion": pair["completion"]} for pair in prompt_completion_pairs]
            list_to_csv(list_to_save, output_csv)
            break
            #else:
            #    retry_count += 1
            #    logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n")
            #    logger.error(f"[generate_prompt_completions] retry_count:Error processing response: \n{response}")
            #    time.sleep(RETRY_SLEEP_SECONDS)
            #    raise Exception(f"[generate_prompt_completions] Error processing OpenAI response: \n{response}")
        except json.JSONDecodeError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
                logger.error(f"Error decoding JSON response: {e}. json_string: {json_string}")
                time.sleep(RETRY_SLEEP_SECONDS)
        except openai.error.OpenAIError as e:
            retry_count += 1            
            logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
            logger.error(f"[generate_prompt_completions] This error may be caused by the length of the context in the corresponding cell in the context.csv file.")
            logger.error(f"[generate_prompt_completions] To resolve this issue, consider reducing the length of the context or check the contents of the file.")
            logger.error(f"[generate_prompt_completions] Retrying in 0.5 seconds...")
            time.sleep(RETRY_SLEEP_SECONDS)
        except openai.error.InvalidRequestError as e:
            # If the request fails due to invalid parameters, increment the retry counter, sleep for 0.5 seconds, and then try again
            retry_count += 1
            logger.error(f"Retry attempt {retry_count} failed due to Invalid Request {e}. Max tokens = {max_tokens}. Trying again in 0.5 seconds...")
            max_tokens = int(max_tokens *0.8)
            time.sleep(RETRY_SLEEP_SECONDS)  # Pause for 0.5 seconds
        except openai.error.ServiceUnavailableError:
            retry_count += 1
            logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter
            logger.error(f"\n\n[generate_prompt_completions] Service is currently unavailable. Waiting for 10 seconds before retrying...\n\n")
            time.sleep(10)  # Pause for 10 seconds
        except AttributeError as e:
            retry_count += 1
            logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
            # Handle the exception and log the error message
            logging.error("[generate_prompt_completions] An AttributeError occurred: %s", e)
            # You can also add additional error handling code here if needed
        except Exception as e:
            logger.error("\n\n[generate_prompt_completions] Exception: \n",e," \n\n")
            retry_count += 1
            time.sleep(20)
            if retry_count > 3:
                logger.error(f"Gave up after {retry_limit} attempts for context: {context_text}. Exit.")
                raise Exception(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")

    if retry_count >=3:
        logger.error(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")
        raise Exception("[generate_prompt_completions] Gave up after max retry limit attempts ...\n Exit.")

    return json_string
