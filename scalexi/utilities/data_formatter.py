import json
import yaml
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
import logging
from pygments import highlight, lexers, formatters

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self):
        pass

    def pretty_print_json(self, data):
        """ Pretty print a Python dictionary as formatted and colored JSON. """
        if type(data) == dict:
            formatted_json = json.dumps(data, sort_keys=True, indent=4)
        else:
            formatted_json = data
        #formatted_json = json.dumps(data, sort_keys=True, indent=4)  # Convert dict to formatted JSON string
        colorful_json = highlight(formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter())  # Apply syntax highlighting
        print(colorful_json)
    
    def format_prompt_completion(self, prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
        """
        Formats and structures a user-defined prompt and its corresponding completion into a pandas DataFrame.

        :method format_prompt_completion: Organizes a prompt and its completion into a structured DataFrame, aiding in data handling and analysis for machine learning tasks.
        :type format_prompt_completion: method

        :param prompt: The initial text or question presented to a model or system, providing context or a scenario for a subsequent completion or answer.
        :type prompt: str

        :param completion: The text or response that follows the prompt, offering additional information or a resolution to the given context.
        :type completion: str

        :param start_sequence: A character sequence denoting the beginning of the prompt, aiding in segmenting data entries in large datasets. Defaults to "\\n\\n###\\n\\n".
        :type start_sequence: str, optional

        :param end_sequence: A character sequence indicating the end of the completion, assisting in marking the conclusion of data entries in large datasets. Defaults to "END".
        :type end_sequence: str, optional

        :return: A DataFrame with two columns, 'Prompt' and 'Completion', facilitating easier data manipulation and analysis.
        :rtype: pandas.DataFrame

        :example:

        ::

            >>> format_prompt_completion("How is the weather today?", "It is sunny and warm.")
            # This will return a DataFrame with the prompt and completion structured in two separate columns.
        """



        logger.info("starting format_prompt_completion ...")
        formatted_prompt = prompt.strip() + start_sequence
        formatted_completion = completion.strip() + end_sequence
        return {"prompt": formatted_prompt, "completion": formatted_completion}

    def format_prompt_completion_df(self, prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
        """
        Formats and structures a user-defined prompt and its corresponding completion into a pandas DataFrame.

        :method [FunctionName]: Organizes a prompt and its completion into a structured DataFrame, aiding in data handling and analysis for machine learning tasks.
        :type [FunctionName]: method

        :param prompt: The initial text or question presented to a model or system, providing context or a scenario for a subsequent completion or answer.
        :type prompt: str

        :param completion: The text or response that follows the prompt, offering additional information or a resolution to the given context.
        :type completion: str

        :param start_sequence: A character sequence denoting the beginning of the prompt, aiding in segmenting data entries in large datasets. Defaults to "\\n\\n###\\n\\n".
        :type start_sequence: str, optional

        :param end_sequence: A character sequence indicating the end of the completion, assisting in marking the conclusion of data entries in large datasets. Defaults to "END".
        :type end_sequence: str, optional

        :return: A DataFrame with two columns, 'Prompt' and 'Completion', facilitating easier data manipulation and analysis.
        :rtype: pandas.DataFrame

        :example:

        ::

            >>> format_prompt_completion_df("How is the weather today?", "It is sunny and warm.")
            # This will return a DataFrame with the prompt and completion structured in two separate columns.
        """

        logger.info("starting format_prompt_completion_df ...")
        formatted_prompt = prompt.strip() + start_sequence
        formatted_completion = completion.strip() + end_sequence
        return pd.DataFrame({"formatted_prompt": [formatted_prompt], "formatted_completion": [formatted_completion]})

    def df_to_json(self, df, json_output):
        """
        Converts a DataFrame to JSON format and saves it to a specified file.

        This method is designed to take a pandas DataFrame, convert it into a JSON format, and save it to a file. 
        It appends to the file if it already exists, or creates a new file if it does not. This function is useful 
        for data serialization and storage, especially in data processing and machine learning workflows.

        :method df_to_json: Converts the given DataFrame to JSON format and saves it to the specified file path.
        :type df_to_json: method

        :param df: The DataFrame that needs to be converted to JSON.
        :type df: pandas.DataFrame

        :param json_output: The file path where the JSON data will be saved. 
                            The method appends to the file if it exists, or creates a new one if it does not.
        :type json_output: str

        :raises Exception: If there is an error during the conversion or file writing process.

        :example:

        ::

            >>> df = pandas.DataFrame({"column1": [1, 2], "column2": [3, 4]})
            >>> df_to_json(df, "output.json")
            # This will save the DataFrame as a JSON file named 'output.json'.
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

    def list_to_csv(self, data_list, output_file):
        """
        Saves a list of dictionaries to a CSV file, effectively serializing structured data for consistency, portability, and interchange.

        :method list_to_csv: Convert and store a list of dictionaries into a CSV file.
        :type list_to_csv: method

        :param data_list: A list containing dictionaries, where each dictionary represents a record with keys as column names and values as data entries.
        :type data_list: list of dict

        :param output_file: The file path where the CSV file will be saved. If a file exists at this path, it will be overwritten.
        :type output_file: str

        :raises IOError: If an error occurs during the file writing process.

        :example:

        ::

            >>> list_to_csv([{"Name": "Alice", "Age": 30}, {"Name": "Bob", "Age": 25}], "people.csv")
            # This saves the provided list of dictionaries to 'people.csv', with 'Name' and 'Age' as column headers.
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

    def df_to_csv(self, df, output_file):
        """
        Appends a DataFrame to a CSV file or creates a new one if it doesn't exist.

        This function is tailored to handle data persistence for pandas DataFrames. It intelligently checks if the specified CSV file exists. 
        If it does not, the function creates a new file and writes the DataFrame with a header. If the file exists, it appends the DataFrame 
        data to the existing file without adding another header, thus maintaining data continuity and avoiding header duplication.

        :method df_to_csv: Handles the appending of DataFrame data to a CSV file or creates a new file if necessary.
        :type df_to_csv: method

        :param output_file: The file path where the DataFrame data will be written or appended. This function takes care to avoid overwriting existing data.
        :type output_file: str

        :param df: The DataFrame that needs to be written to a CSV file.
        :type df: pandas.DataFrame

        :note: The data in the existing CSV file will not be overwritten. New data from the DataFrame will be appended to ensure data integrity.

        :example:

        ::

            >>> df = pandas.DataFrame({"column1": [1, 2], "column2": [3, 4]})
            >>> df_to_csv(df, "output.csv")
            # This will append the DataFrame to 'output.csv' if it exists, or create a new file if it does not.
        """


        if not os.path.isfile(output_file):
            df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)

    def simplify_data(self, data):
        """
        Recursively simplifies data to ensure it only contains types that are serializable.

        :method simplify_data: Convert complex data structures into serializable formats.
        :type simplify_data: method

        :param data: The data structure (such as a dictionary, list, or a basic data type) that needs to be simplified.
        :type data: dict | list | str | int | float | bool | Any

        :return: The simplified data where complex types are converted into serializable basic types.
        :rtype: dict | list | str | int | float | bool

        :example:

        ::

            >>> simplify_data({"user": {"name": "Alice", "age": 30, "preferences": ["reading", "traveling"]}})
            # This will return a simplified version of the nested dictionary, ensuring all elements are serializable.
        """

        if isinstance(data, dict):
            return {key: self.simplify_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.simplify_data(item) for item in data]
        elif isinstance(data, (str, int, float, bool)):
            return data
        else:
            return str(data)  # Convert non-serializable types to string


    def remove_json_markers(self, input_text):
        """
        Removes specific markers, typically used for JSON formatting, from the given text.

        This method is designed to clean up text by removing certain markers that are often used to denote JSON content. 
        It's particularly useful in scenarios where the text includes these markers for formatting purposes, such as in markdown or documentation, 
        and the raw text is required for further processing or analysis.

        :method remove_json_markers: Strips away specific markers from the input text.
        :type remove_json_markers: method

        :param input_text: The text from which the JSON array markers need to be removed. These markers usually include `````json`` and ````` ``.
        :type input_text: str

        :return: The cleaned text with all specified markers removed.
        :rtype: str

        :example:

        ::

            >>> example_text = "``json\\n{\"key\": \"value\"}\\n``"
            >>> remove_json_markers(example_text)
            '{"key": "value"}'
        """

        # Remove the ```json and ``` markers
        stripped_text = input_text.replace("```json", "").replace("``` json", "").replace("``` JSON", "").replace("```JSON", "").replace("```", "").strip()
        
        return stripped_text


    def extract_json_array(self, input_text):
        """
        Extracts a JSON array from a given text by removing specific markers.

        :method extract_json_array: Remove markers from text and parse it as a JSON array.
        :type extract_json_array: method

        :param input_text: The input text containing a JSON array with additional formatting markers.
        :type input_text: str

        :return: The extracted JSON array, or None if the text cannot be parsed as JSON.
        :rtype: list | None

        :raises json.JSONDecodeError: If there is an error in decoding the JSON from the provided text.

        :example:

        ::

            >>> extract_json_array('```json\n[{"name": "Alice"}, {"name": "Bob"}]\n```')
            # This will return [{'name': 'Alice'}, {'name': 'Bob'}].
        """

        # Remove the ```json and ``` markers
        stripped_text = input_text.replace("```json", "").replace("```", "").strip()

        # Parse the JSON array
        try:
            json_array = json.loads(stripped_text)
            return json_array
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

    def convert_prompt_completion_to_conversation(self, input_file, output_file, system_prompt="You are a chatbot assistant. respond to any question properly."):
        """
        Converts a JSONL file containing prompt-completion pairs into a conversational JSONL format.

        This method is particularly useful for transforming datasets with prompt and completion pairs into a format 
        suitable for conversational AI model training or evaluation. The function reads from a JSONL file where each line 
        is a JSON object with 'prompt' and 'completion' keys. It then formats these pairs into a conversational structure, 
        adding a system prompt at the beginning of each conversation, and writes this new format to an output JSONL file.

        :method convert_prompt_completion_to_conversation: Transforms prompt-completion data into a conversational format.
        :type convert_prompt_completion_to_conversation: method

        :param input_file: The file path to the input JSONL file with prompt-completion pairs.
        :type input_file: str

        :param output_file: The file path to the output JSONL file for the conversational data.
        :type output_file: str

        :param system_prompt: A standard prompt to be included as the starting message in each conversation. 
                            Defaults to a generic chatbot assistant prompt.
        :type system_prompt: str, optional

        :raises FileNotFoundError: If the specified input file cannot be found.
        :raises Exception: If there is an error during file processing.

        :example:

        ::

            >>> convert_prompt_completion_to_conversation("input.jsonl", "output.jsonl")
            >>> # This will read from 'input.jsonl' and write the conversational format to 'output.jsonl'.

        Example Input JSONL Line:
            >>> {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence."}

        Example Output JSONL Line:
            {
                "messages": [
                    {"role": "system", "content": "You are a chatbot assistant. respond to any question properly."},
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
                ]
            }
        """
        try:
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    data = json.loads(line.strip())
                    prompt = data['prompt']
                    completion = data['completion']

                    # Construct the desired JSON structure
                    new_data = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion}
                        ]
                    }

                    # Write the new JSON structure to the output file
                    outfile.write(json.dumps(new_data) + '\n')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {input_file}") from e
        except Exception as e:
            raise Exception(f"Error occurred during conversion: {e}") from e



    def convert_conversation_to_prompt_completion(self, input_file, output_file):
        """
        Converts a conversational JSONL file to a format with prompt-completion pairs.

        :method convert_conversation_to_prompt_completion: Transform a conversational JSONL file, 
        extracting user messages as prompts and assistant responses as completions, 
        and save them in an output JSONL file.
        :type convert_conversation_to_prompt_completion: method

        :param input_file: The path to the input conversational JSONL file containing messages.
        :type input_file: str

        :param output_file: The path to the output JSONL file where prompt-completion pairs will be saved.
        :type output_file: str

        :return: None. The function writes the extracted prompt-completion pairs to the output file.
        :rtype: None

        :raises FileNotFoundError: If the input file specified does not exist.
        :raises Exception: If any error occurs during the file reading or writing process.

        :example:

        ::

            # Given an input JSONL file with conversational data:
            # {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence."}]}
            # Running convert_conversation_to_prompt_completion will produce an output JSONL file containing:
            # {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence."}
        """

        try:
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    data = json.loads(line.strip())
                    messages = data['messages']

                    # Extract prompt and completion
                    prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
                    completion = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), None)

                    if prompt and completion:
                        # Construct and write the prompt-completion pair
                        new_data = {"prompt": prompt, "completion": completion}
                        outfile.write(json.dumps(new_data) + '\n')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {input_file}") from e
        except Exception as e:
            raise Exception(f"Error occurred during conversion: {e}") from e



    def convert_prompt_completion_to_llama2_instructions(self, input_file, output_jsonl_file, output_json_file):
        """
        Converts prompt-completion data from a JSONL file to the LLAMA2 instructions format.

        This function is designed to read a JSONL file containing prompt-completion pairs and convert each record 
        into the LLAMA2 instructions format, which is a structured way to represent AI training data. 
        It generates two output files: one in JSONL format and another in JSON format, each containing the converted data.

        :method convert_prompt_completion_to_llama2_instructions: Transforms data into LLAMA2 instructions format for AI training.
        :type convert_prompt_completion_to_llama2_instructions: method

        :param input_file: The file path to the input JSONL file with prompt-completion pairs.
        :type input_file: str

        :param output_jsonl_file: The file path to the output JSONL file which will contain the LLAMA2 formatted data.
        :type output_jsonl_file: str

        :param output_json_file: The file path to the output JSON file which will also contain the LLAMA2 formatted data.
        :type output_json_file: str

        :raises FileNotFoundError: If the specified input file cannot be found.
        :raises Exception: If there is an error during the conversion process or file writing.

        :example:

        ::

            >>> convert_prompt_completion_to_llama2_instructions("input.jsonl", "output.llama2.jsonl", "output.llama2.json")
            # This will read prompt-completion pairs from 'input.jsonl' and write the LLAMA2 formatted data to 'output.llama2.jsonl' and 'output.llama2.json'.

        :Example Input JSONL Record:
            >>> {"prompt": "Explain how a computer works.", "completion": "A computer is a machine that processes information."}

        :Example Output LLAMA2 JSONL Record:
            >>> {
            >>>     "instruction": "Explain how a computer works.",
            >>>     "input": "",
            >>>     "output": "A computer is a machine that processes information."
            >>> }
        """

        try:
            # Step 1: Read the original records from a JSONL file
            original_records = []
            with open(input_file, "r") as f:
                for line in f:
                    original_records.append(json.loads(line))

            # Step 2: Convert each record to the new format
            new_records = []
            for record in original_records:
                new_record = {
                    "instruction": record["prompt"].strip(" \n###"),
                    "input": "",
                    "output": record["completion"].strip(" END")
                }
                new_records.append(new_record)

            # Ensure output directories exist
            os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)
            os.makedirs(os.path.dirname(output_json_file), exist_ok=True)


            # Step 3: Write the new records to a new JSONL file
            with open(output_jsonl_file, "w") as f:
                for record in new_records:
                    f.write(json.dumps(record) + '\n')

            # Step 4: Write the new records to a new JSON file
            with open(output_json_file, "w") as f:
                json.dump(new_records, f, indent=4)

            return new_records

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {input_file}") from e
        except Exception as e:
            raise Exception(f"Error occurred during conversion: {e}") from e

    import json

    def convert_llama2_instructions_to_prompt_completion(self, input_file, output_file):
        """
        Converts data from the LLAMA2 instructions format to the original prompt-completion format.

        :method convert_llama2_instructions_to_prompt_completion: Transform data from LLAMA2 instructions format into prompt-completion pairs, and save them in a JSONL file.
        :type convert_llama2_instructions_to_prompt_completion: method

        :param input_file: The path to the input JSONL file in LLAMA2 instructions format.
        :type input_file: str

        :param output_file: The path to the output JSONL file where the converted prompt-completion pairs will be saved.
        :type output_file: str

        :return: None. The function writes the converted data to the output file.
        :rtype: None

        :raises FileNotFoundError: If the input file specified does not exist.
        :raises Exception: If any error occurs during the file reading or writing process.

        :example:

        ::

            # Given an input JSONL file in LLAMA2 format:
            # {"instruction": "Explain what AI is.", "output": "AI stands for Artificial Intelligence."}
            # Running convert_llama2_instructions_to_prompt_completion will produce an output JSONL file containing:
            # {"prompt": "Explain what AI is.", "completion": "AI stands for Artificial Intelligence."}
        """

        try:
            # Read the original records from the LLAMA2 format JSONL file
            llama2_records = []
            with open(input_file, "r") as f:
                for line in f:
                    llama2_records.append(json.loads(line))

            # Convert each record to the original prompt-completion format
            original_records = []
            for record in llama2_records:
                original_record = {
                    "prompt": record["instruction"],
                    "completion": record["output"] 
                }
                original_records.append(original_record)

            # Write the original records to a new JSONL file
            with open(output_file, "w") as f:
                for record in original_records:
                    f.write(json.dumps(record) + '\n')

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {input_file}") from e
        except Exception as e:
            raise Exception(f"Error occurred during conversion: {e}") from e


    def json_to_yaml(self, json_data):
        """
        Converts a JSON object or string to a YAML-formatted string.

        This function is designed to take a JSON object, either in the form of a Python dictionary or as a JSON-formatted string, 
        and converts it into a YAML-formatted string. The conversion preserves the original structure and data types from the JSON input.

        :method json_to_yaml: Converts JSON data to YAML format.
        :type json_to_yaml: method

        :param json_data: The JSON data to be converted. It can be a dictionary representing the JSON object or a string containing JSON-formatted text.
        :type json_data: dict or str

        :return: A string containing the data in YAML format.
        :rtype: str

        :note: The function utilizes PyYAML's `dump` method for conversion, which generates block-style YAML formatting by default and does not sort keys to maintain the order.

        :example:

        ::

            >>> json_obj = {"name": "John", "age": 30, "city": "New York"}
            >>> yaml_str = json_to_yaml(json_obj)
            >>> print(yaml_str)
            name: John
            age: 30
            city: New York

            >>> json_str = '{"name": "John", "age": 30, "city": "New York"}'
            >>> yaml_str = json_to_yaml(json_str)
            >>> print(yaml_str)
            name: John
            age: 30
            city: New York
        """

        # Parse the JSON string to a dictionary if it's in string format
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        # Convert the dictionary to a YAML string
        yaml_str = yaml.dump(json_data, default_flow_style=False, sort_keys=False)
        return yaml_str


    def yaml_to_json(self, yaml_data):
        """
        Converts a YAML object or string to a JSON-formatted string.

        This function is designed to take a YAML object, either as a Python dictionary resulting from YAML parsing or as a YAML-formatted string, 
        and converts it into a JSON-formatted string. The conversion retains the original structure and data types present in the input YAML.

        :method yaml_to_json: Converts YAML data to a JSON format.
        :type yaml_to_json: method

        :param yaml_data: The YAML data to be converted, provided as a string.
        :type yaml_data: str

        :return: A dictionary representing the JSON data.
        :rtype: dict

        :raises ValueError: If the YAML data does not represent a valid dictionary.

        :example:

        ::

            >>> yaml_str = "name: John\\nage: 30\\ncity: New York"
            >>> json_obj = yaml_to_json(yaml_str)
            >>> print(json_obj)
            {'name': 'John', 'age': 30, 'city': 'New York'}

        :note: The function uses PyYAML's `safe_load` method to parse the YAML data and convert it into a dictionary.
        """

    # Function implementation goes here

        # Function implementation goes here


        # Parse the YAML string to a dictionary
        yaml_dict = yaml.safe_load(yaml_data)

        # Ensure that the result is a dictionary
        if isinstance(yaml_dict, dict):
            return yaml_dict
        else:
            raise ValueError("Failed to convert YAML to JSON. The YAML data does not represent a valid dictionary.")


    def json_to_csv(self, json_data, csv_file_path):
        """
        Converts JSON data into a CSV file.

        This function is specifically designed to take JSON data, represented as a list of dictionaries, and convert it into a CSV file. 
        This can be particularly useful for data transformation tasks where JSON data needs to be presented or analyzed in tabular form.

        :method json_to_csv: Transforms JSON data into a CSV file.
        :type json_to_csv: method

        :param json_data: A list of dictionaries, where each dictionary represents a row of data to be converted into CSV format.
        :type json_data: list of dict

        :param csv_file_path: The file path where the resulting CSV file will be saved.
        :type csv_file_path: str

        :example:

        ::

            >>> json_data = [
                {"name": "John", "age": 30, "city": "New York"},
                {"name": "Alice", "age": 25, "city": "Los Angeles"},
                {"name": "Bob", "age": 35, "city": "Chicago"}
            ]
            >>> json_to_csv(json_data, "data.csv")
            # This will create a CSV file 'data.csv' with the data from 'json_data'.
        """

        # Create a DataFrame from the JSON data
        df = pd.DataFrame(json_data)

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)


    def csv_to_json(csv_file_path, json_file_path):
        """
        Converts data from a CSV file to a JSON file.

        This function is designed to read data from a CSV file and convert it into a JSON format, saving the converted data to a JSON file. 
        It's useful for data format conversion, especially when dealing with data transformation and integration tasks in data processing and machine learning workflows.

        :method csv_to_json: Transforms data from CSV format into JSON format.
        :type csv_to_json: method

        :param csv_file_path: The file path to the CSV file containing the data to be converted.
        :type csv_file_path: str

        :param json_file_path: The file path where the resulting JSON file will be saved.
        :type json_file_path: str

        :example:

        ::

            >>> csv_to_json("data.csv", "data.json")
            # This will read data from 'data.csv' and convert it to JSON format, saving the result in 'data.json'.

        :note: The function reads the CSV data into a pandas DataFrame and then converts it to JSON using pandas' `to_json` method.
        """

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to JSON and save it to a JSON file
        df.to_json(json_file_path, orient="records", lines=True)


    def csv_to_yaml(self, csv_file_path, yaml_file_path):
        """
        Converts CSV data to a YAML file format.

        :method csv_to_yaml: Read data from a CSV file and convert it into a YAML file format.
        :type csv_to_yaml: method

        :param csv_file_path: The file path where the CSV data is located.
        :type csv_file_path: str

        :param yaml_file_path: The file path where the converted YAML data will be saved.
        :type yaml_file_path: str

        :return: None. The function writes the converted YAML data to the specified file.
        :rtype: None

        :raises FileNotFoundError: If the CSV file specified does not exist.
        :raises Exception: If any error occurs during the file reading or conversion process.

        :example:

        ::

            >>> csv_to_yaml("data.csv", "data.yaml")
            # This will read 'data.csv', convert its contents to YAML format, and save it to 'data.yaml'.
        """

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")

        # Write the data to a YAML file
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)


    def yaml_to_csv(self, yaml_file_path, csv_file_path):
        """
        Converts YAML data to a CSV file.

        This function is specifically designed to read data from a YAML file and convert it into a CSV format, subsequently saving the converted data to a CSV file. This is useful in scenarios where YAML-formatted data needs to be represented in a tabular format for easier analysis or integration with other data processing tools.

        :method yaml_to_csv: Transforms data from YAML format into CSV format.
        :type yaml_to_csv: method

        :param yaml_file_path: The file path to the YAML file containing the data to be converted.
        :type yaml_file_path: str

        :param csv_file_path: The file path where the resulting CSV file will be saved.
        :type csv_file_path: str

        :example:

        ::

            >>> yaml_to_csv("data.yaml", "data.csv")
            # This will read data from 'data.yaml', convert it to CSV format, and save the result in 'data.csv'.

        :note: The function reads the YAML data, converts it into a suitable structure (like a pandas DataFrame), and then writes it to a CSV file.
        """

        # Read the YAML file into a list of dictionaries
        with open(yaml_file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)



    def xml_to_csv(self, xml_file_path, csv_file_path):
        """
        Converts XML data to CSV format.

        :method xml_to_csv: Parse an XML file, extract its data, and save it in CSV format.
        :type xml_to_csv: method

        :param xml_file_path: The file path where the XML data is located.
        :type xml_file_path: str

        :param csv_file_path: The file path where the converted CSV data will be saved.
        :type csv_file_path: str

        :return: None. The function writes the converted CSV data to the specified file.
        :rtype: None

        :raises FileNotFoundError: If the XML file specified does not exist.
        :raises Exception: If any error occurs during the file parsing or conversion process.

        :example:

        ::

            >>> xml_to_csv("data.xml", "output.csv")
            # This will read 'data.xml', extract its contents, and save it to 'output.csv' in CSV format.

        :notes:
        
        - This function utilizes the `xml.etree.ElementTree` library for parsing the XML file.
        - The XML elements are transformed into CSV rows, with the header row in the CSV derived from the XML element names.

        :see_also:

        - csv_to_xml: A function to convert CSV data to XML format.
        """

        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Create a CSV file and write the header
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header row based on the XML element names
            header = [element.tag for element in root[0]]
            csv_writer.writerow(header)

            # Write the data rows
            for record in root:
                row_data = [element.text for element in record]
                csv_writer.writerow(row_data)


    

    def csv_to_jsonl(self, csv_file_path, jsonl_file_path):
        """
        Converts data from a CSV file to a JSON Lines (JSONL) file.

        This function is designed to read data from a CSV file and convert it into the JSON Lines format, which is a JSON format where each line is a separate JSON object. This is particularly useful for handling large datasets or stream processing, as it allows for efficient data processing line by line.

        :method csv_to_jsonl: Transforms data from CSV format into JSON Lines format.
        :type csv_to_jsonl: method

        :param csv_file_path: The file path to the CSV file containing the data to be converted.
        :type csv_file_path: str

        :param jsonl_file_path: The file path where the resulting JSON Lines file will be saved.
        :type jsonl_file_path: str

        :example:

        ::

            >>> csv_to_jsonl("data.csv", "data.jsonl")
            # This will read data from 'data.csv' and convert it to JSON Lines format, saving the result in 'data.jsonl'.

        :note: The function reads the CSV data into a pandas DataFrame and then converts it to JSON Lines using pandas' `to_json` method with the `orient="records"` and `lines=True` parameters.
        """

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to JSON Lines and save it to a file
        df.to_json(jsonl_file_path, orient="records", lines=True)

    def jsonl_to_csv(self, jsonl_file_path, csv_file_path):
        """
        Converts JSON Lines (JSONL) data to a CSV file format.

        :method jsonl_to_csv: Read data from a JSONL file and convert it into a CSV file format.
        :type jsonl_to_csv: method

        :param jsonl_file_path: The file path where the JSONL data is located.
        :type jsonl_file_path: str

        :param csv_file_path: The file path where the converted CSV data will be saved.
        :type csv_file_path: str

        :return: None. The function writes the converted CSV data to the specified file.
        :rtype: None

        :raises FileNotFoundError: If the JSONL file specified does not exist.
        :raises Exception: If any error occurs during the file reading or conversion process.

        :example:

        ::

            >>> jsonl_to_csv("data.jsonl", "data.csv")
            # This will read 'data.jsonl', convert its contents to CSV format, and save it to 'data.csv'.
        """

        # Read the JSON Lines file into a DataFrame
        df = pd.read_json(jsonl_file_path, lines=True)

        # Convert the DataFrame to CSV and save it to a file
        df.to_csv(csv_file_path, index=False)
