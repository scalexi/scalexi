import json
import yaml
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
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

class DataFormatter:
    def __init__(self):
        pass

    def format_prompt_completion(self, prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
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

    def format_prompt_completion_df(self, prompt, completion, start_sequence="\n\n###\n\n", end_sequence="END"):
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

    def df_to_json(self, df, json_output):
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

    def list_to_csv(self, data_list, output_file):
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

    def df_to_csv(self, df, output_file):
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

    def simplify_data(self, data):
        """
        Recursively simplify data to ensure it only contains serializable types.
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
        Removes specific markers from a given text.
        
        Args:
            input_text (str): The input text containing the JSON array with additional markers.
        
        Returns:
            str: The text with the markers removed.
        """
        # Remove the ```json and ``` markers
        stripped_text = input_text.replace("```json", "").replace("```", "").strip()
        return stripped_text

    def extract_json_array(self, input_text):
        """
        Extracts a JSON array from a given text by removing specific markers.
        
        Args:
            input_text (str): The input text containing the JSON array with additional markers.
        
        Returns:
            list: The extracted JSON array.
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
        Converts a JSONL file with prompt-completion pairs to a conversational JSONL format.

        This function reads a JSONL file where each line contains a JSON object with 'prompt' and 'completion' keys.
        It then constructs a new JSON object with a 'messages' key, where the value is a list of messages in a
        conversational format, suitable for use with conversational AI models.

        Parameters:
        - input_file (str): The path to the input JSONL file containing prompt-completion pairs.
        - output_file (str): The path to the output JSONL file that will contain the conversational data.
        - system_prompt (str): The system prompt to be included in the conversation.

        Returns:
        - None: The function writes the conversational JSON data to the output file.

        Raises:
        - FileNotFoundError: If the input file does not exist.
        - Exception: If an error occurs during file reading or writing.
        
        Example:
        Input JSONL line:
        {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence."}

        Output JSONL line:
        {
            "messages": [
                {"role": "system", "content": "You are PSUGPT, the ChatBot Assistant at Prince Sultan University..."},
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
        Converts a conversational JSONL file back to a format with prompt-completion pairs.

        This function reads a conversational JSONL file where each line contains a 'messages' key with a list of messages.
        It extracts the user's message as the prompt and the assistant's response as the completion, and writes them to
        an output JSONL file in a prompt-completion format.

        Parameters:
        - input_file (str): The path to the input conversational JSONL file.
        - output_file (str): The path to the output JSONL file for prompt-completion pairs.

        Returns:
        - None: The function writes the prompt-completion data to the output file.

        Raises:
        - FileNotFoundError: If the input file does not exist.
        - Exception: If an error occurs during file reading or writing.
        
        Example:
        Input JSONL line:
        {
            "messages": [
                {"role": "system", "content": "You are a chatbot assistant..."},
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        }

        Output JSONL line:
        {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence."}
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
        Converts prompt-completion data from a JSONL file to LLAMA2 instructions format.

        The function reads a JSONL file with prompt-completion pairs and converts each record to the LLAMA2 instructions
        format. It writes the new records to both a JSONL file and a JSON file.

        Parameters:
        - input_file (str): The path to the input JSONL file with prompt-completion pairs.
        - output_jsonl_file (str): The path to the output JSONL file for LLAMA2 instructions.
        - output_json_file (str): The path to the output JSON file for LLAMA2 instructions.

        Returns:
        - None: The function writes the converted data to the output files.

        Raises:
        - FileNotFoundError: If the input file does not exist.
        - Exception: If an error occurs during file reading or writing.
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
        Converts data from LLAMA2 instructions format to the original prompt-completion format.

        This function reads a JSONL file where each record is in the LLAMA2 instructions format and converts
        them back to the original format with prompt and completion pairs.

        Parameters:
        - input_file (str): The path to the input JSONL file in LLAMA2 instructions format.
        - output_file (str): The path to the output JSONL file for prompt-completion pairs.

        Returns:
        - None: The function writes the converted data to the output file.

        Raises:
        - FileNotFoundError: If the input file does not exist.
        - Exception: If an error occurs during file reading or writing.
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
        Convert a JSON object or string to a YAML-formatted string.

        This function takes a JSON object, which can either be a Python dictionary or a JSON-formatted string, and converts it into a YAML-formatted string. The conversion maintains the original structure and data types present in the input JSON.

        Parameters
        ----------
        json_data : dict or str
            The JSON data to convert. This can either be a dictionary representing the JSON object or a string containing the JSON-formatted text.

        Returns
        -------
        str
            A string containing the YAML-formatted data.

        Examples
        --------
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

        Notes
        -----
        The function uses PyYAML's dump method, which by default generates YAML with block-style formatting. It also turns off key sorting to preserve the order of keys.
        """
        # Parse the JSON string to a dictionary if it's in string format
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        # Convert the dictionary to a YAML string
        yaml_str = yaml.dump(json_data, default_flow_style=False, sort_keys=False)
        return yaml_str


    def yaml_to_json(self, yaml_data):
        """
    Convert a YAML object or string to a JSON-formatted string.

        This function takes a YAML object, which can either be a Python dictionary resulting from YAML parsing 
        or a YAML-formatted string, and converts it into a JSON-formatted string. The conversion maintains the 
        original structure and data types present in the input YAML.

        Parameters
        ----------
        yaml_data : str
            The YAML data to convert as a string.

        Returns
        -------
        dict
            A dictionary representing the JSON data.

        Examples
        --------
        >>> yaml_str = "name: John\nage: 30\ncity: New York"
        >>> json_obj = yaml_to_json(yaml_str)
        >>> print(json_obj)
        {'name': 'John', 'age': 30, 'city': 'New York'}

        Notes
        -----
        The function uses PyYAML's load method to parse the YAML data and convert it into a dictionary.
        """
        # Parse the YAML string to a dictionary
        yaml_dict = yaml.safe_load(yaml_data)

        # Ensure that the result is a dictionary
        if isinstance(yaml_dict, dict):
            return yaml_dict
        else:
            raise ValueError("Failed to convert YAML to JSON. The YAML data does not represent a valid dictionary.")


    def json_to_csv(self, json_data, csv_file_path):
        """
        Convert JSON data to a CSV file.

        This function takes JSON data and converts it into a CSV file.

        Parameters
        ----------
        json_data : list of dict
            A list of dictionaries representing the JSON data.

        csv_file_path : str
            The path to the CSV file where the data will be saved.

        Examples
        --------
        >>> json_data = [
        ...     {"name": "John", "age": 30, "city": "New York"},
        ...     {"name": "Alice", "age": 25, "city": "Los Angeles"},
        ...     {"name": "Bob", "age": 35, "city": "Chicago"}
        ... ]
        >>> json_to_csv(json_data, "data.csv")
        """
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(json_data)

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)


    def csv_to_json(csv_file_path, json_file_path):
        """
        Convert CSV data to a JSON file.

        This function takes CSV data from a file and converts it into a JSON file.

        Parameters
        ----------
        csv_file_path : str
            The path to the CSV file containing the data.

        json_file_path : str
            The path to the JSON file where the data will be saved.

        Examples
        --------
        >>> csv_to_json("data.csv", "data.json")
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to JSON and save it to a JSON file
        df.to_json(json_file_path, orient="records", lines=True)


    def csv_to_yaml(self, csv_file_path, yaml_file_path):
        """
        Convert CSV data to a YAML file.

        This function takes CSV data from a file and converts it into a YAML file.

        Parameters
        ----------
        csv_file_path : str
            The path to the CSV file containing the data.

        yaml_file_path : str
            The path to the YAML file where the data will be saved.

        Examples
        --------
        >>> csv_to_yaml("data.csv", "data.yaml")
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
        Convert YAML data to a CSV file.

        This function takes YAML data from a file and converts it into a CSV file.

        Parameters
        ----------
        yaml_file_path : str
            The path to the YAML file containing the data.

        csv_file_path : str
            The path to the CSV file where the data will be saved.

        Examples
        --------
        >>> yaml_to_csv("data.yaml", "data.csv")
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
        Convert XML data to CSV format.

        This function parses an XML file, extracts the data, and saves it in CSV format.

        Parameters
        ----------
        xml_file_path : str
            The path to the XML file containing the data.

        csv_file_path : str
            The path to the CSV file where the data will be saved.

        Examples
        --------
        >>> xml_to_csv("data.xml", "output.csv")

        Notes
        -----
        - This function uses the `xml.etree.ElementTree` library to parse the XML file.
        - The XML elements are converted into CSV rows.
        - The header row in the CSV file is based on the XML element names.

        See Also
        --------
        csv_to_xml: Convert CSV data to XML format.

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


    def xml_to_csv(self, xml_file_path, csv_file_path):
        """
        Convert XML data to CSV format.

        This function parses an XML file, extracts the data, and saves it in CSV format.

        :param xml_file_path: The path to the XML file containing the data.
        :type xml_file_path: str
        :param csv_file_path: The path to the CSV file where the data will be saved.
        :type csv_file_path: str

        :Example:

        >>> xml_to_csv("data.xml", "output.csv")

        :Notes:

        - This function uses the `xml.etree.ElementTree` library to parse the XML file.
        - The XML elements are converted into CSV rows.
        - The header row in the CSV file is based on the XML element names.

        :See Also:

        - :func:`csv_to_xml`: Convert CSV data to XML format.

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
        Convert CSV data to a JSON Lines file.

        This function takes CSV data from a file and converts it into a JSON Lines file.

        Parameters
        ----------
        csv_file_path : str
            The path to the CSV file containing the data.

        jsonl_file_path : str
            The path to the JSON Lines file where the data will be saved.

        Examples
        --------
        >>> csv_to_jsonl("data.csv", "data.jsonl")
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to JSON Lines and save it to a file
        df.to_json(jsonl_file_path, orient="records", lines=True)

    def jsonl_to_csv(self, jsonl_file_path, csv_file_path):
        """
        Convert JSON Lines data to a CSV file.

        This function takes data from a JSON Lines file and converts it into a CSV file.

        Parameters
        ----------
        jsonl_file_path : str
            The path to the JSON Lines file containing the data.
        
        csv_file_path : str
            The path to the CSV file where the data will be saved.

        Examples
        --------
        >>> jsonl_to_csv("data.jsonl", "data.csv")
        """
        # Read the JSON Lines file into a DataFrame
        df = pd.read_json(jsonl_file_path, lines=True)

        # Convert the DataFrame to CSV and save it to a file
        df.to_csv(csv_file_path, index=False)
