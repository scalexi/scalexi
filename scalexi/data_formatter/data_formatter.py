import json
import yaml
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os

def convert_prompt_completion_to_conversation(input_file, output_file, system_prompt="You are a chatbot assistant. respond to any question properly."):
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



def convert_conversation_to_prompt_completion(input_file, output_file):
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



def convert_prompt_completion_to_llama2_instructions(input_file, output_jsonl_file, output_json_file):
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

def convert_llama2_instructions_to_prompt_completion(input_file, output_file):
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


def json_to_yaml(json_data):
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


def yaml_to_json(yaml_data):
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


def json_to_csv(json_data, csv_file_path):
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


def csv_to_yaml(csv_file_path, yaml_file_path):
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


def yaml_to_csv(yaml_file_path, csv_file_path):
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



def xml_to_csv(xml_file_path, csv_file_path):
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


def xml_to_csv(xml_file_path, csv_file_path):
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
