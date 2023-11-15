import pandas as pd

def context_from_csv_as_series(csv_file_path, column_name="context", encoding="utf-8"):
    """
    Reads a CSV file and returns the specified column as a pandas Series.

    Parameters:
    - csv_file_path (str): The path to the CSV file.
    - column_name (str): The name of the column to extract. Default is "context".

    Returns:
    - pandas.Series: The specified column as a Series.

    Raises:
    - ValueError: If the specified column does not exist in the CSV.
    - FileNotFoundError: If the CSV file does not exist.
    """
    
    try:
        df = pd.read_csv(csv_file_path, encoding=encoding)
    except FileNotFoundError as e:
        print(f"[FileNotFoundError]: File '{csv_file_path}' not found. Check the file path and file name", e)
        raise FileNotFoundError(f"The file '{csv_file_path}' was not found.")
    
    if column_name in df.columns:
        return df[column_name]
    else:
        print(f"[ValueError]: The CSV file does not have a '{column_name}' column.")
        raise ValueError(f"The CSV file does not have a '{column_name}' column.")

def context_from_csv_as_df(csv_file_path, column_name="context", encoding="utf-8"):
    """
    Reads a CSV file and returns the specified column as a pandas DataFrame. 

    Parameters:
    - csv_file_path (str): The path to the CSV file.
    - column_name (str): The name of the column to extract. Default is "context".

    Returns:
    - pandas.DataFrame: The specified column as a DataFrame.

    Raises:
    - ValueError: If the specified column does not exist in the CSV.
    - FileNotFoundError: If the CSV file does not exist.
    """
    
    try:
        df = pd.read_csv(csv_file_path, encoding=encoding)
    except FileNotFoundError as e:
        print(f"[FileNotFoundError]: File '{csv_file_path}' not found. Check the file path and file name", e)
        raise FileNotFoundError(f"The file '{csv_file_path}' was not found.")
    except Exception as e: 
        print(f"[Exception]: Error opening '{csv_file_path}'. Error Type: '{e}'")

    
    if column_name in df.columns:
        return df[[column_name]]
    else:
        print(f"[ValueError]: The CSV file does not have a '{column_name}' column.")
        raise ValueError(f"The CSV file does not have a '{column_name}' column.")
