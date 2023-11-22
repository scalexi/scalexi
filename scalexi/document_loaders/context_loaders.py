import pandas as pd

class ContextExtractor:
    """
    A class to extract specific columns from a CSV file as pandas Series or DataFrame.

    This class provides methods to read a specified column from a CSV file and return it as either a pandas Series or DataFrame. 
    It's useful for data processing tasks where only specific column data is required from a larger dataset.

    :method from_csv_as_series: Reads a CSV file and returns the specified column as a pandas Series.
    :type from_csv_as_series: method

    :method from_csv_as_df: Reads a CSV file and returns the specified column as a pandas DataFrame.
    :type from_csv_as_df: method
    """

    def from_csv_as_series(self, csv_file_path, column_name="context", encoding="utf-8"):
        """
        Reads a CSV file and returns the specified column as a pandas Series.

        This method is designed to extract a single column from a CSV file and present it as a pandas Series, 
        which can be useful for further data analysis or processing.

        :param csv_file_path: The path to the CSV file.
        :type csv_file_path: str

        :param column_name: The name of the column to extract. Default is "context".
        :type column_name: str, optional

        :param encoding: The encoding of the CSV file. Default is "utf-8".
        :type encoding: str, optional

        :return: The specified column as a pandas Series.
        :rtype: pandas.Series

        :raises FileNotFoundError: If the CSV file does not exist.
        :raises ValueError: If the specified column does not exist in the CSV.
        :raises Exception: For any other exceptions that may occur.

        :example:

        ::

            >>> extractor = ContextExtractor()
            >>> series = extractor.from_csv_as_series("data.csv", "context")
        """

        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{csv_file_path}' was not found.")
        except Exception as e:
            raise Exception(f"Error processing the file: {e}")

        if column_name in df.columns:
            return df[column_name]
        else:
            raise ValueError(f"The column '{column_name}' does not exist in the CSV.")

    def from_csv_as_df(self, csv_file_path, column_name="context", encoding="utf-8"):
        """
        Reads a CSV file and returns the specified column as a pandas DataFrame.

        This method extracts a single column from a CSV file and presents it as a pandas DataFrame. 
        It's particularly useful when only one column of data is needed for analysis or processing.

        :param csv_file_path: The path to the CSV file.
        :type csv_file_path: str

        :param column_name: The name of the column to extract. Default is "context".
        :type column_name: str, optional

        :param encoding: The encoding of the CSV file. Default is "utf-8".
        :type encoding: str, optional

        :return: The specified column as a pandas DataFrame.
        :rtype: pandas.DataFrame

        :raises FileNotFoundError: If the CSV file does not exist.
        :raises ValueError: If the specified column does not exist in the CSV.
        :raises Exception: For any other exceptions that may occur.

        :example:

        ::

            >>> extractor = ContextExtractor()
            >>> dataframe = extractor.from_csv_as_df("data.csv", "context")
        """

        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{csv_file_path}' was not found.")
        except Exception as e:
            raise Exception(f"Error processing the file: {e}")

        if column_name in df.columns:
            return df[[column_name]]
        else:
            raise ValueError(f"The column '{column_name}' does not exist in the CSV.")
