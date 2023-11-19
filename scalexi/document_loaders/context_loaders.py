import pandas as pd

class ContextExtractor:
    """
    A class to extract specific columns from a CSV file as pandas Series or DataFrame.

    Attributes:
    ----------
    None

    Methods:
    -------
    from_csv_as_series(csv_file_path, column_name="context", encoding="utf-8")
        Reads a CSV file and returns the specified column as a pandas Series.
    from_csv_as_df(csv_file_path, column_name="context", encoding="utf-8")
        Reads a CSV file and returns the specified column as a pandas DataFrame.
    """

    def from_csv_as_series(self, csv_file_path, column_name="context", encoding="utf-8"):
        """
        Reads a CSV file and returns the specified column as a pandas Series.

        Parameters:
        ----------
        csv_file_path : str
            The path to the CSV file.
        column_name : str, optional
            The name of the column to extract. Default is "context".
        encoding : str, optional
            The encoding of the CSV file. Default is "utf-8".

        Returns:
        -------
        pandas.Series
            The specified column as a Series.

        Raises:
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If the specified column does not exist in the CSV.
        Exception
            For any other exceptions that may occur.

        Example:
        -------
        >>> extractor = CSVContextExtractor()
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

        Parameters:
        ----------
        csv_file_path : str
            The path to the CSV file.
        column_name : str, optional
            The name of the column to extract. Default is "context".
        encoding : str, optional
            The encoding of the CSV file. Default is "utf-8".

        Returns:
        -------
        pandas.DataFrame
            The specified column as a DataFrame.

        Raises:
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If the specified column does not exist in the CSV.
        Exception
            For any other exceptions that may occur.

        Example:
        -------
        >>> extractor = CSVContextExtractor()
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
