import os
import logging

class Logger:
    def __init__(self):
        """
        Initializes the Logger object, configuring the logging settings based on environment variables.

        :method __init__: Sets up logging configuration using an environment variable for the logging level, with a default level of 'WARNING'.
        :type __init__: method

        This method reads the logging level from an environment variable named 'LOGGING_LEVEL'. If this variable is not set or contains an invalid value, it defaults to 'WARNING'. The logging format is set to include timestamps, logging level, and the message.

        :return: None.
        :rtype: None
        """
        # Read logging level from environment variable
        logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

        # Configure logging with the level from the environment variable
        logging.basicConfig(
            level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create and return a logger object
        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        """
        Retrieves the logger instance.

        :method get_logger: Returns the configured logger object for logging messages.
        :type get_logger: method

        This method provides access to the logger instance created during the initialization of the Logger class. It allows for logging messages with the configured logging level and format.

        :return: The logger instance.
        :rtype: logging.Logger
        """

        return self.logger
    
    def setLevel(self, level):
        """
        Sets the logging level of the logger instance.

        :method setLevel: Updates the logging level of the logger to the specified level.
        :type setLevel: method

        :param level: The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        :type level: str

        This method attempts to set the logger's level to the specified level. If the provided level is invalid, it logs a warning message and retains the current logging level.

        :raises AttributeError: If an invalid logging level is specified.
        
        :return: None.
        :rtype: None

        :example:

        ::

            >>> logger = Logger()
            >>> logger.setLevel("INFO")
            # Sets the logging level of the logger to 'INFO'.
        """
        try:
            self.logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            self.logger.warning("Invalid logging level: %s. Keeping the current level.", level)
