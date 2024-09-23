import os
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Logger:
    """
    A class for creating and configuring loggers with the flexibility
    to log to a console, a file, or both based on environment variables,
    with daily rotation of the log file.
    
    The logger's level and output destination(s) are configurable through
    environment variables: LOGGING_LEVEL and LOGGING_OUTPUT, loaded from a .env file.
    """

    def __init__(self):
        """
        Initializes the Logger instance by configuring the logging based on
        environment variables loaded from a .env file. The default logging level
        is WARNING, and the default output is console.
        """
        # Clear existing handlers
        logging.getLogger().handlers = []

        # Read logging level from environment variable
        logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()
        logging_output = os.getenv('LOGGING_OUTPUT', 'console').lower()

        handlers = []
        log_format = ('[LOGGER] %(asctime)s - %(name)s - [%(threadName)s] '
                      '%(levelname)s in %(filename)s:%(lineno)d:%(funcName)s - %(message)s')
        simple_log_format = '[LOGGER] %(asctime)s - %(levelname)s - %(message)s'

        # Configure handlers based on environment variable
        if logging_output in ('file', 'both'):
            # Timed rotating file handler (rotates logs daily)
            file_handler = TimedRotatingFileHandler(
                'app.log', # Log file name
                when='midnight', # Rotate logs at midnight
                interval=1, # Rotate logs every day
                backupCount=30  # Keep 30 days of logs
            )
            file_handler.suffix = "%Y-%m-%d"  # Append the date to the filename
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)

        if logging_output in ('console', 'both'):
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(simple_log_format))
            handlers.append(console_handler)

        # Create and configure a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, logging_level, logging.WARNING))
        
        # Remove all handlers associated with the logger instance
        #while self.logger.hasHandlers():
        #    self.logger.removeHandler(self.logger.handlers[0])
            
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)


        # Add new handlers
        for handler in handlers:
            self.logger.addHandler(handler)

    def get_logger(self):
        """
        Returns the logger instance.
        
        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger

    def set_level(self, level):
        """
        Sets the logging level of the logger.
        
        Parameters:
            level (str): The logging level to set (e.g., 'DEBUG', 'INFO').
        """
        try:
            self.logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            self.logger.warning("Invalid logging level: %s. Keeping the current level.", level)
