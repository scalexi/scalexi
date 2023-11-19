from openai import OpenAI
import openai
from typing import Optional
from typing import Union
import os
from typing import List, Dict
import logging
import httpx 

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

class FineTuningAPI:
    def __init__(self, openai_key=None, enable_timeouts= False, timeouts_options= None):
        self.openai_api_key = openai_key if openai_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key.")
        self.client = OpenAI(api_key=self.openai_api_key, max_retries=3)
        if enable_timeouts:
            if timeouts_options is None:
                timeouts_options = {"total": 120, "read": 60.0, "write": 60.0, "connect": 10.0}
                self.client = self.client.with_options(timeout=httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0))
            else:
                self.client = self.client.with_options(timeout=httpx.Timeout(timeouts_options["total"], timeouts_options["read"], timeouts_options["write"], timeouts_options["connect"]))
        

    def create_fine_tune_file(self, file_path: str, purpose: Optional[str] = 'fine-tune') -> str:
        """
        Create a fine-tuning file in the OpenAI system using the updated API client.

        This method uploads a file to OpenAI for the purpose of fine-tuning a language model.
        It accepts a file path and the purpose of the file upload, then returns the ID of the
        created file. It handles exceptions related to file operations and API errors.

        Parameters
        ----------
        file_path : str
            The path to the JSONL file to be uploaded for fine-tuning.
        purpose : str, optional
            The purpose of the file upload. Default is 'fine-tune'.

        Returns
        -------
        str
            The ID of the created fine-tuning file.

        Raises
        ------
        FileNotFoundError
            If the file at the given file_path does not exist.
        PermissionError
            If there is a permission error while accessing the file.
        Exception
            If an error occurs during the file upload to OpenAI.

        Examples
        --------
        >>> api = FineTuningAPI(api_key="your-api-key")
        >>> file_id = api.create_fine_tune_file("path/to/dataset.jsonl")
        >>> print(file_id)
        'file-LMfxRAZ6cAHxp5KLRXZySlOR'
        """
        try:
            with open(file_path, "rb") as file_data:
                config = self.client.files.create(
                    file=file_data,
                    purpose=purpose
                )
            return config.id
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path {file_path} was not found.")
        except PermissionError:
            raise PermissionError(f"Permission denied when trying to open {file_path}.")
        except Exception as e:
            raise Exception(f"An error occurred with the OpenAI API: {e}")

    
    def create_fine_tuning_job(self, 
                           training_file: str, 
                           model: str, 
                           suffix: Optional[str] = None, 
                           batch_size: Optional[Union[str, int]] = 'auto', 
                           learning_rate_multiplier: Optional[Union[str, float]] = 'auto', 
                           n_epochs: Optional[Union[str, int]] = 'auto', 
                           validation_file: Optional[str] = None) -> dict:
        """
        Start a fine-tuning job using the OpenAI Python SDK.

        This method initiates a fine-tuning job with the specified model and training file.
        Additional parameters like batch size, learning rate multiplier, number of epochs,
        and validation file can be customized.

        Parameters
        ----------
        training_file : str
            The file ID of the training data uploaded to OpenAI API.
        model : str
            The name of the model to fine-tune.
        suffix : str, optional
            A suffix to append to the fine-tuned model's name. Default is None.
        batch_size : str or int, optional
            Number of examples in each batch. Defaults to 'auto'.
        learning_rate_multiplier : str or float, optional
            Scaling factor for the learning rate. Defaults to 'auto'.
        n_epochs : str or int, optional
            The number of epochs to train the model for. Defaults to 'auto'.
        validation_file : str, optional
            The file ID of the validation data uploaded to OpenAI API. Default is None.

        Returns
        -------
        dict
            A dictionary containing information about the fine-tuning job, including its ID.

        Raises
        ------
        ValueError
            If the training_file is not provided.
        Exception
            If an error occurs during the creation of the fine-tuning job.

        Examples
        --------
        >>> api = FineTuningAPI(api_key="your-api-key")
        >>> job_info = api.create_fine_tuning_job(training_file="file-abc123", 
                                                  model="gpt-3.5-turbo",
                                                  suffix="custom-model-name",
                                                  batch_size=4,
                                                  learning_rate_multiplier=0.1,
                                                  n_epochs=2,
                                                  validation_file="file-def456")
        >>> print(job_info)
        {'id': 'ft-xyz789', ...}
        """
        if not training_file:
            raise ValueError("A training_file must be provided to start a fine-tuning job.")

        hyperparameters = {
            'batch_size': batch_size,
            'learning_rate_multiplier': learning_rate_multiplier,
            'n_epochs': n_epochs,
        }

        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file,
                model=model,
                suffix=suffix,
                hyperparameters=hyperparameters,
                validation_file=validation_file
            )
            return response
        except Exception as e:
            raise Exception(f"An error occurred while creating the fine-tuning job: {e}")




    def list_fine_tuning_jobs(self, limit: int = 10) -> List[Dict]:
        """
        List the fine-tuning jobs with an option to limit the number of jobs returned.

        Parameters
        ----------
        limit : int, optional
            The maximum number of fine-tuning jobs to return. Default is 10.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each representing a fine-tuning job.

        Raises
        ------
        openai.error.OpenAIError
            If an error occurs with the OpenAI API request.
        """
        try:
            response = self.client.fine_tuning.jobs.list(limit=limit)
            return response.data
        except openai.error.OpenAIError as e:
            raise openai.error.OpenAIError(f"An error occurred while listing fine-tuning jobs: {e}")

    def retrieve_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Retrieve the state of a specific fine-tuning job.

        Parameters
        ----------
        job_id : str
            The ID of the fine-tuning job to retrieve.

        Returns
        -------
        Dict
            A dictionary containing details about the fine-tuning job.

        Raises
        ------
        ValueError
            If the job_id is not provided.
        openai.error.OpenAIError
            If an error occurs with the OpenAI API request.
        """
        if not job_id:
            raise ValueError("A job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            return response
        
        except openai.APIConnectionError as e:                            
                logger.error(f"[retrieve_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[retrieve_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[retrieve_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[retrieve_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"An error occurred during model evaluation: {e}")


    def cancel_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Cancel a specific fine-tuning job.

        Parameters
        ----------
        job_id : str
            The ID of the fine-tuning job to cancel.

        Returns
        -------
        Dict
            Confirmation of the cancellation.

        Raises
        ------
        ValueError
            If the job_id is not provided.
        openai.error.OpenAIError
            If an error occurs with the OpenAI API request.
        """
        if not job_id:
            raise ValueError("A job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.cancel(job_id)
            return response
        
        except openai.APIConnectionError as e:                            
                logger.error(f"[cancel_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[cancel_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[cancel_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[cancel_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[cancel_fine_tuning_job] An error occurred during model evaluation: {e}")


    
    def list_fine_tune_files(self) -> List[Dict]:
        """
        List files that have been uploaded to OpenAI for fine-tuning.

        This method retrieves a list of files uploaded to the OpenAI API, typically for the purpose
        of fine-tuning. The list includes details such as file IDs, created dates, and file purposes.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each containing details of an uploaded file.

        Raises
        ------
        Exception
            If an error occurs during the API request.

        Examples
        --------
        >>> api = FineTuningAPI(api_key="your-api-key")
        >>> files = api.list_uploaded_files()
        >>> for file in files:
        >>>     print(file)
        """
        try:
            response = self.client.files.list()
            return response.data
        except Exception as e:
            raise Exception(f"An error occurred while listing uploaded files: {e}")


    def list_events_fine_tuning_job(self, fine_tuning_job_id: str, limit: int = 10) -> List[Dict]:
        """
        List up to a specified number of events from a fine-tuning job.

        Parameters
        ----------
        fine_tuning_job_id : str
            The ID of the fine-tuning job to list events from.
        limit : int, optional
            The maximum number of events to return. Default is 10.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each representing an event from the fine-tuning job.

        Raises
        ------
        ValueError
            If the fine_tuning_job_id is not provided.
        openai.error.OpenAIError
            If an error occurs with the OpenAI API request.
        """
        if not fine_tuning_job_id:
            raise ValueError("A fine_tuning_job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tuning_job_id, limit=limit)
            return response
        except openai.APIConnectionError as e:                            
                logger.error(f"[list_events_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[list_events_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[list_events_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[list_events_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[list_events_fine_tuning_job] An error occurred during model evaluation: {e}")

    
    
    def delete_fine_tuned_model(self, model_id: str) -> Dict:
        """
        Delete a fine-tuned model. The caller must be the owner of the organization the model was created in.

        Parameters
        ----------
        model_id : str
            The ID of the fine-tuned model to delete.

        Returns
        -------
        Dict
            Confirmation of the deletion.

        Raises
        ------
        ValueError
            If the model_id is not provided.
        openai.error.OpenAIError
            If an error occurs with the OpenAI API request.
        """
        if not model_id:
            raise ValueError("A model_id must be provided.")

        try:
            response = self.client.models.delete(model_id)
            return response
        except openai.APIConnectionError as e:                            
                logger.error(f"[delete_fine_tuned_model] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[delete_fine_tuned_model] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[delete_fine_tuned_model] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[delete_fine_tuned_model] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[delete_fine_tuned_model] An error occurred during model evaluation: {e}")

    def use_fine_tuned_model(self, model_name: str, user_prompt:str, system_prompt="You are a helpful assistant." ) -> str:
        """
        Use a fine-tuned model to generate responses based on provided messages.

        This method allows for interaction with a fine-tuned model. It sends messages to the model
        and retrieves the generated response. The model should be available for inference after the
        fine-tuning job is completed.

        Parameters
        ----------
        model_name : str
            The name of the fine-tuned model to be used for generating responses.
        messages : List[Dict]
            A list of message dictionaries. Each message should have 'role' and 'content' keys.

        Returns
        -------
        str
            The generated response from the fine-tuned model.

        Raises
        ------
        Exception
            If an error occurs during the API request or while processing the response.

        Examples
        --------
        >>> api = FineTuningAPI(api_key="your-api-key")
        >>> response = api.use_fine_tuned_model(
            "ft:gpt-3.5-turbo:my-org:custom_suffix:id", 
            [{"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": "Hello!"}]
        )
        >>> print(response)
        'Response from the model...'
        """
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
            )
            return response.choices[0].message
        except openai.APIConnectionError as e:                            
                logger.error(f"[use_fine_tuned_model] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[use_fine_tuned_model] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[use_fine_tuned_model] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[use_fine_tuned_model] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[use_fine_tuned_model] An error occurred during model evaluation: {e}")

    def run_dashboard(self):
        """
        Run a fine-tuning dashboard with various operations.

        This method presents a menu to the user and allows them to choose from various
        fine-tuning-related operations, such as creating a fine-tune file, starting a
        fine-tuning job, listing fine-tuning jobs, retrieving job states, canceling jobs,
        listing events from a job, deleting fine-tuned models, and exiting the dashboard.

        Returns
        -------
        None

        """
        while True:
            print("\nMenu:")
            print("1. Create a fine-tune file")
            print("2. Create a fine-tuning job")
            print("3. List of tune-tune files")
            print("4. List 10 fine-tuning jobs")
            print("5. Retrieve the state of a fine-tune")
            print("6. Cancel a job")
            print("7. List up to 10 events from a fine-tuning job")
            print("8. Use a fine-tuned model")
            print("8. Delete a fine-tuned model")
            print("10. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                file_path = input("Enter the file path: ")
                purpose = input("Enter the purpose (fine-tune/other): ")
                print(self.create_fine_tune_file(file_path, purpose))

            elif choice == "2":
                training_file = input("Enter training file ID: ")
                model = input("Enter model name: ")
                suffix = input("Enter suffix (optional): ") or None
                batch_size = input("Enter batch size (auto/number): ") or 'auto'
                learning_rate_multiplier = input("Enter learning rate multiplier (auto/number): ") or 'auto'
                n_epochs = input("Enter number of epochs (auto/number): ") or 'auto'
                validation_file = input("Enter validation file ID (optional): ") or None
                print(self.create_fine_tuning_job(training_file, model, suffix, batch_size, 
                                                  learning_rate_multiplier, n_epochs, validation_file))

            elif choice == "3":
                print(self.list_fine_tune_files())
                print()
            
            elif choice == "4":
                print(self.list_fine_tuning_jobs())

            elif choice == "5":
                job_id = input("Enter fine-tuning job ID: ")
                print(self.retrieve_fine_tuning_job(job_id))

            elif choice == "6":
                job_id = input("Enter fine-tuning job ID to cancel: ")
                print(self.cancel_fine_tuning_job(job_id))

            elif choice == "7":
                job_id = input("Enter fine-tuning job ID for events: ")
                print(self.list_events_fine_tuning_job(job_id))

            elif choice == "8":
                model_name = input("Enter fine-tuned model name: ")
                user_prompt = input("Enter user prompt: ")
                system_prompt = "You are a helpful assistant."

                try:
                    response = self.use_fine_tuned_model(
                        model_name=model_name, 
                        user_prompt=user_prompt,
                        system_prompt=system_prompt
                    )
                    print(response)
                except Exception as e:
                    print(f"An error occurred: {e}")
            
            elif choice == "9":
                model_id = input("Enter fine-tuned model ID to delete: ")
                print(self.delete_fine_tuned_model(model_id))

            elif choice == "10":
                break

            else:
                print("Invalid choice. Please try again.")

