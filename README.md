# scalexi Python API  

_Simplifying LLM Development and Fine-Tuning with Python_


# Overview

[![PyPI version](https://img.shields.io/pypi/v/scalexi.svg)](https://pypi.org/project/scalexi/)
[![Read the Docs](https://img.shields.io/badge/docs-read_the_docs-blue.svg)](http://docs.scalexi.ai/)
[![Project Folder](https://img.shields.io/badge/Google%20Drive-Project%20Folder-green.svg)](https://drive.google.com/drive/u/1/folders/1XoUzkGwpHns1fUa-Q9j7iDIqYoRPch61)


`scalexi` is a versatile open-source Python library, optimized for Python 3.11+, focuses on facilitating low-code development and fine-tuning of diverse Large Language Models (LLMs). It extends beyond its initial OpenAI models integration, offering a scalable framework for various LLMs.

Key to `scalexi` is its low-code approach, significantly reducing the complexity of dataset preparation and manipulation. It features advanced dataset conversion tools, adept at transforming raw contextual data into structured datasets fullfilling LLMs fine-tuning requirements. These tools support multiple question formats, like open-ended, closed-ended, yes-no, and reflective queries, streamlining the creation of customized datasets for LLM fine-tuning.

A standout feature is the library's automated dataset generation, which eases the workload involved in LLM training. `scalexi` also provides essential utilities for cost estimation and token counting, aiding in effective resource management throughout the fine-tuning process.

Developed by [scalexi.ai](https://scalexi.ai/), the library leverages a robust specification to facilitate fine-tuning context-specific models with OpenAI API. Alsom `scalexi` ensures a user-friendly experience while maintaining high performance and error handling.

Explore the full capabilities of Large Language Models with `scalexi`'s intuitive and efficient Python API with minimal coding for easy LLM development and fine-tuning from dataset creation to LLM evaluation.

## Documentation

For comprehensive guides, API references, and usage examples, visit the [`scalexi` Documentation](http://docs.scalexi.ai/). It provides an up-to-date information you need to effectively utilize the `scalexi` library for LLM development and fine-tuning.


## Features

- **Low-Code Interface**: `scalexi` offers a user-friendly, low-code platform that simplifies interactions with LLMs. Its intuitive design minimizes the need for extensive coding, making LLM development accessible to a broader range of users.

- **Automated Dataset Generation**: The library excels in converting raw data into structured formats, aligning with specific LLM fine-tuning requirements. This automation streamlines the dataset preparation process, saving time and reducing manual effort.

- **Versatile Dataset Format Support**: `scalexi` is designed to handle various dataset formats including CSV, JSON, and JSONL. It also facilitates effortless conversion between these formats, providing flexibility in dataset management and utilization.

- **Simplified Fine-Tuning Process**: The library provides simplified interfaces for fine-tuning LLMs. These user-friendly tools allow for easy customization and optimization of models on specific datasets, enhancing model performance and applicability.

- **Efficient Model Evaluation**: `scalexi` includes utilities for the automated evaluation of fine-tuned models. This feature assists in assessing model performance, ensuring the reliability and effectiveness of the fine-tuned models.

- **Token Usage Estimation**: The library incorporates functions to accurately estimate token usage and associated costs. This is crucial for managing resources and budgeting in LLM projects, providing users with a clear understanding of potential expenses.


## Installation

Easily install `scalexi` with pip. Just run the following command in your terminal:

```bash
pip install scalexi
```
This will install scalexi and its dependencies, making it ready for use with Python 3.11 and above (not tested on lower Python versions). 

## Usage

The `scalexi` toolkit offers comprehensive features for creating, evaluating, and fine-tuning Large Language Models (LLMs) with OpenAI's API. It allows users to generate datasets from custom context entries, estimate costs for model training and inference, and convert datasets into formats suitable for fine-tuning. Users can fine-tune models with the FineTuningAPI, which includes a dashboard for managing fine-tuning jobs. Additionally, ScaleXI facilitates the evaluation of fine-tuned LLMs by generating random samples, rephrasing prompts for better generalization, and assessing model performance based on generated completions. This toolkit simplifies and streamlines the process of working with LLMs, making it more accessible and efficient for various applications in research, academia, and industry.

In what follow, we present the different use cases of `scalexi`.

### I. Automated Dataset Generation 
[![01-create-fine-tuning-dataset](https://img.shields.io/badge/Colab-01--create--fine--tuning--dataset-blue.svg)](https://colab.research.google.com/drive/1ClCT0jvThX0RvPN3eiWJnQVNjncW9R4-?authuser=1#scrollTo=Yk7UtLqKBgy5)

#### Context File Setup
To generate a dataset with scalexi, prepare a CSV file with a single column titled 'context'. Populate this column with context entries, each in a new row, ensuring the content is within the LLM's token limit. Save the file in a recognized directory before starting the dataset creation process.

Here is an illutrative example of a `context.csv` file

```js
context,
"Your first context entry goes here. It can be a paragraph or a document that you want to use as the basis for generating questions or prompts.",
"Your second context entry goes here. Make sure that each entry is not too lengthy to stay within the token limits of your LLM."
```

#### Create your dataset
After installing `scalexi`, you can create a fine-tuning dataset for Large Language Models (LLMs) using your own context data. Below is a simple script demonstrating how to generate a dataset:

```python
import os
 from scalexi.dataset_generation.prompt_completion import PromptCompletionGenerator

# Ensure your OpenAI API key is set as an environment variable
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Instantiate the generator with desired settings
generator = PromptCompletionGenerator(enable_timeouts=True)

# Specify the path to your context file and the desired output file for the dataset
context_file = 'path/to/your/context.csv'
output_dataset_file = 'path/to/your/generated_dataset.csv'

# Call the create_dataset method with your parameters
generator.create_dataset(context_file, output_dataset_file,
                        num_questions=1, 
                        question_types=["yes-no", "open-ended", "reflective"],
                        model="gpt-3.5-turbo-1106",
                        temperature=0.3,
                        detailed_explanation=True)
```
This script will generate a dataset with `'yes-no'`, `'open-ended'` and `'reflective'`, type questions based on the context provided in your CSV file.


### II.Cost Estimation and Dataset Formatting with ScaleXI

[![02-estimate-pricing](https://img.shields.io/badge/Colab-02--estimate--pricing-blue.svg)](https://colab.research.google.com/drive/1FWJFW5v82j0j9bkBbHkVlRqakoRnpKS0?authuser=1#scrollTo=WA2EdYTWGWd1)



The ScaleXI library provides utilities for estimating the cost of using OpenAI's models and converting datasets into the required formats.

#### Estimating Costs with OpenAIPricing

The `OpenAIPricing` class can estimate the costs for fine-tuning and inference. Here's how you can use it:

```python
import json
import pkgutil
from scalexi.openai.pricing import OpenAIPricing

# Load the pricing data
data = pkgutil.get_data('scalexi', 'data/openai_pricing.json')
pricing_info = json.loads(data)

# Create an OpenAIPricing instance
pricing = OpenAIPricing(pricing_info)

# Estimate cost for fine-tuning
number_of_tokens = 10000  # Replace with your actual token count
estimated_cost = pricing.estimate_finetune_training_cost(number_of_tokens, model_name="gpt-3.5-turbo")
print(f"Estimated cost for fine-tuning with {number_of_tokens} tokens: ${estimated_cost:.2f}")

# Estimate cost for inference
input_tokens = 10000  # Replace with your actual input token count
output_tokens = 5000  # Replace with your actual output token count
estimated_cost = pricing.estimate_inference_cost(input_tokens, output_tokens, model_name="gpt-3.5-turbo")
print(f"Estimated inference cost: ${estimated_cost:.2f}")
```

### III. Converting Datasets with DataFormatter

[![03-convert_dataset-with-dataformatter](https://img.shields.io/badge/Colab-03--convert_dataset--with--dataformatter-blue.svg)](https://colab.research.google.com/drive/1fQALOQGPv0XalRUZiAZL4h3Km6n5kNMj?authuser=1#scrollTo=SZdMZ5WHH8X1)

The DataFormatter class can convert datasets from CSV to JSONL, which is the required format for fine-tuning datasets on OpenAI.
```python
from scalexi.utilities.data_formatter import DataFormatter

# Initialize the DataFormatter
dfm = DataFormatter()

# Convert a CSV dataset to JSONL format
csv_dataset_path = "path/to/your/dataset.csv"  # Replace with your actual CSV file path
jsonl_dataset_path = "path/to/your/dataset.jsonl"  # Replace with your desired JSONL file path
dfm.csv_to_jsonl(csv_dataset_path, jsonl_dataset_path)
```


####  Fine-Tuning Dataset Conversion
ScaleXI also provides a method to convert a dataset from prompt completion to a conversation format, suitable for fine-tuning of OpenAI GPT-based conversational models:

```python
# Convert prompt completion dataset to conversation format
prompt_completion_dataset_path = "path/to/your/generated_dataset.jsonl"  # Replace with your actual JSONL file path
conversation_dataset_path = "path/to/your/conversation_dataset.jsonl"  # Replace with your desired JSONL file path
dfm.convert_prompt_completion_to_conversation(prompt_completion_dataset_path, conversation_dataset_path)

# Calculate token usage for the conversation dataset
number_of_tokens = pricing.calculate_token_usage_for_dataset(conversation_dataset_path)
print(f"Number of tokens in the conversation dataset: {number_of_tokens}")

# Estimate fine-tuning cost for the conversation dataset
estimated_cost = pricing.estimate_finetune_training_cost(number_of_tokens, model_name="gpt-3.5-turbo")
print(f"Estimated fine-tuning cost for the conversation dataset: ${estimated_cost:.2f}")
```

### IV.Fine-Tuning OpenAI Models with ScaleXI
[![04-fine-tuning-openai-models](https://img.shields.io/badge/Colab-04--fine--tuning--openai--models-blue.svg)](https://colab.research.google.com/drive/14cX5Km2GB89hCFAOuAR68-p7t2djhbli?authuser=1#scrollTo=LoNPMN2UIfu0)

ScaleXI simplifies the process of fine-tuning Large Language Models (LLMs) with OpenAI's API. Here's how you can start fine-tuning your models with the `FineTuningAPI` class.

#### Setting up FineTuningAPI

First, you need to set your OpenAI API key as an environment variable. You can do this within your script or in your shell environment.

```python
import os
from scalexi.openai.fine_tuning_api import FineTuningAPI

# Prompt for the OpenAI API key and set it as an environment variable
api_key = input("Please enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key

# Confirm that the API key has been set
print(f"OpenAI API key set: {os.getenv('OPENAI_API_KEY') is not None}")
```

#### Running the Fine-Tuning Dashboard
After setting up your API key, you can launch the fine-tuning dashboard provided by ScaleXI to monitor and manage your fine-tuning jobs.

```python
from scalexi.openai.fine_tuning_api import FineTuningAPI

# Initialize the FineTuningAPI with the API key from the environment
api = FineTuningAPI(api_key=os.getenv("OPENAI_API_KEY"))

# Run the dashboard
api.run_dashboard()
```

This dashboard allows you to start new fine-tuning jobs, monitor the progress of ongoing jobs, and review past jobs.

```
Menu:
1. Create a fine-tune file
2. Create a fine-tuning job
3. List of tune-tune files
4. List 10 fine-tuning jobs
5. Retrieve the state of a fine-tune
6. Cancel a job
7. List up to 10 events from a fine-tuning job
8. Use a fine-tuned model
8. Delete a fine-tuned model
10. Exit
```


### V.Fine-Tuned LLM Evaluation
[![05-evaluate-a-fine-tuned-model](https://img.shields.io/badge/Colab-05--evaluate--a--fine--tuned--model-blue.svg)](https://colab.research.google.com/drive/1x39mKd0tJt-iAGIT0-FXieMpkKXSII0T?authuser=1#scrollTo=bfYV0a-bL7BO)

#### Step 1: Random Sample Creation
Generate a random sample from your dataset for evaluation purposes.

```python
from scalexi.llm_evaluation.evaluate import LLMEvaluation
import os

# Set your OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize the LLMEvaluation object
llm_evaluation = LLMEvaluation(enable_timeouts=False)

# Define the path to your dataset
conversation_dataset_jsonl = 'path/to/your/files/conversation_dataset.jsonl'

# Specify the output details for the sample dataset
output_folder = 'path/to/output/folder/'
output_file = output_folder + 'random_prompts.csv'

# Create a CSV output needed for evaluation
llm_evaluation.save_random_prompts(conversation_dataset_jsonl, output_file, output_format='csv', n_samples=10)
```

#### Step 2: Rephrase Prompts

Improve your model's ability to generalize by rephrasing prompts and optionally classifying them. You can choose any classes of your interest. 

```python
# Rephrase prompts and classify them if needed
rephrased_dataset_csv = output_folder + 'rephrased_dataset.csv'
llm_evaluation.rephrase_and_classify_prompts_in_dataset(output_file, rephrased_dataset_csv, 
                                            classify=True, 
                                            classes=['ACADEMIC', 'RESEARCH', 'ADMIN', 'SCIENCE', 'OTHERS'])
```
#### Step 3: Evaluate LLM
Assess the performance of your LLM by comparing the generated completions with ground truth.

```python
# Define the fine-tuned model name for evaluation
finetuned_model = 'ft-gpt-3.5-turbo-your-model-id'
evaluation_results_csv = output_folder + 'evaluation_results.csv'
# Evaluate the fine-tuned model using the rephrased dataset
llm_evaluation.evaluate_model(finetuned_model, 
                              rephrased_dataset_csv, 
                              evaluation_results_csv,
                              temperature=0.3, max_tokens=250, top_p=1.0,
                              frequency_penalty=0, presence_penalty=0,
                              llm_evaluator_model_name='gpt-3.5-turbo', 
                              experiment_id=1,  # Experiment identifier
                              save_immediately=False)
```

The `evaluation_results.csv` file compiles the scores from the evaluation, providing a clear metric of your LLM's performance.

# Contributing

We warmly welcome contributions to `scalexi`! Whether you're fixing bugs, adding new features, or improving documentation, your help is invaluable.

Before you start, please take a moment to review our contribution guidelines. They provide important instructions and best practices to follow, ensuring a smooth and efficient contribution process.

You can find all the necessary details in our [Contribution Guidelines](CONTRIBUTING.md). Thank you for your interest in enhancing `scalexi`!

# License

`scalexi` is released under the ScaleXI License 1.0. This license ensures that the software can be freely used, reproduced, and distributed, both for academic and business purposes, while requiring proper attribution to ScaleX Innovation company in any derivative works or distributions.

For full details of the terms and conditions, please refer to the [LICENSE](LICENSE) file included with the software.


# Contact
For support or queries, reach out to us at Anis Koubaa <akoubaa@scalexi.com>.

