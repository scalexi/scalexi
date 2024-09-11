from setuptools import setup, find_packages


setup(
    name="scalexi",
    version="0.4.7.18",
    packages=find_packages(),
    include_package_data=True,
    package_data={'scalexi': ['data/*']},
    install_requires=[
        "pandas",  # Add any package dependencies here
        "openai>=1.10.0", #this package is not compatible with earlier versions of openai
        "sphinx",   # Add any other dependencies as needed
        "cohere",
        "sphinx_rtd_theme",   # Add any other dependencies as needed
        "tiktoken",  # Add any other dependencies as needed
        "pyyaml",
        "lxml",
        "requests",
        "httpx",
        "langchain",
        "pypdf",
        "langchain-openai",
        "fastapi",
        "pymupdf",
        'faiss-cpu',
        'langchain_text_splitters',
        'chromadb',
        'anthropic',
        'langchain_community',
        'aiofiles',
        'asyncio',
        'aiohttp',
        'pdfplumber',
        'PyPDF2',
        'pyautogen',
        'chardet',
        'pycryptodome',
        'seaborn',
        'matplotlib',
        'scipy',
        'pymupdf',
    ],
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here (if applicable)
        ]
    },
    author="ScaleX Innovation",
    author_email="akoubaa@scalexi.ai",
    description="The scalexi package is a versatile open-source Python library that focuses on facilitating low-code development and fine-tuning of diverse Large Language Models (LLMs). It extends beyond its initial OpenAI models integration, offering a scalable framework for various LLMs.",
    long_description_content_type="text/markdown",
    long_description=
"""\
# Overview

`scalexi` is a versatile open-source Python library that focuses on facilitating low-code development and fine-tuning of diverse Large Language Models (LLMs). It extends beyond its initial OpenAI models integration, offering a scalable framework for various LLMs.

Key to `scalexi` is its low-code approach, significantly reducing the complexity of dataset preparation and manipulation. It features advanced dataset conversion tools, adept at transforming raw contextual data into structured datasets fulfilling LLMs fine-tuning requirements. These tools support multiple question formats, like open-ended, closed-ended, yes-no, and reflective queries, streamlining the creation of customized datasets for LLM fine-tuning.

A standout feature is the library's automated dataset generation, which eases the workload involved in LLM training. `scalexi` also provides essential utilities for cost estimation and token counting, aiding in effective resource management throughout the fine-tuning process.

Developed by ScaleX Innovation [scalexi.ai](https://scalexi.ai/), the library leverages a robust specification to facilitate fine-tuning context-specific models with OpenAI API. Also, `scalexi` ensures a user-friendly experience while maintaining high performance and error handling.

Explore the full capabilities of Large Language Models with `scalexi`'s intuitive and efficient Python API with minimal coding for easy LLM development and fine-tuning from dataset creation to LLM evaluation.

## Documentation

For comprehensive guides, API references, and usage examples, visit the [`scalexi` Documentation](https://docs.scalexi.ai/). It provides an up-to-date information you need to effectively utilize the `scalexi` library for LLM development and fine-tuning.

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


## Tutorials

For documentation and tutorials, visit the [`scalexi` Documentation](https://docs.scalexi.ai/). It provides an up-to-date information you need to effectively utilize the `scalexi` library for LLM development and fine-tuning.

## License
This project is licensed under the ScaleXI 1.0 License - see the LICENSE file for details. 
""",
    url="https://github.com/scalexi/scalexi",
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.11",
    ],
)
