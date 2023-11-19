from setuptools import setup, find_packages

setup(
    name="scalexi",
    version="0.4.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'scalexi': ['data/*']},
    install_requires=[
        "pandas",  # Add any package dependencies here
        "openai>=1.0.0", #this package is not compatible with earlier versions of openai
        "sphinx",   # Add any other dependencies as needed
        "cohere",
        "sphinx_rtd_theme",   # Add any other dependencies as needed
        "tiktoken",  # Add any other dependencies as needed
        "pyyaml",
        "lxml",
        "requests",
        "httpx"
    ],
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here (if applicable)
        ]
    },
    author="ScaleX Innovation",
    author_email="info@scalexi.ai",
    description="The scalexi package is a versatile toolkit for generating datasets and conducting experiments with OpenAI's GPT-3 language model.",
    long_description="""The "scalexi" package simplifies the process of crafting prompts, generating responses, and formatting data for various natural language processing tasks. It provides essential tools and functions for researchers, developers, and data scientists who want to harness the power of GPT-3 for dataset creation and experimentation.

            Key Features:
            - Prompt and Completion Formatting: Easily format prompts and completions for GPT-3, including options for starting and ending sequences.
            - JSON and CSV Export: Convert generated data into JSON and CSV formats for easy storage and analysis.
            - Contextual Experimentation: Seamlessly integrate context and prompts to conduct experiments and generate structured responses.
            - Question Generation: Generate user prompts with predefined question types, making it simple to create various types of language tasks.

            Installation:
            To install the package, use pip:

            ```bash
            pip install scalexi
            ```
            """,
    long_description_content_type="text/markdown",
    url="https://github.com/scalexi/scalexi",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.11",
    ],
)
