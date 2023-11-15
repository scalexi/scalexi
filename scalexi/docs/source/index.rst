scalexi Package
===============

Overview
--------

``scalexi``: A Comprehensive Toolkit for Language Model Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scalexi`` package provides a suite of tools catered towards facilitating the development and optimization of language models, especially those of substantial complexity.

Features
^^^^^^^^

- **Dataset Generation**: Simplify the process of creating datasets specifically tailored for enhancing language models.

- **Adaptable Data Formatting**: Customize your datasets to various forms, whether it's for prompt completion, structured instructions, or other model-specific needs.

- **Model Evaluation**: Incorporate mechanisms to assess the output of models, ensuring the alignment with desired outcomes.

- **Future-Ready Design**: While initially centered around certain prominent models, the foundation of ``scalexi`` is laid with a vision of embracing newer models and technologies.

By offering a combination of these tools, ``scalexi`` aims to streamline the workflow for developers and researchers in the domain of language models.


Installation
------------

Installing `scalexi` is a easy. With just a single command, you can have all the powerful tools and utilities of `scalexi` at your fingertips.

.. code-block:: bash

   pip install scalexi

This command fetches the latest version of `scalexi` from the Python Package Index (PyPI) and installs it on your system, ready for use.

With its range of tools and user-centric design, `scalexi` stands out as an essential package for anyone looking to harness the power of large language models effectively.



Getting Started
---------------

A Primer on Using ``scalexi``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scalexi`` package provides intuitive tools to automate tasks associated with large language models. To understand its capabilities better, let's explore a basic example that demonstrates how to use ``scalexi`` for generating questions based on given context.

Sample Usage
^^^^^^^^^^^^

.. code-block:: python

   from scalexi.dataset_generation.prompt_completion import generate_user_prompt, generate_prompt_completions
   from scalexi.document_loaders.context_loaders import context_from_csv_as_df

   def main():    
       context = context_from_csv_as_df("context.csv")
       print("Generated open-ended questions:")
       if context:
           questions = generate_prompt_completions(context, "generated.csv",
                                                   system_prompt="You are an assistant to create question from a context", 
                                                   num_questions=5, 
                                                   question_type="open-ended")
           print(questions())

   if __name__ == "__main__":
       main()

Step-by-Step Explanation
^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Import Necessary Modules**: Begin by importing the required functionalities from ``scalexi``.

2. **Loading Context from CSV**: Use the ``context_from_csv_as_df`` function to load the context data from a CSV file into a DataFrame.

3. **Generate Questions**: With the ``generate_prompt_completions`` function, craft questions based on the loaded context. Specify the system prompt, number of questions, and type of questions you desire.

4. **Displaying the Results**: Print out the generated questions for review.

By following these steps, you can easily harness the capabilities of ``scalexi`` for generating context-based questions.

Using this guide, you can explore and understand the features of ``scalexi`` for generating questions based on context in a systematic manner.



Modules Overview
----------------

``scalexi`` is structured into distinct modules, with each module focusing on a particular set of tasks. This modular approach aims to provide clarity and ease of use.

A brief description of the primary modules in ``scalexi``:

- **dataset_generation**: This module contains tools related to the creation of datasets, designed to support the training of large language models. It facilitates various data formatting tasks and prompt completions.

- **document_loaders**: This module focuses on mechanisms for loading context data. It provides support for importing data from different sources, including CSV files.

- **tests**: This module houses tests designed to verify the proper functioning of ``scalexi`` functionalities.

For a detailed look into each module and their features, refer to the module documentation provided below:

.. toctree::
   :maxdepth: 4

   modules




Advanced Topics
---------------

[Brief description of any advanced topics, functionalities, or features that users should be aware of.]

Best Practices
--------------

[Provide any best practices or recommendations for users when working with `scalexi`.]

Contributions
-------------

We welcome contributions! If you'd like to contribute to `scalexi`, [provide guidelines or refer to contribution guidelines if any.]

Further Reading
---------------

For more detailed documentation on each module and functionality, please refer to the respective sections. If you have questions, feel free to [contact method, e.g., open an issue on GitHub, send an email, etc.].

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::

DEV
---

.. toctree::
   :maxdepth: 2
   :caption: DEV
   :name: devmenu
   
   scalexi


LEARN
-----

.. toctree::
   :maxdepth: 2
   :caption: LEARN
   :name: learnmenu

   scalexi.document_loaders

API
---

.. toctree::
   :maxdepth: 2
   :caption: API
   :name: apimenu

