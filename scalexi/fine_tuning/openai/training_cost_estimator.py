import tiktoken

def calculate_token_usage_for_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Calculate the total number of tokens used by a list of messages.

    This function estimates the token usage for messages based on the model's
    tokenization scheme. It supports different versions of GPT-3.5 Turbo and
    GPT-4 models. For unsupported models, a NotImplementedError is raised.
    This is used to estimate the cost of interactions with OpenAI's API based
    on message lengths.

    Parameters
    ----------
    messages : list of dict
        List of message dictionaries with keys like 'role', 'name', and 'content'.
    model : str, optional
        Identifier of the model to estimate token count. Default is "gpt-3.5-turbo-0613".

    Returns
    -------
    int
        Total number of tokens for the messages as per the model's encoding scheme.

    Raises
    ------
    KeyError
        If the model's encoding is not found.
    NotImplementedError
        If token counting is not implemented for the model.

    Examples
    --------
    >>> messages = [{"role": "user", "content": "Hello!"}, 
    ...             {"role": "assistant", "content": "Hi there!"}]
    >>> calculate_token_usage_for_messages(messages)
    14  # Example token count for "gpt-3.5-turbo-0613" model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Token allocation per model
    tokens_allocation = {
        "gpt-3.5-turbo-0613": (3, 1),
        "gpt-3.5-turbo-16k-0613": (3, 1),
        "gpt-4-0314": (3, 1),
        "gpt-4-32k-0314": (3, 1),
        "gpt-4-0613": (3, 1),
        "gpt-4-32k-0613": (3, 1),
        "gpt-3.5-turbo-0301": (4, -1)  # every message follows {role/name}\n{content}\n
    }

    # Default tokens per message and name
    tokens_per_message, tokens_per_name = tokens_allocation.get(
        model, 
        (3, 1)  # Default values
    )
    
    # Handling specific model updates
    if "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Assuming gpt-3.5-turbo-0613.")
        tokens_per_message, tokens_per_name = tokens_allocation["gpt-3.5-turbo-0613"]
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Assuming gpt-4-0613.")
        tokens_per_message, tokens_per_name = tokens_allocation["gpt-4-0613"]
    else:
        raise NotImplementedError(
            f"Token counting not implemented for model {model}. "
            "See the OpenAI Python library documentation for details."
        )
    
    # Token counting
    num_tokens = 3  # every reply is primed with 'assistant'
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += sum(len(encoding.encode(value)) for key, value in message.items())
        if "name" in message:
            num_tokens += tokens_per_name

    return num_tokens
