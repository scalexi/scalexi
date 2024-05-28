from scalexi.openai.utilities import get_openai_pricing_info, get_context_length, calculate_token_usage_for_text, estimate_inference_cost, extract_response_and_token_usage_and_cost,get_language_model_pricing

# Example text to test token usage calculation
text = "Hello, how are you?"

print(get_language_model_pricing("gpt-4"))


# Test calculate_token_usage_for_text
try:
    tokens = calculate_token_usage_for_text(text, model="gpt-3.5-turbo-1106")
    print(f"Token count for '{text}':", tokens)
except Exception as e:
    print("Error in calculating token usage:", str(e))

# Test get_context_length for a known model
try:
    context_length = get_context_length("gpt-4o")
    print(f"Context length for 'gpt-4o':", context_length)
except Exception as e:
    print("Error in getting context length:", str(e))

# Example token usage for testing cost estimation
token_usage = {"prompt_tokens": 100, "completion_tokens": 50}

# Test estimate_inference_cost
try:
    cost = estimate_inference_cost(token_usage, "gpt-4")
    print(f"Estimated cost for token usage {token_usage} using 'gpt-3.5-turbo': ${cost:.4f}")
except Exception as e:
    print("Error in estimating inference cost:", str(e))

# Ensure pricing_info is loaded correctly for testing
try:
    pricing_info = get_openai_pricing_info()
    print("Pricing info loaded successfully.")
except Exception as e:
    print("Error loading pricing info:", str(e))



# Test the extraction of response, token usage, and cost estimation from a hypothetical API response

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            elif isinstance(value, list):
                value = [DictToObject(item) if isinstance(item, dict) else item for item in value]
            setattr(self, key, value)

mock_response = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-0613",
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "\n\nThis is a test!"
            },
            "logprobs": "null",
            "finish_reason": "stop",
            "index": 0
        }
    ]
}



response_obj = DictToObject(mock_response)
#print((response_obj.choices[0].message.content))

try:
    content, usage, estimated_cost = extract_response_and_token_usage_and_cost(response_obj, "gpt-3.5-turbo")
    print(f"Extracted content: {content}")
    print(f"Token usage: {usage}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
except Exception as e:
    print("Error extracting data from response:", str(e))
