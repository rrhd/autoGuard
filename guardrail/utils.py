import requests


def get_embedding(text: str, model_identifier: str = "", api_url: str = "http://localhost:1234/v1/embeddings") -> dict | None:
    """
    Sends a request to the LM Studio embeddings endpoint to obtain an embedding for the given text.

    Args:
        text: The input text for which to generate an embedding.
        model_identifier: The identifier of the model to use for generating embeddings.
        api_url: The URL of the LM Studio embeddings API.

    Returns:
        A dictionary containing the embedding data if the request is successful; None if there is an error.
    """
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "input": text,
        "model": model_identifier
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def get_embeddings(texts: list[str], model_identifier: str = "", api_url: str = "http://localhost:1234/v1/embeddings") -> list[dict | None]:
    """
    Generates embeddings for a list of texts using the LM Studio embeddings endpoint.

    Args:
        texts: A list of texts for which to generate embeddings.
        model_identifier: The identifier of the model to use for generating embeddings.
        api_url: The URL of the LM Studio embeddings API.

    Returns:
        A list of dictionaries containing the embedding data for each text. If an error occurs for any text, the corresponding list entry will be None.
    """
    embeddings = []
    for text in texts:
        embedding = get_embedding(text, model_identifier, api_url)
        if embedding:
            embeddings.append(embedding['data'][0]['embedding'])
        else:
            embeddings.append(None)
    return embeddings


def get_chat_response(messages: list[dict[str, str]], model: str = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", temperature: float = 0.0) -> str:
    """
    Sends a request to the LM Studio chat completion endpoint to obtain a response based on the input messages.

    Args:
        messages: A list of messages representing the conversation history. Each message should be a dictionary with 'role' and 'content' keys.
        model: The identifier of the model to use for generating the chat response.
        temperature: The sampling temperature to use during response generation.

    Returns:
        The content of the chat response as a string.
    """
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": -1,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
