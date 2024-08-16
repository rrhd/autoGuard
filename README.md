# AutoGuard

## Overview

This repository provides tools to generate and enforce guardrails around chat models. Guardrails are defined by policies that outline allowed and disallowed behaviors. The repository leverages local language models (LLMs) and embedding models via LM Studio to train and evaluate these guardrails.

## Prerequisites

### LM Studio Setup

To use the tools in this repository, you must have LM Studio running locally, or another service that can handle OpenAI-style requests. LM Studio supports running both LLMs and embedding models on your local machine. You must configure it as follows:

1. **LLM and Embedding Model Setup**:
   - LM Studio should be running an LLM model capable of generating chat completions.
   - An embedding model should be loaded before starting the LM Studio server to handle embedding requests.

2. **API Endpoint**:
   - The default API endpoint is `http://localhost:1234`, which should be accessible by the scripts in this repository.

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Repository Structure

- **`guardrail/`**: Contains the core functionality for generating and enforcing guardrails.
  - **`policy.py`**: Defines the `Policy` class, which encapsulates the rules for allowed and disallowed actions.
  - **`guardrail.py`**: Contains the `Guardrail` and `GuardrailFactory` classes for creating and managing guardrails.
  - **`utils.py`**: Utility functions for interacting with LM Studio's API endpoints.
- **`scripts/`**: Scripts for generating and managing guardrails.
  - **`generate_guardrail.py`**: Generates a guardrail from a policy defined in a JSON file.

## Usage

### 1. Defining a Policy

Policies are defined in JSON files and must adhere to the structure required by the `Policy` class. Here’s an example policy definition in JSON format:

```json
{
    "name": "Prohibit Cat Discussion",
    "description": "Prohibit statements, questions, or discussions related to cats...",
    "allowed": [
        "Requests for information about other pets or animals",
        "Discussions about general pet care that don't specifically mention cats"
    ],
    "disallowed": [
        "Requests for advice or opinions on domestic cat breeds",
        "Discussions about cat behavior or training"
    ],
    "example_compliant_prompts": [
        "Can you give me information about dog breeds?",
        "What's the best way to train a parrot?"
    ],
    "example_noncompliant_prompts": [
        "What is the best breed of cat?",
        "How should I train my kitten?"
    ]
}
```

Save this policy as `policy.json`.

### 2. Generating a Guardrail

Use the `generate_guardrail.py` script to generate a guardrail model from the policy JSON file:

```bash
python scripts/generate_guardrail.py policy.json --output guardrail.pkl --num_epochs 3 --min_samples 50
```

- **`policy.json`**: Path to the policy file.
- **`--output`**: Path to save the trained guardrail model (`guardrail.pkl`).
- **`--num_epochs`**: Number of training epochs (default is 3).
- **`--min_samples`**: Minimum samples per category (default is 50).

This command generates a `guardrail.pkl` file that contains the trained guardrail model.

### 3. Loading and Using the Guardrail

To use the generated guardrail model, load it in your application:

```python
from guardrail import create_guardrail
from policies import Policy
import json

# Load the policy
with open('policy.json', 'r') as f:
    policy_data = json.load(f)
policy = Policy(**policy_data)

# Load the guardrail
safe_chat_function = create_guardrail(policy, your_chat_function, load=True)

# Use the guardrail-wrapped chat function
response = safe_chat_function([
    {"role": "user", "content": "Should I invest in cryptocurrency?"}
])
print(response['response'])
```

### 4. Utility Functions

This repository also provides utility functions to interact with LM Studio:

- **`get_embedding`**: Fetches an embedding for a given text.
- **`get_embeddings`**: Fetches embeddings for a list of texts.
- **`get_chat_response`**: Generates a chat response using the LLM model.

These functions are designed to interface with the LM Studio API, allowing you to easily integrate local model inference into your guardrail workflow.

## How It Works

- **Policy Definition**: Policies are JSON files that define allowed and disallowed actions and provide example prompts.
- **Guardrail Training**: The `GuardrailFactory` trains models that classify whether a given prompt is allowed or disallowed according to the policy.
- **Guardrail Enforcement**: The trained model is used to enforce the policy by wrapping the chat function, blocking or modifying responses that violate the policy.

## Output

The primary output of this process is a `guardrail.pkl` file, which contains a trained model that can be used to enforce the defined policy.

## Reloading the Model

Once generated, the `guardrail.pkl` file can be reloaded to enforce the policy without retraining. The `create_guardrail` function simplifies this process by wrapping your chat function with the loaded guardrail model.

## Conclusion

This repository allows you to enforce strict guidelines around your chat models by training guardrails based on customizable policies. By leveraging local models through LM Studio, you can maintain control over the inference process and adapt the system to specific needs.