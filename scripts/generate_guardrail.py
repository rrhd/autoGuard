import click
import json

from guardrail.basemodels import Policy
from guardrail.guardrail import GuardrailFactory
from guardrail.utils import get_chat_response


@click.command()
@click.argument('policy_json', type=click.Path(exists=True))
@click.option('--output', default='guardrail.pkl', help='Path to save the trained guardrail model.')
@click.option('--num_epochs', default=3, help='Number of training epochs.')
@click.option('--min_samples', default=50, help='Minimum samples per category.')
def generate_guardrail(policy_json, output, num_epochs, min_samples):
    """
    Generate a guardrail model from a policy defined in a JSON file.

    POLICY_JSON: The path to the JSON file containing the policy.
    """

    # Load the policy from the JSON file
    with open(policy_json, 'r') as file:
        policy_data = json.load(file)

    policy = Policy(**policy_data)

    # Create a simple chat function to pass into the guardrail factory
    def policy_chatbot(messages):
        messages = [{"role": "system", "content": policy.text()}] + messages
        return get_chat_response(messages)

    # Create a GuardrailFactory
    factory = GuardrailFactory(policy)

    # Train the guardrail
    factory.train_guardrail(policy_chatbot, num_epochs=num_epochs, min_samples_per_category=min_samples)

    # Save the trained model
    factory.guardrail.save_model(output)

    click.echo(f"Guardrail model saved to {output}")


if __name__ == '__main__':
    generate_guardrail()
