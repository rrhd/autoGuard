from pydantic import BaseModel, Field


class Policy(BaseModel):
    name: str = Field(..., description="The name of the policy")
    description: str = Field(..., description="Detailed description of the policy")
    allowed: list[str] = Field(..., description="List of allowed requests or actions")
    disallowed: list[str] = Field(..., description="List of disallowed requests or actions")
    example_compliant_prompts: list[str] = Field(default_factory=list,
                                                 description="Examples of prompts that comply with the policy")
    example_noncompliant_prompts: list[str] = Field(default_factory=list,
                                                    description="Examples of prompts that violate the policy")

    def text(self) -> str:
        """
        Generates a textual representation of the policy, including its name, description,
        allowed actions, disallowed actions, and examples.

        Returns:
            A formatted string representing the policy.
        """
        policy_text = f"Policy: {self.name}\n\n"
        policy_text += f"Description: {self.description}\n\n"

        policy_text += "Allowed Actions:\n"
        for action in self.allowed:
            policy_text += f"- {action}\n"
        policy_text += "\n"

        policy_text += "Disallowed Actions:\n"
        for action in self.disallowed:
            policy_text += f"- {action}\n"
        policy_text += "\n"

        if self.example_compliant_prompts:
            policy_text += "Example Compliant Prompts:\n"
            for prompt in self.example_compliant_prompts:
                policy_text += f"- {prompt}\n"
            policy_text += "\n"

        if self.example_noncompliant_prompts:
            policy_text += "Example Non-Compliant Prompts:\n"
            for prompt in self.example_noncompliant_prompts:
                policy_text += f"- {prompt}\n"

        return policy_text.strip()

    @classmethod
    def from_json(cls, file_path: str) -> 'Policy':
        """
        Loads a policy from a JSON file.

        Args:
            file_path: Path to the JSON file containing the policy data.

        Returns:
            A Policy instance loaded with the data from the JSON file.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
