import boto3
import os

def fetch_secrets_to_env(parameter_mapping, region="eu-central-1"):
    """
    Fetches parameters from AWS Parameter Store (SSM) and sets them as environment variables.
    :param parameter_mapping: A dict mapping from 'parameter_name_in_ssm' -> 'ENV_VARIABLE_NAME'
    :param region: AWS region where SSM parameters are stored
    """
    ssm = boto3.client("ssm", region_name=region)

    for param_name, env_var in parameter_mapping.items():
        try:
            response = ssm.get_parameter(Name=param_name, WithDecryption=True)
            value = response["Parameter"]["Value"]
            os.environ[env_var] = value
            print(f"Loaded secret for {env_var}.")
        except Exception as e:
            print(f"Failed to fetch parameter [{param_name}] for env var [{env_var}]: {str(e)}")

if __name__ == "__main__":
    # A dictionary mapping the SSM parameter paths to the environment variable names
    parameter_mapping = {
        "/running/LANGFUSE_SECRET_KEY": "LANGFUSE_SECRET_KEY",
        "/running/LANGFUSE_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
        "/running/LANGFUSE_HOST":       "LANGFUSE_HOST",
        "/running/OPENAI_API_KEY":      "OPENAI_API_KEY"
    }

    # Fetch secrets from the specified region and set them as environment variables
    fetch_secrets_to_env(parameter_mapping, region="eu-central-1")  # Change region if needed
