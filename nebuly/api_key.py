import os

from nebuly.exceptions import APIKeyNotProvidedError


def get_api_key() -> str:
    api_key = os.environ.get("NEBULY_API_KEY")
    if not api_key:
        raise APIKeyNotProvidedError(
            "API key not provided and not found in environment"
        )
    return api_key
