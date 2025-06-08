from openai import OpenAI
from utils.helper import get_openai_api_key

def initialize_openai_client():
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key is not set in the environment variables.")
    return OpenAI(api_key=api_key)

client = initialize_openai_client()