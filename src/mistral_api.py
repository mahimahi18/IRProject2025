import os
import openai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Check if the API key is loaded correctly
api_key = os.getenv("TOGETHER_API_KEY")

# If the key is None or empty, it means the .env file might not be loaded correctly.
if api_key is None or api_key == "":
    print("API key is not set properly in the .env file.")
else:
    openai.api_key = api_key
    openai.api_base = "https://api.together.xyz/v1"

def query_mistral(prompt: str, model="mistralai/Mistral-7B-Instruct-v0.1") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for academic research."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    return response['choices'][0]['message']['content']
