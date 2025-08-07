# embedding_utils.py
import openai
from utils import get_api_key

# Set API key for OpenAI
openai.api_key = get_api_key()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Generates an embedding vector for the given text using OpenAI's embedding API.
    Returns the embedding as a list of floats.
    """
    try:
        text = text.replace("\n", " ")  # Clean up newlines
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
