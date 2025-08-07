# utils.py
import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (works locally, Streamlit ignores this if using Secrets)
load_dotenv(override=True)

def get_api_key():
    """Fetch the OpenAI API key from environment variables."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Set it in .env or Streamlit Secrets.")
    
    # Debug print: shows only first 10 chars
    print("Using API Key:", key[:-10], "...")
    return key

def hash_pair(job: str, resume: str) -> str:
    """Create a unique hash for job + resume combination."""
    return hashlib.sha256((job + resume).encode()).hexdigest()

def count_tokens(text: str) -> int:
    """Rudimentary token count (approx. by splitting words)."""
    return len(text.split())

def reset_app():
    """Clear all session state and rerun app."""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
