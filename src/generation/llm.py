"""
LLM interface — supports Ollama (local) and a generic HTTP API.
Configure via .env:
  OLLAMA_MODEL=llama3.2          # for Ollama
  UNIVERSITY_API_URL=https://... # for university API
  UNIVERSITY_API_KEY=...
"""

import os
import requests
import logging
from dotenv import load_dotenv
from src.generation.prompt import build_messages

load_dotenv()
logger = logging.getLogger(__name__)


def _call_ollama(messages: list[dict], model: str) -> str:
    import ollama
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=messages,
        options={'temperature': 0.1},
    )
    return response['message']['content']


def _call_university_api(messages: list[dict], model: str) -> str:
    url = os.getenv('UNIVERSITY_API_URL')
    key = os.getenv('UNIVERSITY_API_KEY', '')
    if not url:
        raise ValueError("UNIVERSITY_API_URL not set in .env")

    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.1,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Standard OpenAI-compatible response format
    return data['choices'][0]['message']['content']


def generate(query: str, retrieved_docs: list[dict]) -> str:
    """Generate an answer given a query and retrieved context passages."""
    messages = build_messages(query, retrieved_docs)
    model = os.getenv('OLLAMA_MODEL', 'llama3.2')

    if os.getenv('UNIVERSITY_API_URL'):
        logger.debug("Using university API")
        return _call_university_api(messages, model)
    else:
        logger.debug("Using Ollama with model %s", model)
        return _call_ollama(messages, model)
