"""
Prompt templates for the geography RAG chatbot.
"""

SYSTEM_PROMPT = """You are a geography assistant specialising in Switzerland and nearby regions.
Answer questions using ONLY the provided context passages.
If the context does not contain enough information, say so clearly — do not invent facts.
Always mention population, elevation, or coordinates when they are available and relevant.
Keep answers concise but informative (2–4 sentences for simple queries, more for complex ones)."""


def build_messages(query: str, retrieved_docs: list[dict]) -> list[dict]:
    """Build the messages list for the LLM API call."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['passage']}")
    context = '\n\n'.join(context_parts)

    user_content = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""

    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_content},
    ]
