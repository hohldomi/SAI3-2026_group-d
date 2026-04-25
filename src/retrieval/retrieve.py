"""
Retrieve the top-k most relevant passages for a query using ChromaDB.
"""

import os
import chromadb
from dotenv import load_dotenv

from src.retrieval.embed import embed_query
from src.retrieval.index import load_index

load_dotenv()

FEATURE_KEYWORDS = {
    'T': ['mountain', 'peak', 'alps', 'summit', 'glacier', 'pass', 'col', 'alp',
          'berg', 'gipfel', 'gletscher'],
    'H': ['lake', 'river', 'stream', 'waterfall', 'pond', 'see', 'fluss', 'bach'],
    'P': ['city', 'town', 'village', 'population', 'municipality', 'inhabitants',
          'einwohner', 'gemeinde', 'stadt'],
    'A': ['canton', 'district', 'region', 'county', 'administrative',
          'kanton', 'bezirk'],
}


def detect_feature_class(query: str) -> str | None:
    """Heuristic: infer which GeoNames feature class the query is about."""
    q = query.lower()
    for cls, keywords in FEATURE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return cls
    return None


def retrieve(query: str,
             collection: chromadb.Collection,
             k: int = None,
             min_score: float = None) -> list[dict]:
    """
    Embed query, search ChromaDB collection, return top-k results.
    Signature change from FAISS version: takes a ChromaDB collection
    instead of (index, passages).
    """
    k = k or int(os.getenv('TOP_K', 5))
    min_score = min_score or float(os.getenv('MIN_SCORE', 0.30))

    q_emb = embed_query(query).tolist()

    feature_class = detect_feature_class(query)

    # Fetch more candidates if we plan to filter/rerank
    fetch_k = k * 3

    # Optional: filter by feature class directly in ChromaDB
    where = {"feature_class": feature_class} if feature_class else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # If feature-class filter returned too few results, retry without filter
    if feature_class and len(results['ids'][0]) < k:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

    docs = []
    for doc_id, document, metadata, distance in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        score = round(1 - distance / 2, 4)
        if score < min_score:
            continue

        docs.append({
            'id':            doc_id,
            'passage':       document,
            'name':          metadata.get('name', ''),
            'feature_class': metadata.get('feature_class', ''),
            'latitude':      metadata.get('latitude'),
            'longitude':     metadata.get('longitude'),
            'score':         score,
        })

    docs.sort(key=lambda d: -d['score'])
    return docs[:k]
