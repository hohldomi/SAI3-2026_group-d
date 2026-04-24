"""
Retrieve the top-k most relevant passages for a query.
"""

import os
import numpy as np
import faiss
from dotenv import load_dotenv

from src.retrieval.embed import embed_query

load_dotenv()

FEATURE_KEYWORDS = {
    'T': ['mountain', 'peak', 'alps', 'summit', 'glacier', 'pass', 'col', 'alp'],
    'H': ['lake', 'river', 'stream', 'waterfall', 'sea', 'pond'],
    'P': ['city', 'town', 'village', 'population', 'municipality', 'inhabitants'],
    'A': ['canton', 'district', 'region', 'county', 'administrative'],
}


def detect_feature_class(query: str) -> str | None:
    """Heuristic: infer which GeoNames feature class the query is about."""
    q = query.lower()
    for cls, keywords in FEATURE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return cls
    return None


def retrieve(query: str,
             index: faiss.IndexFlatIP,
             passages: list[dict],
             k: int = None,
             min_score: float = None) -> list[dict]:
    """
    Embed query, search index, return top-k results above min_score.
    Optionally filters by detected feature class.
    """
    k = k or int(os.getenv('TOP_K', 5))
    min_score = min_score or float(os.getenv('MIN_SCORE', 0.30))

    q_emb = embed_query(query).reshape(1, -1).astype('float32')

    # Fetch more candidates if we're going to filter
    fetch_k = k * 3
    scores, indices = index.search(q_emb, min(fetch_k, index.ntotal))

    feature_class = detect_feature_class(query)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if float(score) < min_score:
            continue
        doc = passages[idx].copy()
        doc['score'] = round(float(score), 4)

        # If feature class was detected, prefer matching docs (don't hard-exclude)
        if feature_class and doc.get('feature_class') == feature_class:
            doc['_preferred'] = True
        results.append(doc)

    # Sort: preferred docs first, then by score
    results.sort(key=lambda d: (not d.get('_preferred', False), -d['score']))
    return results[:k]
