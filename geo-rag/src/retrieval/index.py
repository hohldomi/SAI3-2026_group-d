"""
Build and persist a FAISS index from the text corpus.
Run with: python -m src.retrieval.index
"""

import json
import os
import pickle
import logging
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

from src.retrieval.embed import embed_passages

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CORPUS_PATH = os.getenv('CORPUS_PATH', 'data/processed/corpus.jsonl')
INDEX_PATH = os.getenv('INDEX_PATH', 'data/processed/geo_index')


def load_corpus(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """IndexFlatIP = exact inner product (= cosine on normalised vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index


def save_index(index: faiss.IndexFlatIP, passages: list[dict], path: str):
    faiss.write_index(index, f'{path}.faiss')
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(passages, f)
    logger.info("Saved index to %s.faiss + %s.pkl", path, path)


def load_index(path: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    index = faiss.read_index(f'{path}.faiss')
    with open(f'{path}.pkl', 'rb') as f:
        passages = pickle.load(f)
    return index, passages


def main():
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s", CORPUS_PATH)
    passages = load_corpus(CORPUS_PATH)
    logger.info("Loaded %d passages", len(passages))

    texts = [p['passage'] for p in passages]
    logger.info("Embedding passages...")
    embeddings = embed_passages(texts)

    logger.info("Building FAISS index (dim=%d)...", embeddings.shape[1])
    index = build_index(embeddings)

    save_index(index, passages, INDEX_PATH)
    logger.info("Done. Index contains %d vectors.", index.ntotal)


if __name__ == '__main__':
    main()
