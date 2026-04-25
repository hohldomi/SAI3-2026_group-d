"""
Build and persist a ChromaDB collection from the text corpus.
Run with: python -m src.retrieval.index
"""

import json
import os
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from src.retrieval.embed import embed_passages

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CORPUS_PATH      = os.getenv('CORPUS_PATH', 'data/processed/corpus.jsonl')
COLLECTION_NAME  = os.getenv('COLLECTION_NAME', 'switzerland_geo')
CHROMA_HOST      = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT      = int(os.getenv('CHROMA_PORT', 8000))


def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def load_corpus(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def get_or_create_collection(client: chromadb.HttpClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def build_index(passages: list[dict], collection: chromadb.Collection,
                batch_size: int = 100):
    """Embed all passages and upsert into ChromaDB in batches."""
    texts = [p['passage'] for p in passages]
    logger.info("Embedding %d passages...", len(texts))
    embeddings = embed_passages(texts)

    logger.info("Upserting into ChromaDB collection '%s'...", COLLECTION_NAME)
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        batch_emb = embeddings[start:start + batch_size]

        collection.upsert(
            ids=[str(p['geonameid']) for p in batch],
            embeddings=batch_emb.tolist(),
            documents=[p['passage'] for p in batch],
            metadatas=[{
                'name':          p['name'],
                'feature_class': str(p.get('feature_class', '')),
                'latitude':      float(p.get('latitude', 0) or 0),
                'longitude':     float(p.get('longitude', 0) or 0),
            } for p in batch],
        )
        logger.info("  Upserted batch %d-%d", start, start + len(batch))


def load_index(collection_name: str = None) -> chromadb.Collection:
    """Return the ChromaDB collection (replaces the old FAISS load_index)."""
    client = get_client()
    name = collection_name or COLLECTION_NAME
    return client.get_collection(name=name)


def main():
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    client = get_client()
    collection = get_or_create_collection(client)

    logger.info("Loading corpus from %s", CORPUS_PATH)
    passages = load_corpus(CORPUS_PATH)
    logger.info("Loaded %d passages", len(passages))

    build_index(passages, collection)
    logger.info("Done. Collection contains %d vectors.", collection.count())


if __name__ == '__main__':
    main()
