# Architecture

## System overview

GeoRAG is a two-phase system:

**Offline (run once)**
1. Load GeoNames CH.txt → convert rows to prose passages
2. Enrich with Wikipedia summaries
3. Embed all passages with `multilingual-e5-small`
4. Store vectors in a FAISS index

**Online (per query)**
1. Embed the user query
2. Search FAISS for top-5 most similar passages
3. Pass query + retrieved passages to an LLM
4. Return grounded answer

## Embedding model

`intfloat/multilingual-e5-small` (~120 MB, 384-dim vectors)

- Handles German, French, Italian place names natively
- e5 models require `passage: ` prefix for documents and `query: ` prefix for queries
- Normalised embeddings → inner product == cosine similarity

## Vector index

FAISS `IndexFlatIP` — exact search, no approximation.  
With ~30–50k passages this is fast enough (< 50ms per query on CPU).

## LLM

Defaults to Ollama (local). Set `UNIVERSITY_API_URL` in `.env` to use the university API instead.  
Temperature is set to 0.1 for factual, deterministic answers.

## Data sources

| Source | URL | License |
|--------|-----|---------|
| GeoNames | https://download.geonames.org/export/dump/ | CC BY 4.0 |
| Wikipedia | via `wikipedia` Python package | CC BY-SA 4.0 |
