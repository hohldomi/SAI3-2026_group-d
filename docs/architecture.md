# GeoRAG — Architecture

## Component diagram

```
┌─────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE                    │
│                                                         │
│  CH.txt ──► geonames.py ──► wikipedia.py ──► chunker.py│
│  (GeoNames)   (passages)    (enrichment)    (chunks)    │
│                                  │                      │
│                          wiki_cache.json                │
│                          (local cache)                  │
│                                                         │
│  chunks ──► embed.py ──► index.py ──► ChromaDB         │
│           (e5-small)   (build index)  (vector store)   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE                     │
│                                                         │
│  User query                                             │
│      │                                                  │
│      ▼                                                  │
│  embed.py ──► ChromaDB ──► retrieve.py                  │
│  (embed query)  (top-5)    (filter + rerank)            │
│                                │                        │
│                                ▼                        │
│                           prompt.py ──► llm.py          │
│                           (build ctx)   (Ollama)        │
│                                │                        │
│                                ▼                        │
│                           Streamlit UI                  │
└─────────────────────────────────────────────────────────┘
```

---

## Corpus pipeline — technical details

### geonames.py

Each GeoNames row is converted to a natural-language passage using these fields:
`name`, `feature_class`, `feature_code`, `admin1_code` (→ canton name), `population`, `latitude`, `longitude`, `elevation`, `alternatenames` (up to 5), `timezone`.

Feature classes indexed: `P` (populated places > 500 inhabitants), `T` (mountains > 1500 m), `H` (water bodies), `A` (administrative divisions), `L` (regions).

### wikipedia.py

- Direct REST API: `en.wikipedia.org/w/api.php` (search) + `/api/rest_v1/page/summary/` (content)
- **Token Bucket** rate limiter: 2.0 req/s global across 4 parallel workers — prevents Wikipedia 429s
- Swiss article validation: 50+ keywords in DE/EN/FR/IT (`SWISS_KEYWORDS`)
- **Cache**: results stored in `data/raw/wiki_cache.json` — subsequent builds require zero HTTP requests
- Deduplication by name before fetching — avoids redundant requests for duplicate GeoNames entries
- Up to 12 sentences per article, `n_results=5` candidates searched per name

### chunker.py

- Strategy: recursive (sentence → word boundaries)
- Target chunk size: ~400 characters
- Overlap: 80 characters
- Result: 29,534 passages → 36,270 chunks

---

## Embedding model

`intfloat/multilingual-e5-small` — 384-dimensional vectors, ~471 MB

- Handles DE/FR/IT/Romansh place names natively
- Requires `passage: ` prefix for documents, `query: ` prefix for queries
- Normalised embeddings → cosine similarity via inner product

---

## Vector store

ChromaDB (HTTP, Docker) — collection `switzerland_geo`, cosine distance.

> **Ephemeral:** ChromaDB does not persist between `docker compose down` restarts. Corpus (~3 min with cache) and index (~8 min) must be rebuilt after each restart.

Similarity score conversion: `score = 1 - distance / 2` → range [0, 1].

---

## Retrieval — algorithm

1. Embed query with `multilingual-e5-small`
2. Heuristic feature-class detection from query keywords (`T` / `H` / `P` / `A`)
3. Query ChromaDB with optional `where` filter on `feature_class`
4. If filter returns < k results, retry without filter
5. Filter results by `MIN_SCORE` (default: 0.35)
6. Return top-k sorted by similarity score

---

## LLM

- Default: **Ollama** + `llama3.2` (local, no internet after initial download)
- Optional: University API via `UNIVERSITY_API_URL` in `.env`
- Temperature: 0.1 for factual, deterministic answers

---

## Known limitations

- **Semantic inference** ("largest city") not possible without reasoning layer — dense retrieval only matches surface semantics
- **Location proximity** ("mountains near Zermatt") relies on text similarity, not geographic distance
- **Multilingual mismatches** partially mitigated by SWISS_KEYWORDS and alternate names in passages