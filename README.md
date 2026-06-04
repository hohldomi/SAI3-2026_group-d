# GeoRAG — Switzerland Geography Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Swiss geography.  
Built as a group project for the course **Building AI Applications (SAI3)** at Bern University of Applied Sciences.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## What it does

- Answers natural language questions about Swiss cities, mountains, lakes, and cantons
- Combines structured GeoNames data with Wikipedia summaries
- Retrieves the most relevant passages and generates grounded answers via a local LLM

**Example queries:**
- *"What is the population of Bern?"*
- *"Tell me about Zermatt"*
- *"What canton is Lyss in?"*
- *"How high is the Matterhorn?"*

---

## Project structure

```
SAI3-2026_group-d/
├── data/
│   ├── raw/               # CH.txt (GeoNames, git-ignored)
│   ├── processed/         # corpus.jsonl — final text chunks (git-ignored)
│   └── evaluation/        # Evaluation results (JSON, CSV, Markdown)
├── src/
│   ├── pipeline/
│   │   ├── geonames.py    # GeoNames → text passages (canton mapping, feature codes)
│   │   ├── wikipedia.py   # Wikipedia enrichment (REST API, Token Bucket, 2 workers)
│   │   ├── chunker.py     # Recursive chunking of passages
│   │   └── build_corpus.py
│   ├── retrieval/
│   │   ├── embed.py       # Embedding with sentence-transformers
│   │   ├── index.py       # ChromaDB index build + load
│   │   └── retrieve.py    # Query → top-k passages
│   ├── generation/
│   │   ├── prompt.py      # Prompt templates
│   │   └── llm.py         # LLM interface (Ollama)
│   └── evaluation/
│       ├── metrics.py     # Recall@k, MRR
│       ├── run_evaluation.py  # Full evaluation pipeline → saves results to data/evaluation/
│       └── test_queries.json  # 20 test queries with expected results
├── docs/
│   └── architecture.md
├── .env.example
├── .gitignore
├── docker-compose.yml     # ChromaDB + Ollama + App services
├── Dockerfile
├── README.md
└── requirements.txt
```
---

## Getting started (first time)

Everything runs inside Docker — you don't need to install Python or any libraries manually.  
Requirements: **Docker Desktop** and **Git**.

### Step 1 — Clone the repository

```powershell
git clone https://github.com/hohldomi/SAI3-2026_group-d.git
cd SAI3-2026_group-d
```

### Step 2 — Download the GeoNames data

This file contains the raw geographic data (cities, mountains, lakes, etc.) that the assistant learns from.  
It is not included in the repository because of its size.

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://download.geonames.org/export/dump/CH.zip" -OutFile "CH.zip"
Expand-Archive -Path "CH.zip" -DestinationPath "data/raw/"
```

**Mac / Linux:**
```bash
curl -O https://download.geonames.org/export/dump/CH.zip
unzip CH.zip -d data/raw/
```

### Step 3 — Download the Wikipedia cache (optional, recommended for a faster build)

A pre-built cache of all 13'058 Wikipedia summaries is available as a 
release asset, skipping the ~1h 40min enrichment step:

1. Download `wiki_cache.json` from the [latest release](https://github.com/hohldomi/SAI3-2026_group-d/releases/latest)
2. Place it at `data/raw/wiki_cache.json`

Without the cache, `build_corpus` fetches all summaries from scratch 
(~1h 40min min, requires internet access).

### Step 4 — Start all services

```powershell
docker compose up -d
```

> On first start, the AI model (`llama3.2`, ~2 GB) is downloaded automatically. This happens only once.

### Step 5 — Build the knowledge base

Before the assistant can answer questions, it needs to process and index all geographic data.  
Run these two commands — **you only need to do this once per fresh setup**:

```powershell
# Build and chunk text passages from raw data (~1h 40min first time, >1min with cache)
docker compose run --rm app python -m pipeline.build_corpus

# Embed all chunks and load into ChromaDB (~8 min)
docker compose run --rm app python -m retrieval.index
```

> **Wikipedia enrichment:** The corpus build fetches up to 12-sentence summaries from Wikipedia for ~13,000 significant Swiss places (cities > 500 inhabitants, mountains > 1500 m, lakes, rivers, cantons, regions). Results are cached in `data/raw/wiki_cache.json` — subsequent builds skip all Wikipedia requests and finish in >1 minute.

### Step 6 — Open the assistant

**http://localhost:8501**

---

## After a restart

ChromaDB persists its data in a Docker named volume (`chroma_data`) and the corpus is stored as a local file (`data/processed/corpus.jsonl`). **Neither needs to be rebuilt after a normal restart.**

```powershell
# Stop
docker compose down

# Start again — corpus and index are still intact
docker compose up -d

# Open browser
# http://localhost:8501
```

> **Full reset:** If you used `docker compose down -v`, the `chroma_data` volume is deleted and the index must be rebuilt. The corpus file is unaffected and does not need to be rebuilt.
>
> ```powershell
> docker compose up -d
> docker compose run --rm app python -m retrieval.index   # ~8 min
> ```

---

## Evaluation

Run the full evaluation pipeline after building corpus and index:

```powershell
docker compose run --rm app python -m evaluation.run_evaluation
```

Results are saved to `data/evaluation/`:
- `results.json` — all metrics in machine-readable form
- `results_report.md` — human-readable summary (Recall@k, MRR, per-query table)
- `per_query.csv` — per-query breakdown for Excel/Presentation

**Current results (corpus v1, 36,270 passages, 9.96 MB):**

| Metric | Score |
|--------|-------|
| Recall@1 | 0.750 |
| Recall@3 | 0.750 |
| Recall@5 | **0.800** |
| MRR | **0.762** |
| Queries evaluated | 20 |

---

## How it works

GeoRAG uses **Retrieval-Augmented Generation (RAG)**:

1. **Your question** is converted into a vector embedding
2. **ChromaDB** finds the most relevant geographic chunks using cosine similarity
3. **Ollama/llama3.2** reads those chunks and generates a grounded answer

### Corpus pipeline

1. **GeoNames ingestion** — each row from `CH.txt` is converted into a rich text passage including canton, elevation, population, alternate names, and timezone
2. **Wikipedia enrichment** — for significant places, up to 12 sentences are fetched from Wikipedia via REST API and appended. A global Token Bucket (2 req/s) prevents rate limiting. Results are cached locally.
3. **Recursive chunking** — long passages are split into overlapping chunks (~400 chars, 80-char overlap)

---

## Configuration

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | LLM for answer generation |
| `TOP_K` | `5` | Number of passages retrieved per query |
| `MIN_SCORE` | `0.35` | Minimum cosine similarity threshold |
| `COLLECTION_NAME` | `switzerland_geo` | ChromaDB collection name |

---

## Data sources

| Source | License | Description |
|--------|---------|-------------|
| [GeoNames CH](https://download.geonames.org/export/dump/) | CC BY 4.0 | Swiss place names database |
| Wikipedia (REST API) | CC BY-SA 4.0 | Geographic summaries, up to 12 sentences |

---

## Team

| Name | Role |
|------|------|
| Dominik Hohl | Data pipeline, Wikipedia enrichment, embeddings, retrieval & evaluation |
| Brad Bustillos | LLM integration, user interface & report |
| Micha Streit | Local testing & quality assurance |

---

## AI assistance

This project was developed with the support of Claude (Anthropic) as an 
AI coding assistant. AI-generated code and text were not adopted 
uncritically — all suggestions were reviewed, tested, and adapted by the 
team. This applies to code, documentation, and commit messages throughout 
the repository.

---

## Course

**SAI3 — Building AI Applications**  
Bern University of Applied Sciences, 2026