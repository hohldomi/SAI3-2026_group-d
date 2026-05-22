# GeoRAG — Switzerland Geography Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Swiss geography.  
Built as a group project for the course **Building AI Applications (SAI3)** at Bern University of Applied Sciences.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
│   │   ├── wikipedia.py   # Wikipedia enrichment (REST API, Token Bucket, 4 workers)
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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   └── 03_retrieval_evaluation.ipynb
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── architecture.md
├── main.py                # CLI entry point
├── docker-compose.yml     # ChromaDB + Ollama + App services
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
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

### Step 3 — Start all services

```powershell
docker compose up -d
```

> On first start, the AI model (`llama3.2`, ~2 GB) is downloaded automatically. This happens only once.

### Step 4 — Build the knowledge base

Before the assistant can answer questions, it needs to process and index all geographic data.  
Run these two commands — **you only need to do this once per fresh setup**:

```powershell
# Build and chunk text passages from raw data (~45 min first time, ~3 min with cache)
docker compose run --rm app python -m pipeline.build_corpus

# Embed all chunks and load into ChromaDB (~8 min)
docker compose run --rm app python -m retrieval.index
```

> **Wikipedia enrichment:** The corpus build fetches up to 12-sentence summaries from Wikipedia for ~13,000 significant Swiss places (cities > 500 inhabitants, mountains > 1500 m, lakes, rivers, cantons, regions). Results are cached in `data/raw/wiki_cache.json` — subsequent builds skip all Wikipedia requests and finish in ~3 minutes.

### Step 5 — Open the assistant

**http://localhost:8501**

---

## After every restart

ChromaDB does not persist data between `docker compose down` restarts.  
After each restart, run the following commands before using the app:

```powershell
# 1. Start services
docker compose up -d

# 2. Rebuild corpus (~3 min — served from wiki cache, no internet needed)
docker compose run --rm app python -m pipeline.build_corpus

# 3. Rebuild index (~8 min)
docker compose run --rm app python -m retrieval.index

# 4. Open browser
# http://localhost:8501
```

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
| Person A | Data pipeline + embeddings |
| Person B | Retrieval + evaluation |
| Person C | LLM integration + UI |

---

## Course

**SAI3 — Building AI Applications**  
Bern University of Applied Sciences, 2026