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
- *"Explain the significance of the Matterhorn"*

---

## Project structure

```
SAI3-2026_group-d/
├── data/
│   ├── raw/               # CH.txt (GeoNames)
│   └── processed/         # corpus.jsonl — final text chunks
├── src/
│   ├── pipeline/
│   │   ├── geonames.py    # GeoNames → text passages
│   │   ├── wikipedia.py   # Wikipedia enrichment (parallel, 8 threads)
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
│       ├── metrics.py     # Recall@k, MRR, faithfulness
│       └── test_queries.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   └── 03_retrieval_evaluation.ipynb
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── architecture.md
├── main.py                # CLI entry point
├── docker-compose.yml     # ChromaDB + Ollama services
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting started

Everything runs inside Docker — you don't need to install Python or any libraries manually. The only two requirements are **Docker Desktop** and Git.

### What you need before you start

1. **Docker Desktop** — download and install from [docker.com](https://www.docker.com/products/docker-desktop/)
    1. Open Terminal or Command Prompt and run:
    ```bash
    docker --version
    ```
    2. Run the following command to see if Docker can pull and run containers correctly:
    ```bash
    docker run hello-world
    ```
2. **The GeoNames data file** — a publicly available database of Swiss place names

### Step 1 — Download the GeoNames data

This file contains the raw geographic data (cities, mountains, lakes, etc.) that the assistant learns from.

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

### Step 2 — Get the project

```bash
git clone https://github.com/hohldomi/SAI3-2026_group-d.git
cd SAI3-2026_group-d
```

### Step 3 — Start all services

This single command starts everything: the database, the AI model, and the web interface.

```bash
docker compose up
```

> On first start, the AI model (`llama3.2`, ~2 GB) is downloaded automatically. This happens only once — afterwards it is stored locally.

### Step 4 — Build the knowledge base (first time only)

Before the assistant can answer questions, it needs to process and index all the geographic data. Open a second terminal and run these two commands — **you only need to do this once**:

```bash
# Builds and chunks text passages from the raw data (~30–45 minutes)
docker compose run --rm -e PYTHONPATH=/app/src app python -m pipeline.build_corpus

# Loads all chunks into the search database (~15 minutes)
docker compose run --rm -e PYTHONPATH=/app/src app python -m retrieval.index
```

> **Note:** The corpus build fetches Wikipedia summaries for ~11,000 significant places (cities with population > 500, mountains above 2500 m, lakes, and cantons) using 8 parallel threads. Build time depends on your internet connection.

After this, the knowledge base is saved permanently — you won't need to repeat these steps unless the data changes.

> **After `docker compose down`:** ChromaDB data is not persisted between restarts. Run both commands above again after each restart.

### Step 5 — Open the assistant

Once everything is running, open your browser and go to:

**http://localhost:8501**

You'll see the GeoRAG web interface where you can type your questions.

---

## Using the assistant

Type your question in plain English (or German) and press Enter. The assistant will search through its knowledge base and generate an answer.

**Example questions:**
- *"What is the population of Bern?"*
- *"Tell me about Zermatt"*
- *"What canton is Lyss in?"*
- *"Explain the significance of the Matterhorn"*

Under each answer you can expand the **Sources** section to see exactly which passages the assistant used to form its response.

---

## Configuration

Settings are managed through a `.env` file. Copy the example file to get started:

```bash
cp .env.example .env
```

The most relevant settings:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `OLLAMA_MODEL` | `llama3.2` | Which AI model generates the answers |
| `TOP_K` | `5` | How many chunks are retrieved per query |
| `MIN_SCORE` | `0.35` | Minimum relevance score — lower means more (but less relevant) results |
| `COLLECTION_NAME` | `switzerland_geo` | Name of the knowledge base in the database |

---

## How it works

GeoRAG uses a technique called **Retrieval-Augmented Generation (RAG)**:

1. **Your question** is converted into a mathematical representation (an "embedding")
2. **The search database** (ChromaDB) finds the most relevant geographic chunks using this representation
3. **The AI model** (Ollama/llama3.2) reads those chunks and writes a grounded answer

This means the assistant only answers based on real data — it doesn't invent facts.

### Corpus pipeline

The knowledge base is built in three stages:

1. **GeoNames ingestion** — each row from `CH.txt` is converted into a readable text passage (e.g. *"Bern is a populated place in Switzerland with a population of 133,000..."*)
2. **Wikipedia enrichment** — for significant places (cities > 500 inhabitants, mountains > 2500 m, all lakes and cantons), up to 5 sentences are fetched from Wikipedia and appended to the passage
3. **Recursive chunking** — long passages are split into overlapping chunks of ~400 characters with 80-character overlap, so the retrieval system can pinpoint specific facts rather than retrieving entire articles

---

## Data sources

| Source | License | Size |
|--------|---------|------|
| [GeoNames CH](https://download.geonames.org/export/dump/) | CC BY 4.0 | ~5 MB raw |
| Wikipedia (via `wikipedia` Python package) | CC BY-SA 4.0 | ~7–10 MB summaries |

Both sources are freely usable for academic projects.

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