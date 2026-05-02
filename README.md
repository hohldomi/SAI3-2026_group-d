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
│   └── processed/         # corpus.jsonl — final text passages
├── src/
│   ├── pipeline/
│   │   ├── geonames.py    # GeoNames → text passages
│   │   ├── wikipedia.py   # Wikipedia enrichment (parallel, 8 threads)
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

## Quickstart

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Python 3.11+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/hohldomi/SAI3-2026_group-d.git
cd SAI3-2026_group-d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 4. Download GeoNames data

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://download.geonames.org/export/dump/CH.zip" -OutFile "CH.zip"
Expand-Archive -Path "CH.zip" -DestinationPath "data/raw/"
```

**Mac/Linux:**
```bash
curl -O https://download.geonames.org/export/dump/CH.zip
unzip CH.zip -d data/raw/
```

### 5. Start ChromaDB

```bash
docker compose up chromadb -d
```

### 6. Build the corpus

```bash
python -m src.pipeline.build_corpus
# Output: data/processed/corpus.jsonl (~10–15 MB)
# Note: Wikipedia enrichment runs in parallel (8 threads) — takes ~15–30 min
```

### 7. Build the vector index

```bash
python -m src.retrieval.index
# Embeds and loads 29,000+ passages into ChromaDB (~12 min on CPU)
```

### 8. Run the chatbot

```bash
python main.py
```

**Single query with passage display:**
```bash
python main.py --query "What canton is Lyss in?" --verbose
```

---

## Running with Docker (fully containerized)

```bash
docker compose up
```

This starts ChromaDB and Ollama automatically. On first startup, `llama3.2` (~2 GB) is downloaded and cached in a persistent Docker volume — no manual `ollama pull` needed.

> **Note:** Steps 6 and 7 (corpus building and indexing) must be run locally before starting Docker, as they require the GeoNames file and write to `data/processed/`.

---

## Interactive mode

```
GeoRAG — Switzerland Geography Assistant
Type 'quit' to exit, 'verbose' to toggle passage display.

You: What canton is Lyss in?
Assistant: Lyss is located in the canton of Bern.

You: verbose
Verbose mode: on

You: What is the Matterhorn?
--- Retrieved passages ---
  [0.961] Matterhorn: Matterhorn is a mountain straddling the border between Switzerland and Italy...
  ...
Assistant: The Matterhorn is one of Switzerland's most iconic peaks, rising 4,478 metres...
```

---

## Configuration

Copy `.env.example` to `.env`:

```env
# LLM
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434   # use http://ollama:11434 inside Docker

# Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# Retrieval
TOP_K=5
MIN_SCORE=0.35

# ChromaDB
CHROMA_HOST=localhost                # use chromadb inside Docker
CHROMA_PORT=8000
COLLECTION_NAME=switzerland_geo

# Paths
GEONAMES_PATH=data/raw/CH.txt
CORPUS_PATH=data/processed/corpus.jsonl
```

---

## Data sources

| Source | License | Size |
|--------|---------|------|
| [GeoNames CH](https://download.geonames.org/export/dump/) | CC BY 4.0 | ~5 MB raw |
| Wikipedia (via `wikipedia` Python package) | CC BY-SA 4.0 | ~7–10 MB summaries |

Both sources are freely usable for academic projects.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`intfloat/multilingual-e5-small`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| LLM inference | [Ollama](https://ollama.com/) (`llama3.2`) |
| Data processing | Pandas, NumPy |
| Containerization | Docker Compose |
| Experimentation | Jupyter |

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
