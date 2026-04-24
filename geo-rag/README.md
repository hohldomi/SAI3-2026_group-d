# GeoRAG — Switzerland Geography Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Swiss geography.  
Built as a group project for the course **Building AI Applications (SAI3)** at Bern University of Applied Sciences.

---

## What it does

- Answers natural language questions about Swiss cities, mountains, lakes, and cantons
- Combines structured GeoNames data with Wikipedia summaries
- Retrieves the most relevant passages and generates grounded answers via an LLM

Example queries:
- *"What is the population of Bern?"*
- *"Tell me about Zermatt"*
- *"What mountains are near Interlaken?"*
- *"Explain the significance of the Matterhorn"*

---

## Project structure

```
geo-rag/
├── data/
│   ├── raw/               # CH.txt (GeoNames), downloaded Wikipedia summaries
│   └── processed/         # corpus.jsonl — final text passages
├── src/
│   ├── pipeline/
│   │   ├── geonames.py    # GeoNames → text passages
│   │   ├── wikipedia.py   # Wikipedia enrichment
│   │   └── build_corpus.py
│   ├── retrieval/
│   │   ├── embed.py       # Embedding with sentence-transformers
│   │   ├── index.py       # FAISS index build + load
│   │   └── retrieve.py    # Query → top-k passages
│   ├── generation/
│   │   ├── prompt.py      # Prompt templates
│   │   └── llm.py         # LLM interface (Ollama / API)
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
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_ORG/geo-rag.git
cd geo-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download GeoNames data

```bash
# Download CH.zip from GeoNames and extract
curl -O https://download.geonames.org/export/dump/CH.zip
unzip CH.zip -d data/raw/
```

### 3. Build the corpus

```bash
python -m src.pipeline.build_corpus
# Output: data/processed/corpus.jsonl (~10–15 MB)
```

### 4. Build the vector index

```bash
python -m src.retrieval.index
# Output: data/processed/geo_index.faiss + geo_index.pkl
```

### 5. Run the chatbot

```bash
python main.py
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=intfloat/multilingual-e5-small
TOP_K=5
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

- Python 3.11+
- [sentence-transformers](https://www.sbert.net/) — embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — vector search
- [Ollama](https://ollama.com/) — local LLM inference
- Pandas / NumPy — data processing
- Jupyter — experimentation

---

## Team

| Name | Role |
|------|------|
| Person A | Data pipeline + embeddings |
| Person B | Retrieval + evaluation |
| Person C | LLM integration + UI |

---

## Course

SAI3 — Building AI Applications  
Bern University of Applied Sciences, 2025
