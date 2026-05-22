# GeoRAG — Evaluation Results

Generated: 2026-05-22 10:44

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| Recall@1 | **0.750** |
| Recall@3 | **0.750** |
| Recall@5 | **0.800** |
| MRR | **0.762** |
| Queries evaluated | 20 |

---

## Corpus Statistics

| Property | Value |
|----------|-------|
| Total passages | 36,270 |
| Corpus size | 9.96 MB |
| Total words | 1,631,105 |
| Avg words/passage | 45.0 |
| Indexed vectors | 36,270 |

### Passages by Feature Class

| Feature Class | Count |
|---------------|-------|
| A | 4011 |
| H | 3308 |
| L | 471 |
| P | 14153 |
| T | 14327 |

---

## Per-Query Results

| Hit | Query | Expected | Top-1 Result | Score | MRR |
|-----|-------|----------|--------------|-------|-----|
| ✓ | What is the population of Bern? | Bern | Bern | 0.952 | 1.000 |
| ✓ | Tell me about Zurich | Zürich | Zürich | 0.933 | 1.000 |
| ✓ | How high is the Matterhorn? | Matterhorn | Matterhorn | 0.956 | 1.000 |
| ✗ | What is the largest city in Switzerland? | Zürich | Trub | 0.938 | 0.000 |
| ✓ | Tell me about Lake Geneva | Lake Geneva | Lake Geneva | 0.944 | 1.000 |
| ✗ | Where is Interlaken? | Interlaken | Hinterburgseeli | 0.924 | 0.000 |
| ✗ | What mountains are near Zermatt? | Zermatt | Pointe du Mountet | 0.936 | 0.000 |
| ✓ | Population of Basel | Basel | Basel | 0.952 | 1.000 |
| ✗ | Tell me about the Rhine river | Rhein | Dachlisee | 0.930 | 0.000 |
| ✓ | What canton is Lucerne in? | Kanton Luzern | Kanton Luzern | 0.951 | 1.000 |
| ✓ | How high is Jungfrau? | Jungfrau | Jungfrau | 0.957 | 1.000 |
| ✓ | Tell me about Lake Zurich | Zürichsee | Zürichsee | 0.939 | 1.000 |
| ✓ | What is Graubünden? | Kanton Graubünden | Kanton Graubünden | 0.947 | 1.000 |
| ✓ | Elevation of Eiger | Eiger | Eiger | 0.950 | 1.000 |
| ✓ | Tell me about Lausanne | Lausanne | Lausanne | 0.933 | 1.000 |
| ✓ | What is the capital of Switzerland? | Bern | Lausanne | 0.933 | 0.250 |
| ✓ | Tell me about the Aare river | Aare | Aare | 0.939 | 1.000 |
| ✓ | How high is the Finsteraarhorn? | Finsteraarhorn | Finsteraarhorn | 0.955 | 1.000 |
| ✓ | Tell me about Geneva | Geneva | Geneva | 0.927 | 1.000 |
| ✓ | What is the largest lake in Switzerland? | Grand Lac | Grand Lac | 0.945 | 1.000 |

---

## Notes

- **Recall@k**: fraction of queries where the relevant result appears in top-k
- **MRR**: Mean Reciprocal Rank — higher is better (1.0 = always rank 1)
- Embedding model: `intfloat/multilingual-e5-small`
- LLM: `llama3.2` via Ollama
- Vector store: ChromaDB (cosine similarity)
