"""
Full evaluation pipeline — saves all results to data/evaluation/.

Run after corpus + index build:
    docker compose run --rm app python -m evaluation.run_evaluation

Outputs:
    data/evaluation/results.json        — all metrics in machine-readable form
    data/evaluation/results_report.md   — human-readable summary for report/presentation
    data/evaluation/per_query.csv       — per-query breakdown
"""

import json
import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_QUERIES_PATH = os.getenv('TEST_QUERIES_PATH', 'src/evaluation/test_queries.json')
CORPUS_PATH       = os.getenv('CORPUS_PATH', 'data/processed/corpus.jsonl')
OUTPUT_DIR        = Path('data/evaluation')
COLLECTION_NAME   = os.getenv('COLLECTION_NAME', 'switzerland_geo')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(results: list[dict], relevant_name: str, k: int) -> int:
    top_names = [r['name'].lower() for r in results[:k]]
    return 1 if relevant_name.lower() in top_names else 0


def mrr_score(results: list[dict], relevant_name: str) -> float:
    for i, r in enumerate(results):
        if r['name'].lower() == relevant_name.lower():
            return 1.0 / (i + 1)
    return 0.0


def average_score(results: list[dict], k: int) -> float:
    if not results:
        return 0.0
    return round(sum(r['score'] for r in results[:k]) / min(k, len(results)), 4)


# ---------------------------------------------------------------------------
# Corpus stats
# ---------------------------------------------------------------------------

def corpus_stats(path: str) -> dict:
    if not Path(path).exists():
        return {}
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    total_bytes = sum(len(r['passage'].encode('utf-8')) for r in records)
    total_words = sum(len(r['passage'].split()) for r in records)
    wiki_enriched = sum(1 for r in records
                        if len(r['passage'].split()) > 30)  # rough proxy
    feature_counts = {}
    for r in records:
        fc = r.get('feature_class', 'unknown')
        feature_counts[fc] = feature_counts.get(fc, 0) + 1
    return {
        'num_passages': len(records),
        'total_words': total_words,
        'size_mb': round(total_bytes / 1_000_000, 2),
        'avg_words_per_passage': round(total_words / len(records), 1) if records else 0,
        'feature_class_counts': feature_counts,
        'approx_wiki_enriched': wiki_enriched,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Corpus stats ---
    logger.info("Computing corpus statistics...")
    c_stats = corpus_stats(CORPUS_PATH)
    logger.info("Corpus: %s", c_stats)

    # --- Load retrieval pipeline ---
    logger.info("Loading retrieval index...")
    from retrieval.index import load_index
    from retrieval.retrieve import retrieve

    try:
        collection = load_index(COLLECTION_NAME)
        index_count = collection.count()
        logger.info("Index loaded: %d vectors", index_count)
    except Exception as e:
        logger.error("Could not load index: %s", e)
        logger.error("Run 'python -m retrieval.index' first.")
        return

    # --- Load test queries ---
    with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    logger.info("Loaded %d test queries", len(test_cases))

    # --- Run retrieval for each query ---
    per_query_rows = []
    recall1_scores = []
    recall3_scores = []
    recall5_scores = []
    mrr_scores = []

    for tc in test_cases:
        query    = tc['query']
        relevant = tc['relevant']

        results = retrieve(query, collection, k=5)

        r1  = recall_at_k(results, relevant, k=1)
        r3  = recall_at_k(results, relevant, k=3)
        r5  = recall_at_k(results, relevant, k=5)
        mrr = mrr_score(results, relevant)
        avg = average_score(results, k=5)

        recall1_scores.append(r1)
        recall3_scores.append(r3)
        recall5_scores.append(r5)
        mrr_scores.append(mrr)

        top1_name  = results[0]['name'] if results else 'N/A'
        top1_score = results[0]['score'] if results else 0.0

        per_query_rows.append({
            'query':      query,
            'relevant':   relevant,
            'recall@1':   r1,
            'recall@3':   r3,
            'recall@5':   r5,
            'mrr':        round(mrr, 4),
            'avg_score':  avg,
            'top1_name':  top1_name,
            'top1_score': top1_score,
            'hit':        '✓' if r5 else '✗',
        })

        logger.info("[%s] %s → top1: %s (%.3f) | R@5=%d MRR=%.3f",
                    '✓' if r5 else '✗', query, top1_name, top1_score, r5, mrr)

    # --- Aggregate metrics ---
    n = len(test_cases)
    metrics = {
        'recall_at_1': round(sum(recall1_scores) / n, 4),
        'recall_at_3': round(sum(recall3_scores) / n, 4),
        'recall_at_5': round(sum(recall5_scores) / n, 4),
        'mrr':         round(sum(mrr_scores) / n, 4),
        'n_queries':   n,
    }

    # --- Full results dict ---
    results_data = {
        'timestamp':    datetime.now().isoformat(),
        'collection':   COLLECTION_NAME,
        'index_vectors': index_count,
        'corpus':       c_stats,
        'metrics':      metrics,
        'per_query':    per_query_rows,
    }

    # --- Save results.json ---
    json_path = OUTPUT_DIR / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    # --- Save per_query.csv ---
    csv_path = OUTPUT_DIR / 'per_query.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=per_query_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_query_rows)
    logger.info("Saved: %s", csv_path)

    # --- Save results_report.md ---
    fc_table = '\n'.join(
        f"| {fc} | {count} |"
        for fc, count in sorted(c_stats.get('feature_class_counts', {}).items())
    )

    pq_table = '\n'.join(
        f"| {r['hit']} | {r['query']} | {r['relevant']} | {r['top1_name']} "
        f"| {r['top1_score']:.3f} | {r['mrr']:.3f} |"
        for r in per_query_rows
    )

    md = f"""# GeoRAG — Evaluation Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| Recall@1 | **{metrics['recall_at_1']:.3f}** |
| Recall@3 | **{metrics['recall_at_3']:.3f}** |
| Recall@5 | **{metrics['recall_at_5']:.3f}** |
| MRR | **{metrics['mrr']:.3f}** |
| Queries evaluated | {metrics['n_queries']} |

---

## Corpus Statistics

| Property | Value |
|----------|-------|
| Total passages | {c_stats.get('num_passages', 'N/A'):,} |
| Corpus size | {c_stats.get('size_mb', 'N/A')} MB |
| Total words | {c_stats.get('total_words', 'N/A'):,} |
| Avg words/passage | {c_stats.get('avg_words_per_passage', 'N/A')} |
| Indexed vectors | {index_count:,} |

### Passages by Feature Class

| Feature Class | Count |
|---------------|-------|
{fc_table}

---

## Per-Query Results

| Hit | Query | Expected | Top-1 Result | Score | MRR |
|-----|-------|----------|--------------|-------|-----|
{pq_table}

---

## Notes

- **Recall@k**: fraction of queries where the relevant result appears in top-k
- **MRR**: Mean Reciprocal Rank — higher is better (1.0 = always rank 1)
- Embedding model: `intfloat/multilingual-e5-small`
- LLM: `llama3.2` via Ollama
- Vector store: ChromaDB (cosine similarity)
"""

    md_path = OUTPUT_DIR / 'results_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    logger.info("Saved: %s", md_path)

    # --- Print summary ---
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"  Recall@1:  {metrics['recall_at_1']:.3f}")
    print(f"  Recall@3:  {metrics['recall_at_3']:.3f}")
    print(f"  Recall@5:  {metrics['recall_at_5']:.3f}")
    print(f"  MRR:       {metrics['mrr']:.3f}")
    print(f"  Corpus:    {c_stats.get('size_mb', '?')} MB, "
          f"{c_stats.get('num_passages', '?'):,} passages")
    print(f"  Index:     {index_count:,} vectors")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("="*50)


if __name__ == '__main__':
    main()
