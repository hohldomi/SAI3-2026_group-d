"""
Evaluation metrics for retrieval quality.

Usage:
    python -m src.evaluation.metrics
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TEST_QUERIES_PATH = 'src/evaluation/test_queries.json'


def recall_at_k(results: list[dict], relevant_name: str, k: int = 5) -> int:
    top_names = [r['name'].lower() for r in results[:k]]
    return 1 if relevant_name.lower() in top_names else 0


def mrr(results: list[dict], relevant_name: str) -> float:
    for i, r in enumerate(results):
        if r['name'].lower() == relevant_name.lower():
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(results: list[dict], relevant_names: list[str], k: int = 5) -> float:
    top_names = {r['name'].lower() for r in results[:k]}
    relevant = {n.lower() for n in relevant_names}
    return len(top_names & relevant) / k


def run_evaluation(retrieve_fn, test_queries_path: str = TEST_QUERIES_PATH):
    """
    Run retrieval evaluation on test queries.

    test_queries.json format:
    [
      {"query": "population of Bern", "relevant": "Bern"},
      ...
    ]
    """
    with open(test_queries_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    recall_scores = []
    mrr_scores = []

    for tc in test_cases:
        results = retrieve_fn(tc['query'])
        recall_scores.append(recall_at_k(results, tc['relevant']))
        mrr_scores.append(mrr(results, tc['relevant']))

    print(f"Recall@5:  {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"MRR:       {sum(mrr_scores)/len(mrr_scores):.3f}")
    print(f"Evaluated: {len(test_cases)} queries")
    return {
        'recall_at_5': sum(recall_scores) / len(recall_scores),
        'mrr': sum(mrr_scores) / len(mrr_scores),
        'n': len(test_cases),
    }
