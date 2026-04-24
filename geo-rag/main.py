"""
GeoRAG — CLI chatbot entry point.

Usage:
    python main.py
    python main.py --query "What is the population of Bern?"
"""

import argparse
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)

INDEX_PATH = os.getenv('INDEX_PATH', 'data/processed/geo_index')
TOP_K = int(os.getenv('TOP_K', 5))

# Lazy-load heavy dependencies
_index = None
_passages = None


def load():
    global _index, _passages
    if _index is None:
        from src.retrieval.index import load_index
        print("Loading index...")
        _index, _passages = load_index(INDEX_PATH)
        print(f"Ready. {_index.ntotal} passages indexed.\n")


def ask(query: str, verbose: bool = False) -> str:
    load()
    from src.retrieval.retrieve import retrieve
    from src.generation.llm import generate

    docs = retrieve(query, _index, _passages, k=TOP_K)

    if verbose:
        print("\n--- Retrieved passages ---")
        for d in docs:
            print(f"  [{d['score']:.3f}] {d['name']}: {d['passage'][:80]}...")
        print()

    return generate(query, docs)


def main():
    parser = argparse.ArgumentParser(description='GeoRAG — Switzerland geography assistant')
    parser.add_argument('--query', '-q', type=str, help='Single query (non-interactive)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show retrieved passages')
    args = parser.parse_args()

    if args.query:
        answer = ask(args.query, verbose=args.verbose)
        print(answer)
        return

    # Interactive mode
    load()
    print("GeoRAG — Switzerland Geography Assistant")
    print("Type 'quit' to exit, 'verbose' to toggle passage display.\n")
    verbose = False

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if query.lower() == 'verbose':
            verbose = not verbose
            print(f"Verbose mode: {'on' if verbose else 'off'}")
            continue

        answer = ask(query, verbose=verbose)
        print(f"\nAssistant: {answer}\n")


if __name__ == '__main__':
    main()
