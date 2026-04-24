"""
Build the text corpus from GeoNames + Wikipedia.
Run with: python -m src.pipeline.build_corpus

Outputs: data/processed/corpus.jsonl
"""

import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.pipeline.geonames import load_geonames, build_passages
from src.pipeline.wikipedia import enrich_dataframe, merge_passages

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RAW_PATH = os.getenv('GEONAMES_PATH', 'data/raw/CH.txt')
OUTPUT_PATH = os.getenv('CORPUS_PATH', 'data/processed/corpus.jsonl')
TARGET_MB = 12.0


def measure_corpus(passages: list[str]) -> dict:
    total_bytes = sum(len(p.encode('utf-8')) for p in passages)
    total_words = sum(len(p.split()) for p in passages)
    return {
        'num_passages': len(passages),
        'total_words': total_words,
        'size_mb': round(total_bytes / 1_000_000, 2),
    }


def main():
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    logger.info("Loading GeoNames from %s", RAW_PATH)
    df = load_geonames(RAW_PATH)
    logger.info("Loaded %d rows", len(df))

    logger.info("Converting rows to passages...")
    df = build_passages(df)

    logger.info("Enriching with Wikipedia...")
    df = enrich_dataframe(df)
    df = merge_passages(df)

    # Sort by population desc so most important places come first
    df = df.sort_values('population', ascending=False)

    records = df[['geonameid', 'name', 'feature_class',
                  'latitude', 'longitude', 'full_passage']].rename(
        columns={'full_passage': 'passage'}).to_dict('records')

    stats = measure_corpus([r['passage'] for r in records])
    logger.info("Corpus stats: %s", stats)

    if stats['size_mb'] < 10:
        logger.warning(
            "Corpus is %.1f MB — below the 10 MB minimum. "
            "Consider fetching longer Wikipedia summaries.", stats['size_mb']
        )
    elif stats['size_mb'] > 20:
        logger.warning(
            "Corpus is %.1f MB — above the 20 MB limit. "
            "Consider filtering out low-priority feature classes.", stats['size_mb']
        )
    else:
        logger.info("Corpus size OK: %.1f MB (target: 10–20 MB)", stats['size_mb'])

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info("Saved %d passages to %s", len(records), OUTPUT_PATH)


if __name__ == '__main__':
    main()
