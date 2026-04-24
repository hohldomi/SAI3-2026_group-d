"""
Enrich GeoNames passages with Wikipedia summaries.
Fetches 3-sentence summaries for significant places.
"""

import time
import logging
import wikipedia
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_summary(name: str, country: str = "Switzerland",
                  n_sentences: int = 3) -> str | None:
    """Fetch first n_sentences from a Wikipedia article."""
    try:
        results = wikipedia.search(f"{name} {country}", results=3)
        if not results:
            return None
        page = wikipedia.page(results[0], auto_suggest=False)
        sentences = page.summary.split('. ')
        return '. '.join(sentences[:n_sentences]).strip() + '.'
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.summary.split('. ')[0].strip() + '.'
        except Exception:
            return None
    except Exception as exc:
        logger.debug("Wikipedia fetch failed for %s: %s", name, exc)
        return None


def is_significant(row: pd.Series) -> bool:
    """Filter: only enrich places likely to have a Wikipedia article."""
    if row['feature_class'] == 'P' and row['population'] > 500:
        return True
    if row['feature_class'] == 'T' and pd.notna(row['elevation']) and row['elevation'] > 1000:
        return True
    if row['feature_class'] in ('H', 'A', 'L'):
        return True
    return False


def enrich_dataframe(df: pd.DataFrame, rate_limit: float = 0.5) -> pd.DataFrame:
    """
    Add a 'wiki_text' column to df for significant rows.
    rate_limit: seconds to sleep between API calls.
    """
    df = df.copy()
    df['wiki_text'] = None
    mask = df.apply(is_significant, axis=1)
    significant = df[mask]

    logger.info("Fetching Wikipedia summaries for %d places...", len(significant))

    for idx, row in tqdm(significant.iterrows(), total=len(significant),
                         desc="Wikipedia"):
        wiki = fetch_summary(row['name'])
        if wiki:
            df.at[idx, 'wiki_text'] = wiki
        time.sleep(rate_limit)

    return df


def merge_passages(df: pd.DataFrame) -> pd.DataFrame:
    """Combine GeoNames passage with Wikipedia summary into full_passage."""
    df = df.copy()
    df['full_passage'] = df.apply(
        lambda r: (r['passage'] + ' ' + r['wiki_text']).strip()
        if pd.notna(r['wiki_text']) else r['passage'],
        axis=1
    )
    return df
