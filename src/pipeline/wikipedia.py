"""
Enrich GeoNames passages with Wikipedia summaries.
Fetches 3-sentence summaries for significant places.
Uses parallel requests for faster scraping.
"""

import logging
import warnings
import wikipedia
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress BeautifulSoup parser warning from the wikipedia library
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

logger = logging.getLogger(__name__)

# Number of parallel threads — increase for faster scraping,
# decrease if Wikipedia starts rate-limiting you (HTTP 429 errors)
MAX_WORKERS = 8


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
    if row['feature_class'] == 'P' and row['population'] > 2000:  # war 500
        return True
    if row['feature_class'] == 'T' and pd.notna(row['elevation']) and row['elevation'] > 2000:  # war 1000
        return True
    if row['feature_class'] in ('H', 'A'):  # 'L' entfernt
        return True
    return False


def enrich_dataframe(df: pd.DataFrame, rate_limit: float = 0.5) -> pd.DataFrame:
    """
    Add a 'wiki_text' column to df for significant rows.
    Uses ThreadPoolExecutor for parallel requests.
    rate_limit parameter kept for API compatibility but no longer used.
    """
    df = df.copy()
    df['wiki_text'] = None
    mask = df.apply(is_significant, axis=1)
    significant = df[mask]

    logger.info("Fetching Wikipedia summaries for %d places (parallel, %d workers)...",
                len(significant), MAX_WORKERS)

    # Map from index → result
    results: dict[int, str | None] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(fetch_summary, row['name']): idx
            for idx, row in significant.iterrows()
        }

        # Collect results with progress bar
        with tqdm(total=len(significant), desc="Wikipedia") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.debug("Unexpected error for index %s: %s", idx, exc)
                    results[idx] = None
                pbar.update(1)

    # Write results back to dataframe
    for idx, wiki_text in results.items():
        if wiki_text:
            df.at[idx, 'wiki_text'] = wiki_text

    found = sum(1 for v in results.values() if v)
    logger.info("Wikipedia enrichment done: %d/%d articles found.", found, len(significant))
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
