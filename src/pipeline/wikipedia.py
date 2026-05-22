"""
Enrich GeoNames passages with Wikipedia summaries.
Uses the Wikipedia REST API directly (no wikipedia library) for reliability.
Uses parallel requests for faster scraping.
"""

import logging
import time
import warnings
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

MAX_WORKERS = 8
SLEEP_TIME = 0.8

HEADERS = {
    "User-Agent": "SAI3-GeoRAG/1.0 (BFH student project; contact: dominik@students.bfh.ch)",
    "Accept": "application/json",
}

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Extended Swiss keywords — catches articles that mention cantons/cities
# but not explicitly "Switzerland"
SWISS_KEYWORDS = {
    "switzerland", "swiss", "schweiz", "suisse", "svizzera", "helvetia",
    "canton", "graubünden", "grisons", "valais", "wallis", "ticino",
    "bern", "zurich", "zürich", "geneva", "genève", "basel", "lucerne",
    "luzern", "appenzell", "glarus", "thurgau", "aargau", "solothurn",
    "fribourg", "freiburg", "neuchâtel", "schaffhausen", "schwyz",
    "obwalden", "nidwalden", "uri", "zug", "jura", "vaud",
}


def _is_swiss_article(summary: str) -> bool:
    """Return True if the article summary mentions Switzerland in any way."""
    lower = summary.lower()
    return any(kw in lower for kw in SWISS_KEYWORDS)


def _search_candidates(name: str, country: str = "Switzerland") -> list[str]:
    """Search Wikipedia for candidate article titles via the MediaWiki API."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{name} {country}",
        "srlimit": 5,
        "format": "json",
    }
    try:
        r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 429:
            logger.info("Rate limited on search for '%s', sleeping 60s...", name)
            time.sleep(60)
            r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [item["title"] for item in data.get("query", {}).get("search", [])]
    except Exception as exc:
        logger.info("Search failed for '%s': %s", name, exc)
        return []


def _fetch_page_summary(title: str) -> str | None:
    """Fetch the summary of a Wikipedia page via the REST API."""
    url = SUMMARY_URL.format(requests.utils.quote(title, safe=""))
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 429:
            logger.info("Rate limited on page '%s', sleeping 60s...", title)
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        return data.get("extract", None)
    except Exception as exc:
        logger.info("Page fetch failed for '%s': %s", title, exc)
        return None


def fetch_summary(name: str, country: str = "Switzerland",
                  n_sentences: int = 8) -> str | None:
    """
    Fetch first n_sentences from the most relevant Swiss Wikipedia article.
    Uses requests directly instead of the wikipedia library.
    """
    time.sleep(SLEEP_TIME)

    candidates = _search_candidates(name, country)
    if not candidates:
        return None

    for title in candidates:
        extract = _fetch_page_summary(title)
        if not extract:
            continue
        if not _is_swiss_article(extract):
            logger.info("Skipping '%s' for '%s' — not a Swiss article", title, name)
            continue
        sentences = extract.split(". ")
        return ". ".join(sentences[:n_sentences]).strip() + "."

    logger.info("No valid Swiss article found for: %s", name)
    return None


def is_significant(row: pd.Series) -> bool:
    fc = row["feature_class"]
    if fc == "P" and row["population"] > 500:
        return True
    if fc == "T" and pd.notna(row["elevation"]) and row["elevation"] > 1500:
        return True
    if fc in ("H", "A", "L", "S"):
        return True
    return False


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Deduplicate significant rows by name before fetching.
    Returns unique rows and a mapping name -> wiki_text to fill back in.
    """
    mask = df.apply(is_significant, axis=1)
    significant = df[mask].copy()
    unique_names = significant.drop_duplicates(subset=["name"])
    logger.info(
        "Deduplication: %d significant rows → %d unique names",
        len(significant), len(unique_names),
    )
    return unique_names, mask


def enrich_dataframe(df: pd.DataFrame, rate_limit: float = 0.5) -> pd.DataFrame:
    """
    Add a 'wiki_text' column to df for significant rows.
    Deduplicates by name before fetching to avoid redundant API calls.
    Uses ThreadPoolExecutor for parallel requests.
    """
    df = df.copy()
    df["wiki_text"] = None

    unique_rows, mask = _deduplicate(df)

    logger.info(
        "Fetching Wikipedia summaries for %d unique places (parallel, %d workers)...",
        len(unique_rows), MAX_WORKERS,
    )

    # Fetch only unique names
    name_to_wiki: dict[str, str | None] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(fetch_summary, row["name"]): row["name"]
            for _, row in unique_rows.iterrows()
        }

        with tqdm(total=len(unique_rows), desc="Wikipedia") as pbar:
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    name_to_wiki[name] = future.result()
                except Exception as exc:
                    logger.info("Unexpected error for '%s': %s", name, exc)
                    name_to_wiki[name] = None
                pbar.update(1)

    # Write results back — all rows with the same name get the same wiki_text
    df.loc[mask, "wiki_text"] = df.loc[mask, "name"].map(name_to_wiki)

    found = sum(1 for v in name_to_wiki.values() if v)
    logger.info(
        "Wikipedia enrichment done: %d/%d unique articles found (%.1f%%).",
        found,
        len(unique_rows),
        100 * found / len(unique_rows) if len(unique_rows) > 0 else 0,
    )
    return df


def merge_passages(df: pd.DataFrame) -> pd.DataFrame:
    """Combine GeoNames passage with Wikipedia summary into full_passage."""
    df = df.copy()
    df["full_passage"] = df.apply(
        lambda r: (r["passage"] + " " + r["wiki_text"]).strip()
        if pd.notna(r["wiki_text"]) else r["passage"],
        axis=1,
    )
    return df