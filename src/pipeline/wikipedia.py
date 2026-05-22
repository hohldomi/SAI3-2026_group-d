"""
Enrich GeoNames passages with Wikipedia summaries.
Uses the Wikipedia REST API directly (no wikipedia library) for reliability.
Uses a global Token Bucket rate limiter to avoid Wikipedia 429s.
Caches results to data/raw/wiki_cache.json to skip re-fetching on subsequent builds.
"""

import json
import logging
import time
import threading
import warnings
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

MAX_WORKERS  = 4
GLOBAL_RPS   = 2.0   # max requests per second globally — stays safely under Wikipedia's limit
CACHE_PATH   = Path("data/raw/wiki_cache.json")

HEADERS = {
    "User-Agent": "SAI3-GeoRAG/1.0 (BFH student project; contact: dominik@students.bfh.ch)",
    "Accept":     "application/json",
}

SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

SWISS_KEYWORDS = {
    # Landesbezeichnungen
    "switzerland", "swiss", "schweiz", "suisse", "svizzera", "helvetia",
    # Kantone EN/DE/FR/IT
    "canton", "kanton",
    "graubünden", "grisons", "grigioni",
    "valais", "wallis",
    "ticino", "tessin",
    "bern", "berne",
    "zurich", "zürich",
    "geneva", "genève", "genf",
    "basel",
    "lucerne", "luzern",
    "appenzell",
    "glarus", "glaris",
    "thurgau", "thurgovie",
    "aargau", "argovie",
    "solothurn", "soleure",
    "fribourg", "freiburg",
    "neuchâtel", "neuenburg",
    "schaffhausen", "schaffhouse",
    "schwyz",
    "obwalden", "nidwalden",
    "uri",
    "zug",
    "jura",
    "vaud", "waadt",
    "st. gallen", "st gallen", "saint-gall",
    # Geographische Begriffe CH-spezifisch
    "municipality in", "commune in", "gemeinde",
    "swiss alps", "bernese alps", "pennine alps", "lepontine alps",
    "rhaetian alps", "glarus alps", "urner alps",
    "alpine", "alpine pass",
    "rhine", "rhône", "aare", "limmat", "reuss", "inn", "ticino river",
    "lake geneva", "lake zurich", "lake constance", "lake lucerne",
    "lake maggiore", "lake lugano",
    "bodensee", "vierwaldstättersee", "zürichsee", "genfersee",
    # Regionen
    "mittelland", "emmental", "bernese oberland", "engadin", "engadine",
    "prättigau", "surselva", "leventina", "maggia",
}


# ---------------------------------------------------------------------------
# Global Token Bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """Thread-safe token bucket — limits total requests/sec across all workers."""
    def __init__(self, rate: float):
        self._rate   = rate
        self._tokens = rate
        self._last   = time.monotonic()
        self._lock   = threading.Lock()

    def acquire(self):
        with self._lock:
            now   = time.monotonic()
            delta = now - self._last
            self._last   = now
            self._tokens = min(self._rate, self._tokens + delta * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
        time.sleep(wait)


_bucket = TokenBucket(rate=GLOBAL_RPS)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, str | None]:
    """Load the wiki cache from disk. Returns empty dict if not found."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded wiki cache: %d entries from %s", len(cache), CACHE_PATH)
            return cache
        except Exception as exc:
            logger.warning("Could not load wiki cache (%s), starting fresh.", exc)
    return {}


def _save_cache(cache: dict[str, str | None]) -> None:
    """Persist the wiki cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info("Saved wiki cache: %d entries to %s", len(cache), CACHE_PATH)
    except Exception as exc:
        logger.warning("Could not save wiki cache: %s", exc)


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """Rate-limited GET with automatic retry on 429."""
    for attempt in range(retries):
        _bucket.acquire()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                logger.info("429 received — sleeping %ds (attempt %d/%d)...",
                            wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.JSONDecodeError:
            logger.info("JSON decode error on attempt %d", attempt + 1)
            time.sleep(5)
        except Exception as exc:
            logger.info("Request error on attempt %d: %s", attempt + 1, exc)
            time.sleep(5)
    return None


def _search_candidates(name: str, country: str = "Switzerland") -> list[str]:
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": f"{name} {country}",
        "srlimit":  5,
        "format":   "json",
    }
    data = _get(SEARCH_URL, params=params)
    if not data:
        return []
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def _fetch_page_summary(title: str) -> str | None:
    url  = SUMMARY_URL.format(requests.utils.quote(title, safe=""))
    data = _get(url)
    if not data:
        return None
    return data.get("extract") or None


def _is_swiss_article(summary: str) -> bool:
    lower = summary.lower()
    return any(kw in lower for kw in SWISS_KEYWORDS)


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------

def fetch_summary(name: str, country: str = "Switzerland",
                  n_sentences: int = 12) -> str | None:
    """Fetch first n_sentences from the most relevant Swiss Wikipedia article."""
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


# ---------------------------------------------------------------------------
# Significance filter
# ---------------------------------------------------------------------------

def is_significant(row: pd.Series) -> bool:
    fc = row["feature_class"]
    if fc == "P" and row["population"] > 500:
        return True
    if fc == "T" and pd.notna(row["elevation"]) and row["elevation"] > 1500:
        return True
    if fc in ("H", "A", "L"):
        return True
    return False


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame, rate_limit: float = 0.5) -> pd.DataFrame:
    """
    Add a 'wiki_text' column to df for significant rows.
    - Loads existing cache from data/raw/wiki_cache.json
    - Only fetches names not already in cache
    - Saves updated cache after fetching
    - Uses ThreadPoolExecutor with global token bucket rate limiter
    """
    df   = df.copy()
    df["wiki_text"] = None
    mask = df.apply(is_significant, axis=1)
    significant  = df[mask]
    unique_rows  = significant.drop_duplicates(subset=["name"])

    logger.info(
        "Deduplication: %d significant rows → %d unique names",
        len(significant), len(unique_rows),
    )

    # Load cache
    cache = _load_cache()

    # Split into cached and to-fetch
    all_names     = unique_rows["name"].tolist()
    cached_names  = {n for n in all_names if n in cache}
    missing_names = [n for n in all_names if n not in cache]

    logger.info(
        "Cache: %d/%d names already cached, fetching %d new names...",
        len(cached_names), len(all_names), len(missing_names),
    )

    # Fetch only missing names
    if missing_names:
        logger.info(
            "Fetching Wikipedia summaries for %d unique places "
            "(parallel, %d workers, %.1f req/s global limit)...",
            len(missing_names), MAX_WORKERS, GLOBAL_RPS,
        )
        missing_rows = unique_rows[unique_rows["name"].isin(missing_names)]
        new_results: dict[str, str | None] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_name = {
                executor.submit(fetch_summary, row["name"]): row["name"]
                for _, row in missing_rows.iterrows()
            }

            with tqdm(total=len(missing_rows), desc="Wikipedia") as pbar:
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        new_results[name] = future.result()
                    except Exception as exc:
                        logger.info("Unexpected error for '%s': %s", name, exc)
                        new_results[name] = None
                    pbar.update(1)

        # Merge new results into cache and save
        cache.update(new_results)
        _save_cache(cache)

        found_new = sum(1 for v in new_results.values() if v)
        logger.info(
            "Fetched %d new articles (%d/%d found, %.1f%%).",
            len(missing_names), found_new, len(missing_names),
            100 * found_new / len(missing_names) if missing_names else 0,
        )
    else:
        logger.info("All names served from cache — no Wikipedia requests needed.")

    # Write results back from cache — all rows with same name share the result
    df.loc[mask, "wiki_text"] = df.loc[mask, "name"].map(cache)

    found = df.loc[mask, "wiki_text"].notna().sum()
    logger.info(
        "Wikipedia enrichment done: %d/%d significant rows enriched (%.1f%%).",
        found, len(significant),
        100 * found / len(significant) if len(significant) > 0 else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Passage merger
# ---------------------------------------------------------------------------

def merge_passages(df: pd.DataFrame) -> pd.DataFrame:
    """Combine GeoNames passage with Wikipedia summary into full_passage."""
    df = df.copy()
    df["full_passage"] = df.apply(
        lambda r: (r["passage"] + " " + r["wiki_text"]).strip()
        if pd.notna(r["wiki_text"]) else r["passage"],
        axis=1,
    )
    return df