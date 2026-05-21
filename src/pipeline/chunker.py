"""
Recursive Chunking for GeoRAG passages.

Splits long passages into overlapping chunks using a hierarchy of separators:
  paragraph → sentence → word

Short passages (below CHUNK_SIZE) are kept as-is.
Each chunk inherits the metadata of its source passage.
"""

from __future__ import annotations

CHUNK_SIZE = 400       # target characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks
MIN_CHUNK_SIZE = 80    # discard chunks shorter than this

# Separator hierarchy: try splitting on these in order
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Recursively split text using the first separator that works."""
    if not separators:
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), max(1, CHUNK_SIZE - CHUNK_OVERLAP))]

    sep = separators[0]
    parts = text.split(sep)

    if len(parts) == 1:
        return _split_text(text, separators[1:])

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current and len(current) >= MIN_CHUNK_SIZE:
                chunks.append(current)

            if len(part) > CHUNK_SIZE:
                sub_chunks = _split_text(part, separators[1:])
                chunks.extend(sub_chunks)
                current = sub_chunks[-1][-CHUNK_OVERLAP:] if sub_chunks else ""
            else:
                current = part.strip()

    if current and len(current) >= MIN_CHUNK_SIZE:
        chunks.append(current)

    return chunks if chunks else [text]


def chunk_passage(record: dict) -> list[dict]:
    """
    Split a record dict into one or more chunk dicts.
    Uses the 'passage' key (matching build_corpus.py field name).
    Each chunk gets a unique 'id' and inherits all metadata.
    """
    text = record.get("passage", "").strip()
    source_id = str(record.get("geonameid", "unknown"))

    if len(text) <= CHUNK_SIZE:
        return [{**record, "id": f"{source_id}_0", "chunk_index": 0, "total_chunks": 1}]

    raw_chunks = _split_text(text, SEPARATORS)

    # Add overlap between consecutive chunks
    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            overlap_text = raw_chunks[i - 1][-CHUNK_OVERLAP:]
            overlapped.append((overlap_text + " " + chunk).strip())

    total = len(overlapped)
    return [
        {
            **record,
            "passage": chunk_text,
            "id": f"{source_id}_{i}",
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk_text in enumerate(overlapped)
    ]


def chunk_corpus(records: list[dict]) -> list[dict]:
    """Apply chunk_passage to every record and return the flat list of chunks."""
    chunks: list[dict] = []
    for r in records:
        chunks.extend(chunk_passage(r))
    return chunks