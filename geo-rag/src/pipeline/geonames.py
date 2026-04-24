"""
Convert raw GeoNames CH.txt rows into natural-language text passages.
"""

import pandas as pd

GEONAMES_COLS = [
    'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
    'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
    'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
    'dem', 'timezone', 'modification_date'
]

FEATURE_LABELS = {
    'P': 'populated place',
    'T': 'mountain or peak',
    'H': 'water body',
    'L': 'region or area',
    'A': 'administrative division',
    'S': 'site or building',
}

# Only index these feature classes (skip misc/noise)
RELEVANT_CLASSES = {'P', 'T', 'H', 'A', 'L'}


def load_geonames(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', names=GEONAMES_COLS,
                     low_memory=False, na_values='')
    df = df[df['feature_class'].isin(RELEVANT_CLASSES)].copy()
    df['population'] = df['population'].fillna(0).astype(int)
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    return df


def row_to_passage(row: pd.Series) -> str:
    ftype = FEATURE_LABELS.get(row['feature_class'], 'place')
    parts = [f"{row['name']} is a {ftype} in Switzerland."]

    if row['population'] > 0:
        parts.append(f"It has a population of {row['population']:,}.")

    if pd.notna(row['latitude']):
        parts.append(
            f"It is located at {row['latitude']:.4f}°N, {row['longitude']:.4f}°E."
        )

    if pd.notna(row['elevation']) and row['elevation'] > 0:
        parts.append(f"Its elevation is {int(row['elevation'])} metres above sea level.")

    if pd.notna(row['alternatenames']) and row['alternatenames']:
        alts = [a.strip() for a in str(row['alternatenames']).split(',') if a.strip()][:3]
        if alts:
            parts.append(f"It is also known as {', '.join(alts)}.")

    if pd.notna(row['timezone']):
        parts.append(f"Time zone: {row['timezone']}.")

    return ' '.join(parts)


def build_passages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['passage'] = df.apply(row_to_passage, axis=1)
    return df
