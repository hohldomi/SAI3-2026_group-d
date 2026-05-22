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

# GeoNames admin1_code → Swiss canton names (both English and German)
CANTON_NAMES = {
    'AG': 'canton of Aargau (Aargau)',
    'AI': 'canton of Appenzell Innerrhoden',
    'AR': 'canton of Appenzell Ausserrhoden',
    'BE': 'canton of Bern (Berne)',
    'BL': 'canton of Basel-Landschaft',
    'BS': 'canton of Basel-Stadt',
    'FR': 'canton of Fribourg (Freiburg)',
    'GE': 'canton of Geneva (Genève)',
    'GL': 'canton of Glarus',
    'GR': 'canton of Graubünden (Grisons)',
    'JU': 'canton of Jura',
    'LU': 'canton of Lucerne (Luzern)',
    'NE': 'canton of Neuchâtel',
    'NW': 'canton of Nidwalden',
    'OW': 'canton of Obwalden',
    'SG': 'canton of St. Gallen',
    'SH': 'canton of Schaffhausen',
    'SO': 'canton of Solothurn',
    'SZ': 'canton of Schwyz',
    'TG': 'canton of Thurgau',
    'TI': 'canton of Ticino',
    'UR': 'canton of Uri',
    'VD': 'canton of Vaud',
    'VS': 'canton of Valais (Wallis)',
    'ZG': 'canton of Zug',
    'ZH': 'canton of Zurich (Zürich)',
}

# GeoNames feature_code → human-readable descriptions for common codes
FEATURE_CODE_LABELS = {
    # Populated places
    'PPL':   'municipality',
    'PPLA':  'cantonal capital',
    'PPLA2': 'district capital',
    'PPLC':  'federal capital',
    'PPLX':  'section of a populated place',
    # Mountains / terrain
    'MT':    'mountain',
    'PK':    'peak',
    'PASS':  'mountain pass',
    'GLCR':  'glacier',
    'CLF':   'cliff',
    'GRGE':  'gorge',
    'VAL':   'valley',
    'VLC':   'volcano',
    # Water
    'LK':    'lake',
    'STM':   'stream or river',
    'STMR':  'river',
    'STMI':  'intermittent stream',
    'WTRH':  'waterfall',
    'SPNG':  'spring',
    'RSV':   'reservoir',
    'CNL':   'canal',
    # Administrative
    'ADM1':  'canton',
    'ADM2':  'district',
    'ADM3':  'municipality',
    # Sites
    'CSTL':  'castle',
    'MNST':  'monastery',
    'CH':    'church',
    'MSQE':  'mosque',
    'TMPL':  'temple',
    'RUIN':  'ruins',
    'CAVE':  'cave',
    'HSTS':  'historical site',
    'MUS':   'museum',
    'UNIV':  'university',
    'AIRP':  'airport',
    'RSRT':  'resort',
    'SPA':   'spa',
}


def load_geonames(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', names=GEONAMES_COLS,
                     low_memory=False, na_values='')
    df = df[df['feature_class'].isin(RELEVANT_CLASSES)].copy()
    df['population'] = df['population'].fillna(0).astype(int)
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    return df


def row_to_passage(row: pd.Series) -> str:
    ftype = FEATURE_LABELS.get(row['feature_class'], 'place')

    # Use specific feature code label if available for a richer description
    fcode_label = FEATURE_CODE_LABELS.get(str(row.get('feature_code', '')), '')

    # Canton context
    canton = CANTON_NAMES.get(str(row.get('admin1_code', '')), '')

    # Opening sentence — include canton if known
    if canton:
        parts = [f"{row['name']} is a {ftype} in Switzerland, located in the {canton}."]
    else:
        parts = [f"{row['name']} is a {ftype} in Switzerland."]

    # Feature code clarification (e.g. "It is specifically a mountain pass.")
    if fcode_label and fcode_label != ftype:
        parts.append(f"It is specifically a {fcode_label}.")

    # Population
    if row['population'] > 0:
        parts.append(f"It has a population of {row['population']:,}.")

    # Coordinates
    if pd.notna(row['latitude']):
        parts.append(
            f"It is located at {row['latitude']:.4f}°N, {row['longitude']:.4f}°E."
        )

    # Elevation
    if pd.notna(row['elevation']) and row['elevation'] > 0:
        parts.append(f"Its elevation is {int(row['elevation'])} metres above sea level.")

    # Alternate names — increased from 3 to 5, and formatted more naturally
    if pd.notna(row['alternatenames']) and row['alternatenames']:
        alts = [a.strip() for a in str(row['alternatenames']).split(',') if a.strip()]
        # Filter out pure numeric strings (e.g. postcodes sometimes appear here)
        alts = [a for a in alts if not a.isdigit()][:5]
        if alts:
            parts.append(f"It is also known as {', '.join(alts)}.")

    # Timezone — formatted more naturally
    if pd.notna(row['timezone']):
        tz = str(row['timezone'])
        parts.append(f"It lies in the {tz} timezone.")

    return ' '.join(parts)


def build_passages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['passage'] = df.apply(row_to_passage, axis=1)
    return df