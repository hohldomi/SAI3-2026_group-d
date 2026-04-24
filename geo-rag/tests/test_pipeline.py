"""
Basic tests for the pipeline modules.
Run with: pytest tests/
"""

import pytest
import pandas as pd
from src.pipeline.geonames import row_to_passage, FEATURE_LABELS
from src.retrieval.retrieve import detect_feature_class
from src.generation.prompt import build_messages


# --- geonames.py ---

def make_row(**kwargs):
    defaults = {
        'geonameid': 1, 'name': 'Testdorf', 'asciiname': 'Testdorf',
        'alternatenames': '', 'latitude': 46.9, 'longitude': 7.4,
        'feature_class': 'P', 'feature_code': 'PPL', 'country_code': 'CH',
        'cc2': '', 'admin1_code': 'BE', 'admin2_code': '', 'admin3_code': '',
        'admin4_code': '', 'population': 1000, 'elevation': 560.0,
        'dem': 560, 'timezone': 'Europe/Zurich', 'modification_date': '2024-01-01'
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_row_to_passage_basic():
    row = make_row(name='Bern', population=133000, elevation=542.0)
    passage = row_to_passage(row)
    assert 'Bern' in passage
    assert 'populated place' in passage
    assert '133,000' in passage
    assert '542' in passage


def test_row_to_passage_no_population():
    row = make_row(name='Kleindorf', population=0)
    passage = row_to_passage(row)
    assert 'population' not in passage


def test_row_to_passage_mountain():
    row = make_row(name='Matterhorn', feature_class='T', population=0, elevation=4478.0)
    passage = row_to_passage(row)
    assert 'mountain or peak' in passage
    assert '4,478' in passage or '4478' in passage


# --- retrieve.py ---

def test_detect_feature_class_mountain():
    assert detect_feature_class("What mountains are near Zermatt?") == 'T'


def test_detect_feature_class_city():
    assert detect_feature_class("What is the population of Bern?") == 'P'


def test_detect_feature_class_lake():
    assert detect_feature_class("Tell me about Lake Geneva") == 'H'


def test_detect_feature_class_none():
    assert detect_feature_class("Tell me about Zermatt") is None


# --- prompt.py ---

def test_build_messages_structure():
    docs = [{'passage': 'Bern is the capital.', 'name': 'Bern', 'score': 0.9}]
    messages = build_messages("What is Bern?", docs)
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert 'Bern is the capital.' in messages[1]['content']
    assert 'What is Bern?' in messages[1]['content']


def test_build_messages_multiple_docs():
    docs = [
        {'passage': 'Bern is the capital.', 'name': 'Bern', 'score': 0.9},
        {'passage': 'Bern has 133,000 inhabitants.', 'name': 'Bern', 'score': 0.8},
    ]
    messages = build_messages("Tell me about Bern", docs)
    assert '[1]' in messages[1]['content']
    assert '[2]' in messages[1]['content']
