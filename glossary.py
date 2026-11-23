"""Lookup of single medical words / short phrases in a hand-written glossary.

No model, no torch: fast, predictable and unable to make anything up.
    from glossary import lookup
    lookup("Hypertension")  ->  {"term": "hypertension", "plain": "high blood pressure"}
    lookup("unknown word")  ->  None
"""
import json
import re
from pathlib import Path

PATH = Path(__file__).parent / "glossary.json"

_index = {}


def _norm(s):
    s = re.sub(r"[^\w\s\-]", " ", s.lower())  # drop punctuation
    return " ".join(s.split())


def _load():
    if _index:
        return _index
    for e in json.load(open(PATH, encoding="utf-8"))["entries"]:
        for name in [e["term"], *e.get("aliases", [])]:
            _index[_norm(name)] = e
    return _index


def _variants(key):
    """Simple inflections: 'infarctions' -> 'infarction', 'lesions' -> 'lesion'."""
    yield key
    if key.endswith("ies"):
        yield key[:-3] + "y"
    if key.endswith("es"):
        yield key[:-2]
    if key.endswith("s"):
        yield key[:-1]


def lookup(text):
    """Returns {'term', 'plain'} or None if the word/phrase is not in the glossary."""
    idx = _load()
    for k in _variants(_norm(text)):
        if k in idx:
            e = idx[k]
            return {"term": e["term"], "plain": e["plain"]}
    return None
