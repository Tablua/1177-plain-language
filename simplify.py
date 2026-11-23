"""The raw model: hard English medical text in, simpler text out — with a fact check.

Python:    from simplify import simplify, simplify_with_flags, simplify_selection
           simplify("A whole paragraph ...")             # safe simplified text (see the hierarchy below)
           simplify_with_flags("A whole paragraph ...")  # per sentence: text, colour, reasons
           simplify_selection("a selection")             # what a button in the UI should call
Terminal:  python simplify.py "text ..."

HIERARCHY (strict, by design): the model generates several suggestions per sentence, and they
are chosen in this order
  1. Facts correct      — no new/changed number, no flipped negation, no new/changed term,
                          no garbled symbol, no swapped word, no added explanation.   (required)
  2. No info lost       — every number/abbreviation from the original must remain.    (required)
  3. Actually simplified — suggestions identical to the original do not count.
  4. As simple as possible — the lowest reading level (FKGL) wins.
  5. Closest to the original — the model's most likely suggestion wins ties.
Colour per sentence:
  green  facts correct and clearly simplified (reading level drops at least MIN_GRADE_GAIN)  -> the suggestion is shown
  yellow facts correct but little/not simplified                                             -> the suggestion is shown (the original if nothing changed)
  red    every simplification changed facts                                                  -> the ORIGINAL is shown
The text shown therefore only contains simplifications that passed the fact check.
The fact check (flagger.py) catches a lot but not everything — green means safe, not guaranteed.
"""
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from flagger import check
from glossary import lookup
from textfix import clean, mask, unmask

MAX_LOOKUP_WORDS = 4   # selections of up to this many words are looked up in the glossary
NUM_CANDIDATES = 4     # number of suggestions the model generates per sentence
MIN_GRADE_GAIN = 1.0   # how much the reading level (FKGL, roughly "school year") must drop for green

PREFIX = "Simplify this medical sentence into plain language: "
HERE = Path(__file__).parent
# Model folder: the SIMPLIFY_MODEL env var if set, otherwise models/model_base_v2, otherwise
# models/model_base (fallback so the error is clear if no model has been trained yet).
MODEL_DIR = Path(os.environ.get("SIMPLIFY_MODEL") or next(
    (HERE / "models" / d for d in ("model_base_v2", "model_base") if (HERE / "models" / d).exists()),
    HERE / "models" / "model_base"))

_cache = {}


def _load():
    if "m" not in _cache:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        _cache["m"] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(dev).eval()
        _cache["dev"] = dev
    return _cache["tok"], _cache["m"], _cache["dev"]


ABBREV = r"\b(Dr|Mr|Mrs|Ms|Prof|vs|e\.g|i\.e|et al|Fig|Figs|approx|No|Vol|St)\."


def split_sentences(text):
    """Simple sentence splitting: split on . ! ? followed by whitespace and a capital letter/digit.
    Periods inside common abbreviations (Dr., vs., e.g. ...) are protected from splitting."""
    text = clean(text)
    text = re.sub(ABBREV, lambda m: m.group(0).replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.replace("\x00", ".") for p in parts if p]


# ---------- readability ----------

def _syllables(word):
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def fkgl(sentence):
    """Flesch-Kincaid grade level for ONE sentence (roughly which school year the text needs)."""
    words = re.findall(r"[A-Za-z]+", sentence)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 0.39 * len(words) + 11.8 * syl / len(words) - 15.59


# ---------- generation ----------

def _generate(sentences, k, scored=False):
    """k suggestions per sentence (beam search), in the model's probability order.
    scored=True: each suggestion becomes (text, probability), where the probability is the model's
    average certainty per word (geometric mean of the word probabilities, 0-1)."""
    tok, model, dev = _load()
    x = tok([PREFIX + mask(clean(s)) for s in sentences], max_length=128, truncation=True,
            padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**x, num_beams=k, num_return_sequences=k, max_new_tokens=128,
                             no_repeat_ngram_size=3, early_stopping=True,
                             return_dict_in_generate=True, output_scores=True)
    dec = [unmask(t.strip()) for t in tok.batch_decode(out.sequences, skip_special_tokens=True)]
    if scored:
        probs = torch.exp(out.sequences_scores).tolist()  # length-normalised log-probability -> 0-1
        dec = list(zip(dec, probs))
    return [dec[i * k:(i + 1) * k] for i in range(len(sentences))]


def simplify_sentences(sentences):
    """Raw model: the model's best suggestion per sentence, WITHOUT the fact check (for experiments)."""
    if not sentences:
        return []
    return [c[0] for c in _generate(sentences, NUM_CANDIDATES)]


def simplify_sentence(sentence):
    return simplify_sentences([sentence])[0]


def confidence(n_passed, n_total, model_prob):
    """Confidence score 0-100 for a green/yellow sentence. Half: how many of the model's suggestions
    passed the fact check (a stable answer = safer). Half: the model's own word certainty for the
    chosen suggestion. NOTE: ranks sentences by confidence, but is NOT a calibrated probability
    that the facts are correct."""
    parts = [n_passed / max(n_total, 1)] + ([model_prob] if model_prob is not None else [])
    return round(100 * sum(parts) / len(parts))


def choose(original, candidates, probs=None):
    """Chooses among the model's suggestions according to the hierarchy. Returns a dict for one sentence.
    probs: the model's probability per suggestion (same order), used for the confidence score."""
    g0 = fkgl(original)
    passing, failed = [], []
    for rank, c in enumerate(candidates):
        flags = check(original, c)
        codes = {f["code"] for f in flags}
        if codes == {"not_simplified"}:
            continue                                  # step 3: identical, does not count
        if flags:
            failed.append((c, flags))                 # steps 1-2: facts/info deviate -> discarded
        else:
            passing.append((fkgl(c), rank, c))        # steps 4-5: simplest, then most likely

    if passing:
        g, rank, best = min(passing)
        gain = g0 - g
        prob = probs[rank] if probs else None
        conf = confidence(len(passing), len(candidates), prob)
        conf_txt = (f"Confidence {conf}%: {len(passing)} of {len(candidates)} model suggestions passed the "
                    "fact check" + (f", model word certainty {round(100 * prob)}%" if prob is not None else "")
                    + ". (A ranking, not a guaranteed probability.)")
        extra = {"model_output": best, "confidence": conf,
                 "confidence_parts": {"candidates_passed": len(passing), "candidates": len(candidates),
                                      "model_certainty": round(100 * prob) if prob is not None else None}}
        if gain >= MIN_GRADE_GAIN:
            return {"simplified": best, "status": "green", **extra,
                    "reasons": [f"Simplified (reading level {g0:.0f} -> {g:.0f}). Facts checked: "
                                "numbers, negations, terms and symbols match the original.", conf_txt]}
        # passed the fact check but wasn't clearly simpler: still shown (by design) — only red gets replaced
        return {"simplified": best, "status": "yellow", **extra,
                "reasons": [f"Facts match the original, but the text is only slightly simpler "
                            f"(reading level {g0:.0f} -> {g:.0f}).", conf_txt]}
    if failed:
        rejected, flags = failed[0]
        return {"simplified": original, "status": "red", "model_output": rejected, "confidence": None,
                "reasons": ["The model's simplification changed facts, so the original is shown. "
                            f"Rejected: \"{rejected}\""] + [f["detail"] for f in flags]}
    return {"simplified": original, "status": "yellow", "model_output": candidates[0] if candidates else "",
            "confidence": None,
            "reasons": ["The model could not simplify this sentence, so the original is shown."]}


def simplify_with_flags(text, batch_size=16):
    """Per sentence: what is shown, the colour and the reasons.

    Returns [{"source_id": "s1", "original", "simplified" (what gets shown),
              "status": "green"|"yellow"|"red", "reasons": [...], "model_output",
              "confidence": 0-100 for green/yellow, None for red}, ...]
    """
    sents = split_sentences(text)
    res = []
    for i in range(0, len(sents), batch_size):
        batch = sents[i:i + batch_size]
        for s, scored in zip(batch, _generate(batch, NUM_CANDIDATES, scored=True)):
            cands, probs = [c for c, _ in scored], [p for _, p in scored]
            res.append({"source_id": f"s{len(res) + 1}", "original": s, **choose(s, cands, probs)})
    return res


def simplify(text, batch_size=16):
    """Simplifies a whole paragraph. Returns the safe text (the original where the facts didn't hold)."""
    return " ".join(s["simplified"] for s in simplify_with_flags(text, batch_size))


def simplify_selection(text):
    """What a button in the UI should call: takes a selection (word, phrase, sentence or
    paragraph) and picks the right method.

    - Words/short phrases (max 4 words, no sentence-ending punctuation): looked up in the glossary.
      If there is no explanation, we say so — the model never guesses on single words.
    - Everything else: the model simplifies sentence by sentence according to the hierarchy.

    Returns {"mode": "glossary"|"model"|"empty", "found": bool, "text": str, "term": str|None,
             "sentences": [...]}   # sentences = simplify_with_flags(...) in model mode, else []
    """
    text = clean(text)
    if not text:
        return {"mode": "empty", "found": False, "text": "", "term": None, "sentences": []}
    if len(text.split()) <= MAX_LOOKUP_WORDS:
        hit = lookup(text)  # ignores punctuation, so "HbA1c." matches too
        if hit:
            return {"mode": "glossary", "found": True, "text": hit["plain"], "term": hit["term"],
                    "sentences": []}
        if not re.search(r"[.!?]$", text):  # no hit and not a sentence -> say so, don't guess
            return {"mode": "glossary", "found": False,
                    "text": "No plain-language explanation found for this word.", "term": None,
                    "sentences": []}
    sents = simplify_with_flags(text)
    return {"mode": "model", "found": True, "term": None, "sentences": sents,
            "text": " ".join(s["simplified"] for s in sents)}


if __name__ == "__main__":
    inp = " ".join(sys.argv[1:]) or sys.stdin.read()
    print(simplify_selection(inp)["text"])
