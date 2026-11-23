"""How good is the fact check (flagger.check)? Plants known errors into the human-written
simplifications and measures how large a share is caught. No model needed.

Run:  python eval_fact_check.py            (the validation split, not test)

Method:
  1. Take PLABA val pairs (hard sentence, human simplification) where the check finds NO problem.
  2. For each error type: plant the error in the human simplification (if possible) and run the check again.
  3. Share caught = how often the check now flags.
Limitation: only measures the error types below. Errors nobody thought of are not measured.
"""
import random
import sys
import re
import time
from pathlib import Path

from flagger import QUALIFIERS, check, _acronyms, _drugs
from textfix import unmask
from train import load_pairs

random.seed(0)

DATA = Path(__file__).parent / "data" / "data.json"


def _sub_first(pattern, repl, text, flags=0):
    new = re.sub(pattern, repl, text, count=1, flags=flags)
    return new if new != text else None


def m_number(s):            # 12 -> 120 (changed value)
    m = re.search(r"\b\d+(?:\.\d+)?\b", s)
    return s[:m.start()] + str(int(float(m.group()) * 10) if "." not in m.group() else m.group() + "5") + s[m.end():] if m else None


def m_unit(s):              # 5 mg -> 5 g
    return _sub_first(r"(\d)\s?mg\b", r"\1 g", s)


def m_negation_add(s):      # "was" -> "was not"
    return _sub_first(r"\b(is|are|was|were|can|will|does|did)\b", r"\1 not", s)


def m_negation_remove(s):   # "did not reduce" -> "did reduce"
    return _sub_first(r"\b(not|no|never|without)\s", "", s, re.I)


def m_hedge_remove(s):      # "may reduce" -> "reduce"
    return _sub_first(r"\b(may|might|could|possibly|probably|likely)\s", "", s, re.I)


def m_qualifier_drop(s):    # "systolic blood pressure" -> "blood pressure"
    for w in QUALIFIERS:
        new = _sub_first(rf"\b{w}\w*\s", "", s, re.I)
        if new:
            return new
    return None


def m_qualifier_add(s):     # "pain" -> "severe pain"
    return _sub_first(r"\b(pain|disease|symptoms|infection|patients|people|children|adults)\b", r"severe \1", s, re.I)


def m_drug_swap(s):         # metformin -> warfarin
    d = sorted(_drugs(s))
    return re.sub(rf"\b{re.escape(d[0])}\b", "warfarin" if d[0] != "warfarin" else "aspirin", s, flags=re.I) if d else None


def m_acronym_swap(s):      # CHF -> COVID-19
    a = sorted(_acronyms(s))
    a = [x for x in a if x.isalpha()]
    return s.replace(a[0], "COVID-19" if a[0] != "COVID" else "CHF", 1) if a else None


def m_laterality(s):        # left <-> right
    if re.search(r"\bleft\b", s, re.I):
        return _sub_first(r"\bleft\b", "right", s, re.I)
    return _sub_first(r"\bright\b", "left", s, re.I)


def m_history_drop(s):      # "history of stroke" -> "stroke"
    return _sub_first(r"\b(a history of|history of|previous|prior)\s", "", s, re.I)


def m_explanation(s):       # added explanation in parentheses
    words = re.findall(r"\b[a-z]{7,}\b", s)
    if not words:
        return None
    w = words[0]
    return s.replace(w, f"{w} (a type of protein that protects cells from infection)", 1)


def m_clause(s):            # made-up clause
    return re.sub(r"\.?\s*$", ", which can also cause heart failure and memory loss in older adults.", s, count=1)


MUTATIONS = {
    "Changed number (12 -> 120)": m_number,
    "Changed unit (mg -> g)": m_unit,
    "Added negation": m_negation_add,
    "Removed negation": m_negation_remove,
    "Removed uncertainty word (may)": m_hedge_remove,
    "Removed important word (systolic, chronic ...)": m_qualifier_drop,
    "Added important word (severe)": m_qualifier_add,
    "Swapped medicine": m_drug_swap,
    "Swapped abbreviation (CHF -> COVID-19)": m_acronym_swap,
    "Left <-> right": m_laterality,
    "Removed time marker (history of)": m_history_drop,
    "Made-up explanation in parentheses": m_explanation,
    "Made-up clause": m_clause,
}


def main():
    t0 = time.time()
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    pairs = [(unmask(s), unmask(t)) for s, t in load_pairs(DATA)[split]]
    print(f"Split: {split}", flush=True)
    flagged_human = sum(bool(check(s, t)) for s, t in pairs)
    clean = [(s, t) for s, t in pairs if not check(s, t)]
    print(f"Pairs: {len(pairs)}. The human simplification is flagged in {flagged_human / len(pairs):.0%} "
          f"(usually reasonable rewrites, see below). Clean pairs to plant errors in: {len(clean)}.\n", flush=True)

    print(f"{'Planted error':45s} {'count':>6s} {'caught':>8s}", flush=True)
    tot_n = tot_hit = 0
    for i, (name, fn) in enumerate(MUTATIONS.items(), 1):
        n = hit = 0
        for s, t in clean:
            m = fn(t)
            if not m or m == t:
                continue
            n += 1
            hit += bool([f for f in check(s, m) if f["code"] != "not_simplified"])
        tot_n += n
        tot_hit += hit
        print(f"{name:45s} {n:6d} {hit / n if n else 0:8.0%}   [{i}/{len(MUTATIONS)}]", flush=True)
    print(f"\n{'TOTAL':45s} {tot_n:6d} {tot_hit / tot_n:8.0%}", flush=True)
    print(f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
