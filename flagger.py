"""Fact check: could a simplification have changed the meaning? Pure Python rules, no model.

    from flagger import check
    check(original, simplified)  ->  [{"code", "severity", "detail"}, ...]

In strict mode (simplify.choose) EVERY flag counts as an error, except not_simplified.
severity (high/medium/low) only describes how serious the problem is, for the UI explanation.

Codes:
  number_changed     high   a number in the simplification that is not in the original
  unit_changed       high   same number but a different unit (5 mg -> 5 g)
  number_changed     low    a number from the original is missing
  negation_flipped   high   a negation added/removed (parity; "not only", "gram-negative" don't count)
  hedge_changed      medium uncertainty removed/added ("may reduce" -> "reduces")
  laterality_changed high   left/right swapped or dropped
  time_changed       medium "history of"/"previous" removed or added
  drug_changed       high   a medicine in the simplification that is not in the original
  drug_changed       low    a medicine from the original is missing
  qualifier_changed  medium an important qualifier dropped (systolic, chronic, severe ... see QUALIFIERS)
  qualifier_changed  high   an important qualifier added that is not in the original
  term_changed       high   an abbreviation/term in the simplification that is not in the original (CHF -> COVID-19)
  term_changed       low    an abbreviation from the original is missing
  word_swapped       medium an uncommon word replaced by a similar-looking different word (decorticate -> decorative)
  connective_changed medium "and" turned into "or"
  unsupported_addition medium an explanation/clause added that is not in the original
  symbol_changed     medium a special symbol (Δ, β, μ, ±, ≥, <, >) removed or garbled
  content_dropped    medium the model gave no answer
  not_simplified     low    the sentence came back unchanged (not an error, just not simplified)
"""
import re
from difflib import SequenceMatcher

from textfix import SYMBOLS

# ---------- numbers and units ----------
_UNITS = (r"mg/kg|mcg/kg|ml/kg|mg/dl|mg/ml|mg/l|g/dl|g/l|mmol/l|µmol/l|μmol/l|umol/l|ng/ml|pg/ml|iu/l|u/l|mmhg|"
          r"mcg|µg|μg|ug|mg|kg|ml|dl|mmol|iu|bpm|cm|mm|g|l")
NUM_UNIT = re.compile(rf"(\d+(?:\.\d+)?)\s?({_UNITS})(?![A-Za-z])", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
_UNIT_ALIAS = {"µg": "mcg", "μg": "mcg", "ug": "mcg", "µmol/l": "umol/l", "μmol/l": "umol/l"}
_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
              "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100", "half": "50",
              "once": "1", "twice": "2", "thrice": "3"}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def _numbers(text):
    """All numbers in the text: digits + numbers written as words (two -> 2, sixty-nine -> 69)."""
    low = text.lower()
    vals = set(NUM.findall(text))
    combo = re.compile(r"\b(" + "|".join(_TENS) + r")[- ](" + "|".join(_ONES) + r")\b")
    for t, o in combo.findall(low):
        vals.add(str(_TENS[t] + _ONES[o]))
    low = combo.sub(" ", low)  # "sixty-nine" must not also yield "nine"
    for w in re.findall(r"\b(" + "|".join(_NUM_WORDS) + r")\b", low):
        vals.add(_NUM_WORDS[w])
    return vals


def _num_units(text):
    return {(v, _UNIT_ALIAS.get(u.lower(), u.lower())) for v, u in NUM_UNIT.findall(text)}


# ---------- negation, uncertainty, side, time ----------
_PSEUDO_NEG = re.compile(r"\bnot only\b|\bnot just\b|\bno\.\s*\d|\bgram[- ]negative\b|\bnot (?:certain|sure)\b|"
                         r"\b(?:can(?:'?t|not)|could not|unable to)\s+(?:be\s+)?(?:rule[d]?|exclude[d]?)\s*(?:out)?\b|"
                         r"\bnot\s+(?:be\s+)?ruled out\b", re.I)
NEG = re.compile(r"\b(?:no|not|none|never|neither|nor|without|denies|denied|absence of|absent|free of|lack of|"
                 r"lacking|lacks|fail(?:s|ed)? to|unable to|cannot)\b|n't\b", re.I)
LAY_NEG = re.compile(r"\b(?:similar|the same|unchanged|equivalent|normal)\b", re.I)  # "did not differ" -> "were similar"
HEDGE = re.compile(r"\b(?:may|might|could|possible|possibly|probable|probably|likely|unlikely|suggests?|suggested|"
                   r"suggesting|appears?|seems?|potentially|uncertain|unclear|perhaps|maybe|"  # "potential" is often a noun
                   r"(?:can(?:'?t|not)|could not)\s+(?:be\s+)?(?:rule[d]?|exclude[d]?)\s*(?:out)?)\b", re.I)
_LAT = re.compile(r"\b(left|right|bilateral(?:ly)?|both sides|unilateral)\b", re.I)
_LAT_SKIP = re.compile(r"\bright (?:away|now|to|of way)\b|\ball right\b", re.I)
HIST = re.compile(r"\b(?:history of|previous(?:ly)?|prior(?! to)|past|former(?:ly)?|status post|resolved|"
                  r"recovered|used to|ago|earlier)\b", re.I)

# ---------- medicines (suffixes + common names) ----------
_DRUG_SUFFIX = re.compile(r"\b[a-z]{2,}(?:mab|nib|pril|sartan|statin|olol|azole|cillin|mycin|oxacin|cycline|formin|"
                          r"gliptin|glitazone|parin|vir|dipine|prazole|tidine|setron|triptan|barbital|caine|profen|"
                          r"fenac|coxib|thiazide|semide|platin|rubicin|solone|terol|zepam|zolam)\b", re.I)
_DRUGS = {"aspirin", "insulin", "warfarin", "heparin", "morphine", "codeine", "digoxin", "paracetamol",
          "acetaminophen", "tramadol", "oxycodone", "fentanyl", "amiodarone", "lithium", "tamoxifen",
          "methotrexate", "prednisone", "levothyroxine", "clopidogrel", "apixaban", "rivaroxaban", "amoxicillin",
          "gabapentin", "diazepam", "sertraline", "fluoxetine", "ibuprofen", "naproxen", "albuterol",
          "salbutamol", "nitroglycerin", "epinephrine", "quinine", "duloxetine", "azithromycin", "rifampin",
          "clindamycin", "penicillin"}

# symbols that may be written out as words in the simplification ("<" -> "less than")
_SYM_WORDS = {"<": r"less than|below|under|fewer than|lower than", ">": r"more than|greater than|above|over|higher than",
              "≤": r"at most|no more than|up to", "≥": r"at least|or more", "±": r"plus or minus",
              "Δ": r"change|difference|delta"}

# Important qualifiers that change the meaning if they disappear or get added.
# word -> plain-language equivalents that count as "the same thing" (chronic -> long-term is ok,
# but the word simply disappearing is not). Words also match with suffixes
# (significant/significantly, adult/adults). Extend freely.
QUALIFIERS = {
    "systolic": ["top number", "upper number", "upper"],
    "diastolic": ["bottom number", "lower number"],
    "significant": ["clear", "real", "meaningful", "noticeable", "important",
                    "similar", "the same", "no difference", "unchanged"],  # "no significant difference" -> "similar"
    "acute": ["sudden", "short-term", "short term"],
    "chronic": ["long-term", "long term", "long-lasting", "long lasting", "ongoing", "lasting"],
    "mild": ["slight", "minor", "small"],
    "moderate": ["medium", "middle"],
    "severe": ["serious", "very bad", "extreme"],
    "benign": ["not cancer", "non-cancerous", "noncancerous", "harmless"],
    "malignant": ["cancer", "cancerous"],
    "metastatic": ["spread", "spreading"],
    "fatal": ["death", "deadly", "die", "died", "kill"],
    "lethal": ["deadly", "death", "die", "kill"],
    "congenital": ["born with", "from birth", "at birth"],
    "hereditary": ["inherited", "runs in families", "genetic"],
    "pediatric": ["child", "children", "kids"], "paediatric": ["child", "children", "kids"],
    "neonatal": ["newborn", "newborns", "new-born"],
    "elderly": ["older", "old"],
    "maternal": ["mother", "mothers"],
    "fetal": ["baby", "unborn"], "foetal": ["baby", "unborn"],
    "intravenous": ["into a vein", "through a vein", "in a vein", "iv", "drip"],
    "oral": ["by mouth", "mouth", "swallow"],
    "topical": ["on the skin", "skin"],
    "subcutaneous": ["under the skin"],
    "daily": ["every day", "each day", "a day", "per day"],
    "weekly": ["every week", "each week", "a week", "per week"],
    "twice": ["two times", "2 times"],
    "placebo": ["sham", "dummy", "fake", "inactive"],
}

SYMBOL_WORDS = {"less", "than", "more", "greater", "below", "above", "under", "over", "fewer", "lower", "higher",
                "least", "most", "plus", "minus", "change", "difference", "delta"}

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-()]*")
WORD = re.compile(r"[a-z]{6,}")
STOPWORDS = {"that", "this", "these", "those", "with", "from", "into", "about", "also", "were", "have",
             "been", "they", "their", "which", "when", "what", "will", "would", "could", "than",
             "then", "there", "such", "some", "more", "most", "other", "very", "only", "being"}


def _is_acronym(tok):
    """Abbreviation/term code: at least two capitals (CHF, COVID-19, HbA1c) or letters mixed with digits."""
    letters = sum(c.isalpha() for c in tok)
    if letters == 0:
        return False
    upper = sum(c.isupper() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    return upper >= 2 or (has_digit and letters >= 1 and tok[0].isalpha() and upper >= 1)


def _acronyms(text):
    return {t.strip("()-") for t in TOKEN.findall(text) if _is_acronym(t.strip("()-"))}


def _same_stem(a, b):
    """Inflected forms of the same word (effective/effectiveness, therapeutical/therapies)."""
    return b.startswith(a[:-3]) or a.startswith(b[:-3])


def _laterality(text):
    found = set()
    for m in _LAT.finditer(_LAT_SKIP.sub(" ", text)):
        w = m.group(1).lower()
        found.add("left" if w == "left" else "right" if w == "right" else "bilateral")
    return {"bilateral"} if {"left", "right"} <= found else found


def _drugs(text):
    low = text.lower()
    return {m.group(0).lower() for m in _DRUG_SUFFIX.finditer(low)} | {d for d in _DRUGS if re.search(rf"\b{d}\b", low)}


def _flag(code, severity, detail):
    return {"code": code, "severity": severity, "detail": detail}


def check(original, simplified):
    flags = []
    o, s = original or "", simplified or ""
    if not s.strip():
        return [_flag("content_dropped", "medium", "The model gave no answer for this sentence.")]

    if re.findall(r"[a-z0-9]+", o.lower()) == re.findall(r"[a-z0-9]+", s.lower()):
        return [_flag("not_simplified", "low", "The sentence was not simplified (returned unchanged). Meaning is intact.")]

    # numbers (digits and number words count as the same: "two" == "2")
    on, sn = _numbers(o), _numbers(s)
    if sn - on:
        flags.append(_flag("number_changed", "high", f"New number not in the original: {', '.join(sorted(sn - on))}."))
    if on - sn:
        flags.append(_flag("number_changed", "low", f"Number from the original is missing: {', '.join(sorted(on - sn))}."))
    # units: same value but a different unit
    ou, su = _num_units(o), _num_units(s)
    changed = sorted({f"{v} {u}" for v, u in ou if (v, u) not in su and any(v == v2 and u != u2 for v2, u2 in su)})
    if changed:
        flags.append(_flag("unit_changed", "high", f"Same number but a different unit: {', '.join(changed)}."))

    # negation: count markers (false ones like "not only" removed), compare parity; plain-language equivalents count
    neg_o = len(NEG.findall(_PSEUDO_NEG.sub(" ", o)))
    s_neg = _PSEUDO_NEG.sub(" ", s)
    for w, eq in QUALIFIERS.items():  # "benign" -> "not cancer": the negation is part of the explanation
        if re.search(rf"\b{re.escape(w)}\w*", o.lower()):
            for e in eq:
                if NEG.search(e):
                    s_neg = re.sub(rf"\b{re.escape(e)}\b", " ", s_neg, flags=re.I)
    neg_s = len(NEG.findall(s_neg))
    if neg_s < neg_o:
        neg_s += min(len(LAY_NEG.findall(s)), neg_o - neg_s)
    if neg_o % 2 != neg_s % 2:
        what = "removed" if neg_s < neg_o else "added"
        flags.append(_flag("negation_flipped", "high", f"A negation was {what} (no/not/without)."))

    # uncertainty ("may", "possible", "suggests"): removing it makes a claim sound more certain than it is
    ho, hs = bool(HEDGE.search(o)), bool(HEDGE.search(s))
    if ho != hs:
        flags.append(_flag("hedge_changed", "medium",
                           "Uncertainty removed - the original says 'may/possibly/suggests'." if ho
                           else "Uncertainty added that is not in the original."))

    # side (left/right)
    lo, ls = _laterality(o), _laterality(s)
    if lo != ls:
        flags.append(_flag("laterality_changed", "high", f"Side changed: original {sorted(lo) or 'none'}, "
                                                         f"simplified {sorted(ls) or 'none'}."))

    # time ("history of", "previous"): a past condition must not read as current
    to, ts = bool(HIST.search(o)), bool(HIST.search(s))
    if to != ts:
        flags.append(_flag("time_changed", "medium", "Past/history wording removed - may now read as current." if to
                           else "Past/history wording added that is not in the original."))

    # medicines
    do, ds = _drugs(o), _drugs(s)
    if ds - do:
        flags.append(_flag("drug_changed", "high", f"Medicine not in the original: {', '.join(sorted(ds - do))}."))
    if do - ds:
        flags.append(_flag("drug_changed", "low", f"Medicine from the original is missing: {', '.join(sorted(do - ds))}."))

    # important qualifiers: must not disappear (unless a plain-language equivalent is present) or get added
    ol, sl = o.lower(), s.lower()
    has = lambda text, w: re.search(rf"\b{re.escape(w)}\w*", text) is not None
    has_any = lambda text, words: any(re.search(rf"\b{re.escape(e)}\b", text) for e in words)
    dropped = [w for w, eq in QUALIFIERS.items() if has(ol, w) and not has(sl, w) and not has_any(sl, eq)]
    added = [w for w, eq in QUALIFIERS.items() if has(sl, w) and not has(ol, w) and not has_any(ol, eq)]
    if dropped:
        flags.append(_flag("qualifier_changed", "medium", f"Important word from the original is missing: {', '.join(dropped)}."))
    if added:
        flags.append(_flag("qualifier_changed", "high", f"Important word added that is not in the original: {', '.join(added)}."))

    # abbreviations/terms
    oa, sa = _acronyms(o), _acronyms(s)
    if sa - oa:
        flags.append(_flag("term_changed", "high", f"Term not in the original: {', '.join(sorted(sa - oa))}."))
    if oa - sa:
        flags.append(_flag("term_changed", "low", f"Term from the original is missing: {', '.join(sorted(oa - sa))}."))

    # a word replaced by a similar-looking but different word
    ow, sw = set(WORD.findall(o.lower())), set(WORD.findall(s.lower()))
    new_w, gone_w = sw - ow, ow - sw
    new_w = {n for n in new_w if not any(_same_stem(g, n) for g in gone_w)}
    gone_w = {g for g in gone_w if not any(_same_stem(g, n) for n in sw - ow)}
    swaps = sorted(f"{g} -> {n}" for n in new_w for g in gone_w if SequenceMatcher(None, g, n).ratio() >= 0.7)
    if swaps:
        flags.append(_flag("word_swapped", "medium",
                           f"A word was replaced by a similar-looking different word: {', '.join(swaps)}."))

    # special symbols: removed/garbled; a symbol written out as words ("<" -> "less than") is ok
    mu = lambda t: t.replace("µ", "μ")
    sym_o = {c for c in mu(o) if c in SYMBOLS}
    sym_s = {c for c in mu(s) if c in SYMBOLS}
    lost = {c for c in sym_o - sym_s if not (c in _SYM_WORDS and re.search(_SYM_WORDS[c], s.lower()))}
    bad = lost | (sym_s - sym_o)
    if bad:
        flags.append(_flag("symbol_changed", "medium",
                           f"Special symbol or comparison sign lost, added or garbled: {' '.join(sorted(bad))}."))

    # explanation added in parentheses
    o_set = set(re.findall(r"[a-z]+", o.lower()))

    def _new_words(inner):  # content words inside the parentheses that are not in the original (symbol words don't count)
        return [w for w in re.findall(r"[a-z]+", inner.lower())
                if len(w) >= 4 and w not in o_set and w not in STOPWORDS and w not in SYMBOL_WORDS]
    added = [m.group(1).strip() for m in re.finditer(r"\(([^()]{3,})\)", s)
             if m.group(1).strip().lower() not in o.lower()
             and len(_new_words(m.group(1))) >= 1]  # at least one new content word = not verifiable
    if added:
        flags.append(_flag("unsupported_addition", "medium", "Explanation added that is not in the original - "
                           "check it is correct: " + "; ".join(f'"{a}"' for a in added) + "."))

    # added content without parentheses: clearly longer AND several new content words that replace nothing
    o_words, s_words = re.findall(r"[a-z]+", o.lower()), re.findall(r"[a-z]+", s.lower())
    content = lambda ws, other: {w for w in ws if w not in set(other) and len(w) >= 4 and w not in STOPWORDS}
    new_content, removed = content(s_words, o_words), content(o_words, s_words)
    # ...or a consecutive run of at least 4 new content words (an inserted clause). A normal
    # replacement ("kidney disease", "high blood pressure") yields at most 2-3 new words in a row.
    run = longest = 0
    o_vocab = set(o_words)
    for w in s_words:
        if len(w) < 4 or w in STOPWORDS or w in SYMBOL_WORDS:
            continue                      # short words/function words do not break the run
        run = run + 1 if w not in o_vocab else 0
        longest = max(longest, run)
    if longest >= 4 or (len(s_words) > 1.3 * max(len(o_words), 1) and len(new_content) - 3 * len(removed) >= 3):
        flags.append(_flag("unsupported_addition", "medium", "The simplification adds content that is not in the "
                           "original: " + ", ".join(sorted(new_content)) + "."))

    count = lambda t, w: len(re.findall(rf"\b{w}\b", t.lower()))
    if count(s, "or") > count(o, "or") and count(s, "and") < count(o, "and"):
        flags.append(_flag("connective_changed", "medium", "'and' was changed to 'or'."))
    return flags
