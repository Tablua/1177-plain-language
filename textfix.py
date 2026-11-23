"""Preprocessing for T5: cleaning + masking of characters the T5 tokenizer drops.

T5's vocabulary lacks '<', Greek letters, ± ≥ ≤ ≈ ≠ × ÷, superscript digits and arrows,
among others: 'ΔF508' becomes 'F508', 'IL-6 ≥ 5' becomes 'IL-6 5'. They are replaced with
word markers before the model and restored afterwards. Used by both train.py and
simplify.py so the model sees the same thing.

NOTE: if the list below changes, the model must be retrained to learn the new markers.
"""
import re

# character -> (marker, how spacing is restored): "right" = joins the next character (Δ, μ),
# "left" = joins the previous one (², ³), "spaced" = keeps surrounding spaces as they are (≥, ±)
SYMBOLS = {
    "<": ("lessthan", "spaced"),
    ">": ("greaterthan", "spaced"),
    "Δ": ("symdelta", "right"), "δ": ("symsmalldelta", "right"),
    "α": ("symalpha", "right"), "β": ("symbeta", "right"), "γ": ("symgamma", "right"),
    "σ": ("symsigma", "right"), "λ": ("symlambda", "right"), "π": ("sympi", "right"),
    "Ω": ("symomega", "right"), "ω": ("symsmallomega", "right"), "Σ": ("symcapsigma", "right"),
    "μ": ("symmu", "right"), "µ": ("symmu", "right"),  # Greek mu and the micro sign
    "±": ("symplusminus", "spaced"), "≥": ("symgeq", "spaced"), "≤": ("symleq", "spaced"),
    "≈": ("symapprox", "spaced"), "≠": ("symneq", "spaced"), "×": ("symtimes", "spaced"),
    "÷": ("symdiv", "spaced"), "→": ("symarrow", "spaced"), "~": ("symtilde", "spaced"),
    "²": ("symsup2", "left"), "³": ("symsup3", "left"), "¹": ("symsup1", "left"),
}
# marker -> character (the first character per marker wins, so µ/μ becomes μ)
_BACK = {}
for _ch, (_name, _how) in SYMBOLS.items():
    _BACK.setdefault(_name, (_ch, _how))


def clean(s):
    """Cleans whitespace and broken characters (U+FFFD where ° stood, 'Â·' = broken dot in 'p<0Â·001')."""
    s = s.replace(chr(0xFFFD), "°").replace("Â·", "·")
    s = re.sub(r"(?<=\d)·(?=\d)", ".", s)  # 0·001 -> 0.001
    s = s.replace("‑", "-").replace("…", "...")
    return " ".join(s.split())


def mask(s):
    for ch, (name, _) in SYMBOLS.items():
        s = s.replace(ch, f" {name} ")
    return s


def unmask(s):
    for name, (ch, how) in _BACK.items():
        if how == "right":      # Δ + F508 -> ΔF508, 5 + μ + g -> 5 μg (space before kept)
            s = re.sub(rf"(\s?){name}\s?", lambda m: m.group(1) + ch, s, flags=re.I)
        elif how == "left":     # m + ² -> m²
            s = re.sub(rf"\s?{name}(\s?)", lambda m: ch + m.group(1), s, flags=re.I)
        elif name in ("lessthan", "greaterthan"):  # p<0.05 (as before: drop surrounding spaces)
            s = re.sub(rf"\s*{name}\s*", ch, s, flags=re.I)
        else:                   # ≥ ±: the marker is replaced, surrounding spaces kept
            s = re.sub(name, ch, s, flags=re.I)
    s = re.sub(r"\s+", " ", s)            # clean up double spaces the masking creates
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    return s.strip()
