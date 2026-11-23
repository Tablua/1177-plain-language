# Plain-Language Medical Records — a 1177 concept add-on

A proof-of-concept add-on for **[1177](https://www.1177.se)** (Sweden's national health
portal) that shows medical record notes in plain language. Select a sentence and get an
explanation you can actually understand — with a safety net that guarantees the
simplification never changes the medical facts.

Built in one day during the **Hacking Healthcare hackathon**, where the team reached the
**final (top 5)**.

> **Prototype disclaimer:** this is a hackathon demo, not a product, and it is not
> affiliated with 1177. It runs on public PubMed research abstracts (the PLABA dataset),
> never on real patient data.

![Three-column view: original note, plain-language version, model suggestion](docs/screenshots/ui.png)

## What it does

Doctors' notes are written in medical language — "No erythema or edema", "afebrile",
"systolic blood pressure 165 mmHg". This concept adds a **"Simplify text"** button to the
1177 medical record view:

- **The patient view** shows the note rewritten in plain everyday language, sentence by
  sentence.
- **Every sentence is fact-checked** before it is shown. If the model's rewrite changes a
  fact (a number, a negation, a medicine, a side...), the patient sees the *original*
  sentence instead — never a wrong simplification.
- Hover any sentence to see why it was shown as-is, simplified, or rejected.
- **Ask the assistant** — click a sentence and ask a chat assistant what it means in
  simple words.

## How it works

```
clinical note
    │  split into sentences
    ▼
fine-tuned flan-t5-base   (4 suggestions per sentence, beam search)
    │
    ▼
rule-based fact check     (numbers, units, negations, medicines, abbreviations, symbols ...)
    │
    ▼
strict selection hierarchy (facts correct → no info lost → actually simplified → simplest → most likely)
    │
    ├─ green   facts correct and simplified       → show the simplification
    ├─ yellow  facts correct, little/no change    → show the simplification
    └─ red     facts changed                      → show the ORIGINAL sentence
```

- **Model** (`simplify.py`): `google/flan-t5-base` fine-tuned on PLABA sentence pairs
  (hard medical sentence → human-written plain version).
- **Fact check** (`flagger.py`): pure Python rules — no model, nothing that can
  hallucinate. It compares numbers, units, negations, uncertainty words, left/right, time
  markers, medicines, important qualifiers, abbreviations, similar-looking word swaps,
  special symbols and invented additions between the original and the rewrite.
- **Selection hierarchy** (`simplify.py`): among the suggestions that pass the fact
  check, the system picks the one that is most simplified (lowest reading level), ties
  broken by model confidence.
- **Glossary** (`glossary.py`): single words and short phrases are answered from a
  hand-written medical glossary instead of the model.
- **Chat assistant** (`frontend/chat.py`): Claude (Anthropics API) explains one clicked
  sentence in plain words. Only that one sentence is sent to the API; the system prompt
  forbids sharing it further.

## Results

All numbers measured on the **PLABA test split** (300 randomly sampled sentences,
seed 0), which was never used for training or tuning:

|                                                          | model_base | **model_base_v2** (final) |
|----------------------------------------------------------|-----------:|--------------------------:|
| 🟢 Green — facts correct, clearly simplified             |      48.0% |                   **51.7%** |
| 🟡 Yellow — facts correct, slightly simplified           |      35.7% |                    32.7% |
| 🔴 Red — facts changed, original shown                   |      16.3% |                  **15.7%** |
| Shown text containing a fact error the check can see     |          0 |                      **0** |
| Reading level (FKGL): original → shown                   | 16.0 → 14.6 |            16.0 → **14.6** |
| Reading level of the human simplifications               |       14.6 |                    14.6 |
| Confidence score, mean green / yellow                    | 81% / 74%  |              80% / 76%  |
| Seconds per sentence (GPU, RTX 3060)                     |       0.43 |                  **0.35** |

**84% of sentences are shown simplified** (green + yellow); for the remaining 16% the
original is shown because the model's rewrite changed facts.

**The fact check catches 99% of planted fact errors** (1,225 errors of 13 types planted
into human simplifications, `eval_fact_check.py`):

| Planted error                                        | Count (test) | Caught |
|------------------------------------------------------|-------------:|-------:|
| Changed number (12 → 120)                            |           76 |   100% |
| Changed unit (mg → g)                                |            5 |   100% |
| Added negation                                       |          192 |   100% |
| Removed negation                                     |           35 |    97% |
| Removed uncertainty word (*may*)                     |           36 |    83% |
| Removed important word (*systolic*, *chronic* …)     |           20 |    90% |
| Added important word (*severe*)                      |          103 |    96% |
| Swapped medicine                                     |           21 |   100% |
| Swapped abbreviation (CHF → COVID-19)                |           76 |   100% |
| Removed time marker (*history of*)                   |            3 |   100% |
| Made-up explanation in parentheses                   |          326 |   100% |
| Made-up clause                                       |          332 |   100% |
| **Total**                                            |      **1,225** |  **99%** |

Without the fact check, the raw model dropped 11.5% of the numbers from the originals on
the validation split — 0 in the shown text.

**Real errors the model made before the check existed, all now blocked:**

- "patients with **CHF**" (heart failure) → "patients with **COVID-19**"
- "**decorticate** posturing" → "**decorative** posturing"
- "α-synuclein" → "**beta**-synuclein"
- "systolic blood pressure" → "systolic blood pressure (**high blood pressure**)" — a wrong explanation
- an invented clause: "…posturing, *when the body's natural defense system is weak*"

## Running locally

Requires Python 3.10+ (tested on 3.12/3.13).

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Train the model (weights are not committed; this reproduces the final model.
#    ~55 min on an RTX 3060, faster on bigger GPUs)
python train.py --model google/flan-t5-base --epochs 3 --batch 8 --out models/model_base_v2

# 3. Start the backend
uvicorn api:app --port 8000

# 4. Open the UI
#    frontend/index.html — open it directly in a browser
```

The model folder can also be set with the `SIMPLIFY_MODEL` env var. Without a GPU the
model runs on CPU (~1 s/sentence).

### Chat assistant (optional)

The chat assistant needs an Anthropics API key. Copy `frontend/.env.example` to
`frontend/.env` (git-ignored) and paste your key:

```
ANTHROPIC_API_KEY=...
```

Everything else works without it.

## Repository layout

```
api.py               FastAPI bridge: /example, /simplify, /chat
simplify.py          sentence splitting, generation, strict selection hierarchy
flagger.py           rule-based fact check (the safety net)
textfix.py           masks symbols the T5 tokenizer drops (Δ, <, μ, ±, ...)
glossary.py / .json  hand-written medical word glossary
train.py             fine-tunes flan-t5-base on PLABA
results.py           measures the full system on the test split
eval_fact_check.py   measures the fact check by planting 13 types of errors
frontend/            the 1177-style UI (index.html) + chat assistant (chat.py)
data/data.json       PLABA dataset (public PubMed abstracts + human simplifications)
models/              trained weights (git-ignored, generated by train.py)
docs/screenshots/    UI screenshots
```

## Data

PLABA — Plain Language Adaptation of Biomedical Abstracts.
Attal, K., Ondov, B. & Demner-Fushman, D. *Scientific Data* 9, 803 (2023).
DOI: [10.1038/s41597-022-01920-3](https://doi.org/10.1038/s41597-022-01920-3),
downloaded from <https://osf.io/rnpmf/>.

749 PubMed abstracts with human-written plain-language simplifications, aligned sentence
by sentence: 6,307 train / 1,431 validation / 1,347 test pairs (PLABA's own per-question
split). The texts are public PubMed abstracts — **no patient data**. See
`data/README.md`.

## Limitations (honest)

- **The fact check is strict on purpose.** It also rejects many *good* rewrites: 75% of
  the human simplifications would get at least one flag. Better to show the original
  than risk a wrong simplification.
- **The fact check doesn't catch everything.** The 99% applies to the 13 error types
  that were planted. Green means safe, not guaranteed.
- **The confidence score is a ranking, not a calibrated probability.**
- **Reading level (FKGL) is a rough proxy** — it measures word/sentence length, not
  whether the text actually became understandable.
- **Research text, not real records.** Trained and tested on PubMed abstracts; real
  journal notes look different and would need separate testing.

## Contributors

Built by a team of four during the Hacking Healthcare hackathon (finalists, top 5):

- **Hugo** and **Ruben** — the core pipeline: fine-tuned model, training, evaluation and
  results (Hugo); data pipeline and quality estimation (Ruben)
- **Tindra** — chat assistant
- **Morgan** — UI/frontend
