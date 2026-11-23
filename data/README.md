# Data

`data.json` — the PLABA dataset (Plain Language Adaptation of Biomedical Abstracts),
sentence-aligned: the abstract's sentence `"3"` has its plain-language version under the
same key `"3"` in `adaptations`, which gives us `source_id` for free
(`<pmid>-<sentence-number>`).

Used by `train.py` (fine-tuning), `results.py` (evaluation) and `api.py` (`/example` —
the UI loads random abstracts from the test split, so the demo never touches patient
data).

## Splits

PLABA's own per-question split (question ids are listed in `train.py`):

- train: 6,307 sentence pairs
- validation: 1,431 pairs
- test: 1,347 pairs

The test split has never been used for training or tuning.

## Source and license

PLABA — Plain Language Adaptation of Biomedical Abstracts.
Attal, K., Ondov, B. & Demner-Fushman, D. *Scientific Data* 9, 803 (2023).
DOI: [10.1038/s41597-022-01920-3](https://doi.org/10.1038/s41597-022-01920-3) ·
download: <https://osf.io/rnpmf/>

The texts are public PubMed abstracts — **no patient data**. The paper is open access;
check the license statement on the OSF page before redistributing the dataset yourself.

## Known pitfall

The text contains `35-36 �C` where `°` has been lost (U+FFFD). The ingest step
(`textfix.clean`) replaces `�` with `°`, otherwise the unit would be misread during
flagging.
