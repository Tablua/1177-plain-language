# Screenshots

Drop your UI screenshots in this folder and they will show up in the root README
(`ui.png` is referenced there; `inspector.png` and `chat.png` are optional extras).

## 1. Get the app running locally

```bash
cd 1177-plain-language

# (once) dependencies
pip install -r requirements.txt

# The trained weights are NOT in this repo (models/ is git-ignored). Either retrain
# (~55 min on an RTX 3060):
python train.py --model google/flan-t5-base --epochs 3 --batch 8 --out models/model_base_v2
# ...or, if you still have the trained folder from the hackathon, copy it to
# models/model_base_v2 (config.json, generation_config.json, model.safetensors,
# tokenizer.json, tokenizer_config.json). You can also point the SIMPLIFY_MODEL
# env var at any other folder.

# start the backend
uvicorn api:app --port 8000
```

Then open `frontend/index.html` directly in a browser (double-click the file) — no
build step.

## 2. Take the shots

1. Expand the first note (**Inpatient Care, Physician**). It auto-loads three public
   PLABA test abstracts (no patient data).
2. Click **Simplify text** on the first abstract and wait for the three columns to
   render: original clinical note | plain-language version for the patient | the
   model's colour-coded suggestion.
3. Capture with your OS tool (Windows: `Win+Shift+S`, macOS: `Cmd+Shift+4`), or use a
   full-page screenshot extension for a taller shot.

Suggested files:

- `ui.png` — the headline shot: the full three-column view. This is the image the
  root README shows.
- `inspector.png` — click a red sentence in the "Model suggestion" column so the
  inspector below it explains *why* the original was kept instead of the rewrite.
  (Clicking keeps the inspector on screen, which is easier to capture than hover.)
- `chat.png` — click a sentence, open "Ask the assistant for help" (bottom right),
  ask "What does it mean?" and show the answer. This one needs the optional
  Anthropic key in `frontend/.env` (see the root README).

Save them in this folder with these names, commit and push — they appear in the
README on GitHub automatically.
