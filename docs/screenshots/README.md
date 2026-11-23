# Screenshots

The four shots used in the root README live in this folder:

- `start.png` — the start page (the note list) before simplification. This is the
  headline image at the top of the README.
- `green.png` — a sentence whose suggestion came out **green**: facts correct and
  clearly simplified, so the plain-language rewrite is shown.
- `yellow.png` — a sentence whose suggestion came out **yellow**: facts correct but
  little change.
- `red.png` — a sentence whose suggestion came out **red**: facts changed, so the
  ORIGINAL is kept instead of the rewrite (with the inspector explaining why).

## Re-capturing the shots

### 1. Get the app running locally

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

### 2. Take the shots

1. Capture the **start page** before clicking anything (or reload the page to reset) —
   that's `start.png`. The first note (**Inpatient Care, Physician**) can be expanded so
   the auto-loaded PLABA test abstract (no patient data) is visible.
2. Click **Simplify text** on the first abstract and wait for the three columns to
   render: original clinical note | plain-language version for the patient | the
   model's colour-coded suggestion.
3. Capture one shot per outcome colour from the "Model suggestion" column — hover (or
   click) a sentence of each colour. Clicking a red sentence also pins the inspector
   below it, which explains *why* the rewrite was rejected.
4. Capture with your OS tool (Windows: `Win+Shift+S`, macOS: `Cmd+Shift+4`), or use a
   full-page screenshot extension for a taller shot.

Save them in this folder with the names above, commit and push — they appear in the
README on GitHub automatically.
