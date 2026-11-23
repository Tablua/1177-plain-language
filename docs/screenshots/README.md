# Screenshots

Drop your UI screenshots in this folder and they will show up in the root README.

To take them:

1. Train or point to the model and start the backend (see the root README):
   `uvicorn api:app --port 8000`
2. Open `frontend/index.html` in a browser.
3. Click **Simplify text** on the first note and wait for the three columns to render.

Suggested shots (save with these names):

- `ui.png` — the full three-column view: original clinical note | plain-language
  version for the patient | the model's colour-coded suggestion. This is the headline
  screenshot referenced in the README.
- `inspector.png` — hover over (or click) a red sentence so the risk inspector shows
  *why* the original was kept instead of the model's suggestion.
- `chat.png` — click a sentence, open the "Ask the assistant" panel and show a
  question/answer exchange.

The README references `docs/screenshots/ui.png`; the others are optional extras you can
also reference yourself.
