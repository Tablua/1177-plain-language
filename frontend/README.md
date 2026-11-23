# Frontend

A static, 1177-style mockup of the medical record view with the plain-language feature added.
Open `index.html` directly in a browser — no build step, no server.

The page talks to the FastAPI backend (`../api.py`) running on port 8000:

- `GET /example` — loads a random PLABA test-split abstract into each paragraph
- `POST /simplify` — the three-column view: original | plain-language version for the
  patient | the model's colour-coded suggestion
- `POST /chat` — the chat assistant (see below)

## Chat assistant (optional)

The chat assistant uses Claude via the Anthropics API. Copy `.env.example` to `.env`
(git-ignored) and paste your key:

```
ANTHROPIC_API_KEY=
```

Everything else works without it.
