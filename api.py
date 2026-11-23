"""Bridge between the web UI and the simplification model. No simplification logic of its
own — it calls simplify.py.

Run from the repo root:   uvicorn api:app --port 8000

Endpoints (frontend/index.html works as-is against this backend):

  GET  /example    a random PLABA test-split abstract to paste into the UI
  POST /simplify   {"text": "..."}  ->  {"data": [{id, source_id, original, simplified,
                                                    risk, reasons, matched}, ...]}
  POST /chat       the chat assistant in frontend/index.html (loaded from frontend/chat.py)
"""
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import simplify as S

HERE = Path(__file__).parent
MAX_CHARS = 6000  # the longest PLABA abstract is ~5,250 characters
# PLABA's test-split question ids (same list as train.py), used only for /example
TEST_Q = {"5", "12", "16", "22", "30", "36", "42", "48", "54", "61", "68"}


@asynccontextmanager
async def lifespan(app):
    S._load()  # load the model at startup so the first request doesn't hang
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _include_chat():
    """The chat window in frontend/index.html calls POST /chat. That logic lives in
    frontend/chat.py (the chat assistant, Claude via the Anthropics API). The file is loaded
    unchanged so the page works fully against the simplification model. If the 'anthropic'
    package is missing, everything else still works. chat.py reads the key from
    ANTHROPIC_API_KEY or from frontend/.env (git-ignored)."""
    import importlib.util
    path = HERE / "frontend" / "chat.py"
    try:
        spec = importlib.util.spec_from_file_location("frontend_chat", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app.include_router(mod.router)
        print("Chat assistant (/chat) is active.")
    except Exception as e:
        print(f"Chat assistant (/chat) is not available ({type(e).__name__}: {e}). "
              f"Simplification still works.")


_include_chat()


class Request(BaseModel):
    text: str


# The colours come from the hierarchy in simplify.py (facts first, then simplicity):
#   green  -> risk LOW    facts correct and simplified          (the suggestion is shown)
#   yellow -> risk MEDIUM facts correct but little/no change    (shown if it passed the fact check)
#   red    -> risk HIGH   the simplification changed facts      (the ORIGINAL is shown)
RISK = {"green": "LOW", "yellow": "MEDIUM", "red": "HIGH"}


@app.get("/example")
def example():
    data = json.load(open(HERE / "data" / "data.json", encoding="utf-8"))
    docs = [(pmid, d) for q in TEST_Q for pmid, d in data.get(q, {}).items()
            if isinstance(d, dict) and "abstract" in d]
    pmid, doc = random.choice(docs)
    return {"pmid": pmid, "title": doc.get("Title", ""),
            "text": " ".join(s.strip() for s in doc["abstract"].values())}


@app.post("/simplify")
def simplify_text(req: Request):
    text = req.text.strip()
    if not text:
        return {"data": []}
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]  # truncate — better part of the text than an error
    try:
        r = S.simplify_selection(text)
    except Exception as e:  # a model error must never surface as a stack trace in the page
        raise HTTPException(status_code=500, detail=f"Model error: {type(e).__name__}")

    if r["mode"] == "glossary":  # one word / a short phrase: dictionary lookup, no model
        return {"data": [{
            "id": 1, "source_id": "s1", "original": text, "simplified": r["text"],
            "model_output": r["text"], "status": "green" if r["found"] else "yellow",
            "risk": "LOW" if r["found"] else "MEDIUM",
            "reasons": ["Dictionary definition (no model used)."] if r["found"]
                       else ["Word not in the glossary."],
            "matched": r["found"]}]}

    return {"data": [{
        "id": i, "source_id": s["source_id"], "original": s["original"],
        "simplified": s["simplified"],      # safe text: the original wherever the facts didn't hold
        "model_output": s["model_output"],  # the model's own suggestion (even the rejected ones)
        "status": s["status"],              # green | yellow | red
        "confidence": s.get("confidence"),  # 0-100 for green/yellow (a ranking, not calibrated), None for red
        "confidence_parts": s.get("confidence_parts"),
        "risk": RISK[s["status"]], "reasons": s["reasons"], "matched": True}
        for i, s in enumerate(r["sentences"], start=1)]}
