"""Fine-tunes flan-t5 on PLABA sentence pairs (hard sentence -> plain sentence).

Run:  python train.py                      (quick run, flan-t5-small, 1 epoch)
      python train.py --model google/flan-t5-base --epochs 3 --batch 8 --out models/model_base_v2
Saves the model under models/.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textfix import clean as textfix_clean, mask

HERE = Path(__file__).parent
PREFIX = "Simplify this medical sentence into plain language: "

# PLABA's own per-question split (from the baseline code). Everything else = train.
TEST_Q = {"5", "12", "16", "22", "30", "36", "42", "48", "54", "61", "68"}
VAL_Q = {"2", "7", "13", "17", "26", "34", "40", "46", "52", "58", "66"}


def load_pairs(path):
    """Returns {'train': [...], 'val': [...], 'test': [...]} of (hard, plain) pairs."""
    data = json.load(open(path, encoding="utf-8"))
    out = {"train": [], "val": [], "test": []}
    for qid, q in data.items():
        split = "test" if qid in TEST_Q else "val" if qid in VAL_Q else "train"
        for pmid, doc in q.items():
            if pmid in ("question", "question_type"):
                continue
            for adapt in doc["adaptations"].values():
                for sid, src in doc["abstract"].items():
                    tgt = adapt.get(sid, "")
                    src, tgt = clean(src), clean(tgt)
                    if src and tgt:  # skip sentences the human annotator deleted
                        out[split].append((src, tgt))
    return out


def clean(s):
    # fix broken characters + mask '<'/'>' (T5 cannot represent them), same as at inference time
    return mask(textfix_clean(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 = all (set low for a quick test)")
    ap.add_argument("--data", default=str(HERE / "data" / "data.json"))
    ap.add_argument("--out", default=str(HERE / "models" / "model"))
    a = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs(a.data)
    print({k: len(v) for k, v in pairs.items()}, "device:", dev)
    train = pairs["train"][:]
    random.shuffle(train)
    if a.max_pairs:
        train = train[: a.max_pairs]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(dev)

    def collate(batch):
        x = tok([PREFIX + s for s, _ in batch], max_length=128, truncation=True,
                padding=True, return_tensors="pt")
        y = tok([t for _, t in batch], max_length=128, truncation=True,
                padding=True, return_tensors="pt").input_ids
        y[y == tok.pad_token_id] = -100
        x["labels"] = y
        return x

    def val_loss():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for b in DataLoader(pairs["val"], batch_size=a.batch, collate_fn=collate):
                b = {k: v.to(dev) for k, v in b.items()}
                tot += model(**b).loss.item()
                n += 1
        model.train()
        return tot / max(n, 1)

    loader = DataLoader(train, batch_size=a.batch, shuffle=True, collate_fn=collate)
    steps = int(len(loader) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    print(f"val loss before training: {val_loss():.3f}")
    step = 0
    t0 = time.time()
    while step < steps:
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == steps:
                el = time.time() - t0
                eta = el / step * (steps - step)
                print(f"step {step}/{steps} ({step / steps:.0%})  loss {loss.item():.3f}  "
                      f"elapsed {el / 60:.1f} min  eta ~{eta / 60:.1f} min", flush=True)
            if step >= steps:
                break
    print(f"val loss after training: {val_loss():.3f}")
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved in", a.out)


if __name__ == "__main__":
    main()
