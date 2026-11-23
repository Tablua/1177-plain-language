"""Measures the full system (model + fact hierarchy) for the results documentation.
Saves results_<split>.json.

Run:  python results.py --split test --n 300 --models model_base model_base_v2
Progress and remaining time are printed continuously. The test split has never been used
for training or tuning.
"""
import argparse
import json
import random
import statistics
import time
from collections import Counter

import simplify as S
from flagger import check
from textfix import unmask
from train import load_pairs


def run_model(name, srcs, refs):
    S._cache.clear()
    S.MODEL_DIR = S.HERE / "models" / name
    S._load()
    t0 = time.time()
    res = []
    for i in range(0, len(srcs), 16):
        batch = srcs[i:i + 16]
        for s, scored in zip(batch, S._generate(batch, S.NUM_CANDIDATES, scored=True)):
            res.append(S.choose(s, [c for c, _ in scored], [p for _, p in scored]))
        done = min(i + 16, len(srcs))
        el = time.time() - t0
        print(f"  [{name}] {done}/{len(srcs)} ({done/len(srcs):.0%}) elapsed {el/60:.1f} min "
              f"eta ~{el/done*(len(srcs)-done)/60:.1f} min", flush=True)
    secs = (time.time() - t0) / len(srcs)
    n = len(srcs)
    st = Counter(r["status"] for r in res)
    shown = [r["simplified"] for r in res]
    conf = {c: [r["confidence"] for r in res if r["status"] == c and r.get("confidence") is not None]
            for c in ("green", "yellow")}
    out = {
        "model": name, "n": n, "sec_per_sentence": round(secs, 2),
        "green": st["green"] / n, "yellow": st["yellow"] / n, "red": st["red"] / n,
        "original_shown": sum(sh == s for sh, s in zip(shown, srcs)) / n,
        "shown_with_fact_flag": sum(1 for s, sh in zip(srcs, shown) if sh != s and check(s, sh)),
        "fkgl_original": statistics.mean(S.fkgl(s) for s in srcs),
        "fkgl_shown": statistics.mean(S.fkgl(x) for x in shown),
        "fkgl_human": statistics.mean(S.fkgl(r) for r in refs),
        "confidence_green_mean": statistics.mean(conf["green"]) if conf["green"] else None,
        "confidence_yellow_mean": statistics.mean(conf["yellow"]) if conf["yellow"] else None,
        "examples": {c: [{"original": s, "model": r["model_output"], "shown": r["simplified"],
                          "confidence": r.get("confidence"), "reason": r["reasons"][0]}
                         for s, r in zip(srcs, res) if r["status"] == c][:3]
                     for c in ("green", "yellow", "red")},
    }
    print(f"{name}: GREEN {out['green']:.1%} | YELLOW {out['yellow']:.1%} | RED {out['red']:.1%} | "
          f"original shown {out['original_shown']:.1%} | shown text with a fact flag {out['shown_with_fact_flag']} | "
          f"FKGL original {out['fkgl_original']:.1f} -> shown {out['fkgl_shown']:.1f} (human {out['fkgl_human']:.1f}) | "
          f"{secs:.2f} s/sentence", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=["model_base", "model_base_v2"])
    a = ap.parse_args()
    pairs = load_pairs(S.HERE / "data" / "data.json")[a.split]
    random.seed(0)
    sample = random.sample(pairs, min(a.n, len(pairs)))
    srcs, refs = [unmask(s) for s, _ in sample], [unmask(t) for _, t in sample]
    print(f"Split: {a.split}, {len(srcs)} sentences (random, seed 0)", flush=True)
    results = [run_model(m, srcs, refs) for m in a.models]
    path = S.HERE / f"results_{a.split}.json"
    json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Saved:", path.name, "\nDONE", flush=True)


if __name__ == "__main__":
    main()
