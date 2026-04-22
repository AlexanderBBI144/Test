"""Empirical capacity test: JL-projected ada-002 + vec2text corrector.

For each projection dim k, embed phrases → project to k-d (fixed-seed
orthonormal basis P) → lift back to 1536-d → unit-normalise → run
vec2text inversion. Measure cos(embed(inverted), embed(original)).

Run:  uv run --extra poc --extra invert python -m poc_catfur.experiments.capacity_test
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # must run before hf imports

import csv
import numpy as np
import torch
import vec2text
from openai import OpenAI
from tqdm import tqdm

ADA = "text-embedding-ada-002"
D = 1536
KS = [64, 128, 256, D]  # last is identity baseline
STEPS = 1
SEED = 0xCAFE

PHRASES = [
    "i love geese",
    "chocolate is delicious",
    "the sky is blue",
    "geese are not cats but birds",
    "i hate spiders",
    "the sun is shining",
]


def embed_ada(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=ADA, input=texts)
    arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def jl_basis(k: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, k)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q  # (D, k) orthonormal columns


def main() -> None:
    client = OpenAI()
    print("Loading vec2text corrector from cache...", flush=True)
    # Force CPU: patch out MPS so vec2text loads all weights there from the start
    _mps_avail = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    corrector = vec2text.load_pretrained_corrector(ADA)
    torch.backends.mps.is_available = _mps_avail
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    print("Embedding originals...", flush=True)
    V_orig = embed_ada(client, PHRASES)  # (N, D)

    out_dir = Path(__file__).parent
    out_csv = out_dir / "capacity_results.csv"
    summaries: dict[int, list[float]] = {}

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "phrase", "inverted", "cos_to_orig"])

        for k in tqdm(KS, desc="k-dims", position=0):
            tqdm.write(f"\n=== k = {k} ===")
            if k == D:
                V_lifted = V_orig.copy()
            else:
                P = jl_basis(k)
                V_packed = V_orig @ P
                V_lifted = V_packed @ P.T
            norms = np.linalg.norm(V_lifted, axis=1, keepdims=True) + 1e-12
            V_lifted /= norms

            inverted_all: list[str] = []
            phrase_bar = tqdm(enumerate(PHRASES), total=len(PHRASES), desc=f"  phrases", position=1, leave=False)
            for i, p in phrase_bar:
                V_t = torch.from_numpy(V_lifted[i : i + 1]).float().to(device)
                inv_list = vec2text.invert_embeddings(embeddings=V_t, corrector=corrector, num_steps=STEPS)
                inverted_all.append(inv_list[0])
                phrase_bar.set_postfix({"last": inv_list[0][:30]})

            V_inv = embed_ada(client, inverted_all)
            scores = (V_inv * V_orig).sum(axis=1)
            for p, inv, c in zip(PHRASES, inverted_all, scores):
                w.writerow([k, p, inv, f"{c:.4f}"])
                tqdm.write(f"  cos={c:.3f}  {p!r} -> {inv!r}")
            summaries[k] = scores.tolist()
            f.flush()

    print("\n=== SUMMARY ===")
    print(f"{'k':>6} {'mean':>7} {'median':>7} {'>0.85':>7} {'>0.70':>7}")
    for k, scores in summaries.items():
        arr = np.array(scores)
        print(
            f"{k:>6} {arr.mean():>7.3f} {np.median(arr):>7.3f}  "
            f"{(arr > 0.85).mean():>6.2f}  {(arr > 0.70).mean():>6.2f}"
        )


if __name__ == "__main__":
    main()
