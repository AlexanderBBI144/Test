"""Tiny Flask playground for the cat-fur encoder.

Endpoints:
  GET  /                 — single-page UI.
  POST /api/roundtrip    — {text, level} -> encoded PNG + (optionally
                           distorted) PNG + decoded top-k words.
  POST /api/decode       — multipart upload of a photograph -> top-k
                           words from the decoder.

Run:
    uv run --extra poc python -m poc_catfur.web
"""
from __future__ import annotations

import base64
import io
import time

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from . import decoder, distort, encoder, tokenize, vocab

app = Flask(__name__, template_folder="templates", static_folder="static")

_vocab = None


def vocab_bundle():
    global _vocab
    if _vocab is None:
        _vocab = vocab.load_or_build()
    return _vocab


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/roundtrip", methods=["POST"])
def api_roundtrip():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    level = data.get("level") or None
    seed = int(data.get("seed", 0))
    if not text:
        return jsonify(error="empty text"), 400

    vb = vocab_bundle()
    t0 = time.time()
    norm = tokenize.normalize(text)
    subwords = tokenize.subword_tokens(text)
    words = tokenize.words(text)
    v = vocab.project(norm, vb)

    clean_img = encoder.encode(v, vb["v_lo"], vb["v_hi"])
    channel_img = clean_img if not level else distort.distort(clean_img, seed=seed, level=level)

    try:
        v_hat, diag = decoder.decode(channel_img, vb["v_lo"], vb["v_hi"])
        matches = vocab.lookup(v_hat, vb, k=8)
        cos_orig = float(np.dot(v_hat, v))
        ok = True
        err = None
    except Exception as e:  # decoder can fail on absurd distortions
        matches = []
        cos_orig = 0.0
        ok = False
        err = str(e)
        diag = {}

    return jsonify(
        ok=ok,
        error=err,
        normalized=norm,
        subwords=subwords,
        words=words,
        level=level,
        clean_png=_png_b64(clean_img),
        channel_png=_png_b64(channel_img),
        matches=[{"word": w, "sim": s} for w, s in matches],
        cos_to_original=cos_orig,
        n_blobs=diag.get("n_blobs"),
        elapsed_ms=int((time.time() - t0) * 1000),
    )


@app.route("/api/decode", methods=["POST"])
def api_decode():
    if "image" not in request.files:
        return jsonify(error="missing image"), 400
    img = Image.open(request.files["image"].stream).convert("L")
    # Normalise to the canonical canvas by resizing square.
    img = img.resize((1400, 1400), Image.BICUBIC)

    vb = vocab_bundle()
    try:
        v_hat, diag = decoder.decode(img, vb["v_lo"], vb["v_hi"])
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 200
    matches = vocab.lookup(v_hat, vb, k=10)
    return jsonify(
        ok=True,
        matches=[{"word": w, "sim": s} for w, s in matches],
        n_blobs=diag.get("n_blobs"),
        uploaded_png=_png_b64(img),
    )


def main() -> None:
    print("Warming up vocab...")
    vocab_bundle()
    print("Ready. Serving at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
