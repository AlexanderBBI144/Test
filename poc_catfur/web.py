"""FastAPI playground for the cat-fur encoder.

Endpoints:
  GET  /                 single-page UI.
  POST /api/roundtrip    {text, level, seed} -> PNGs + decoded phrases.
  POST /api/decode       multipart upload -> decoded phrases.

Run:
    uv run --extra poc python -m poc_catfur.web
"""

import base64
import io
import os
import tempfile
import time
from functools import cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from . import decoder, distort, encoder, tokenize, vocab
from . import shape_encoder as shape_enc
from . import shape_scan_decoder as shape_sdec

INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="cat-fur codec")


@cache
def vocab_bundle() -> dict:
    return vocab.load_or_build()


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def phrase_list(v: np.ndarray, vb: dict, k: int = 10) -> list[dict]:
    return [{"phrase": p, "sim": s} for p, s in vocab.lookup_phrases(v, vb, k=k)]


class RoundtripRequest(BaseModel):
    text: str
    level: str | None = None
    seed: int = 0


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/roundtrip")
def api_roundtrip(req: RoundtripRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")

    vb = vocab_bundle()
    t0 = time.time()
    norm = tokenize.normalize(text)
    v = vocab.project(norm, vb)

    clean = encoder.encode(v, vb["v_lo"], vb["v_hi"])
    channel = distort.distort(clean, seed=req.seed, level=req.level) if req.level else clean

    resp = {
        "ok": True,
        "error": None,
        "normalized": norm,
        "subwords": tokenize.subword_tokens(text),
        "words": tokenize.words(text),
        "level": req.level,
        "clean_png": png_b64(clean),
        "channel_png": png_b64(channel),
        "phrase_matches": [],
        "cos_to_original": 0.0,
        "n_blobs": None,
    }
    try:
        v_hat, diag = decoder.decode(channel, vb["v_lo"], vb["v_hi"])
        resp["phrase_matches"] = phrase_list(v_hat, vb)
        resp["cos_to_original"] = float(np.dot(v_hat, v))
        resp["n_blobs"] = diag.get("n_blobs")
    except Exception as e:
        resp["ok"] = False
        resp["error"] = str(e)

    resp["elapsed_ms"] = int((time.time() - t0) * 1000)
    return resp


@app.post("/api/decode")
async def api_decode(image: UploadFile = File(...)) -> dict:
    img = Image.open(io.BytesIO(await image.read())).convert("L")
    img = img.resize((1400, 1400), Image.BICUBIC)
    vb = vocab_bundle()
    try:
        v_hat, diag = decoder.decode(img, vb["v_lo"], vb["v_hi"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "phrase_matches": phrase_list(v_hat, vb),
        "n_blobs": diag.get("n_blobs"),
        "uploaded_png": png_b64(img),
    }


class ShapeEncodeRequest(BaseModel):
    text: str


@app.post("/api/shape/encode")
def api_shape_encode(req: ShapeEncodeRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        svg = shape_enc.encode(v)
        b64 = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
        return {"ok": True, "svg_b64": b64, "svg": svg}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/shape/decode")
async def api_shape_decode(image: UploadFile = File(...)) -> dict:
    data = await image.read()
    try:
        from .shape_vec2text import top_phrases
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            v = shape_sdec.decode_image(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        phrases = top_phrases(v, n=10)
        return {"ok": True, "phrases": phrases}
    except Exception as e:
        return {"ok": False, "error": str(e), "phrases": []}


def main() -> None:
    import uvicorn

    print("Warming up vocab...")
    vocab_bundle()
    host = os.environ.get("CATFUR_HOST", "127.0.0.1")
    port = int(os.environ.get("CATFUR_PORT", "5050"))
    print(f"Ready. Serving at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
