"""Shape codec web app.

Endpoints:
  GET  /                   UI
  POST /api/shape/encode   {text} → SVG
  POST /api/shape/decode   multipart photo → top-10 phrases via vec2text

Run:
    uv run --extra poc python -m poc_catfur.web
"""

import base64
import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from . import shape_encoder as shape_enc
from . import shape_scan_decoder as shape_sdec

INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="shape codec")


class ShapeEncodeRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


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
    host = os.environ.get("CATFUR_HOST", "127.0.0.1")
    port = int(os.environ.get("CATFUR_PORT", "5050"))
    print(f"Serving at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
