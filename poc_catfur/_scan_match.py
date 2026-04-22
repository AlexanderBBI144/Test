"""Try every SVG × orientation to find which was printed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from . import shape_decoder as dec
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
OUT_DIR = Path(__file__).parent / "out"

SVG_CANDIDATES = {
    "i_love_geese": ["shape_i_love_geese.svg", "scan_i_love_geese.svg"],
    "i_hate_spiders": ["shape_i_hate_spiders.svg"],
    "chocolate_is_delicious": ["shape_chocolate_is_delicious.svg"],
    "the_sky_is_blue": ["shape_the_sky_is_blue.svg"],
    "the_sun_is_shining": ["shape_the_sun_is_shining.svg"],
    "geese_are_not_cats_but_birds": ["shape_geese_are_not_cats_but_birds.svg"],
}


def cyan_mask(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (g + b) * 0.5 - r


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def prepare_gray(photo_path: Path, flip: bool) -> Path:
    rgb = np.array(Image.open(photo_path).convert("RGB"))
    if flip:
        rgb = rgb[:, ::-1]
    cyan = cyan_mask(rgb)
    ink_gray = (255 - np.clip(cyan * 4.0, 0, 255)).astype(np.uint8)
    tmp = OUT_DIR / f"_match_{'flip' if flip else 'norm'}.png"
    Image.fromarray(ink_gray, mode="L").save(tmp)
    return tmp


def main():
    client = OpenAI()
    print("Caching ada-002 embeddings for all candidate phrases...")
    phrase_vec = {}
    for phrase in SVG_CANDIDATES:
        text = phrase.replace("_", " ")
        r = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        v = np.array(r.data[0].embedding, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        phrase_vec[phrase] = v

    path_norm = prepare_gray(PHOTO, flip=False)
    path_flip = prepare_gray(PHOTO, flip=True)

    results = []
    for phrase, files in SVG_CANDIDATES.items():
        for fname in files:
            svg = (OUT_DIR / fname).read_text()
            v_svg_dec = dec.decode(svg)
            # cos (svg_decoder, orig) as sanity — should be ~0.93 if codec correct
            cos_svg_orig = cosine(v_svg_dec, phrase_vec[phrase])

            for flip, img_path in [("norm", path_norm), ("flip", path_flip)]:
                try:
                    v_scan = sdec.decode_image(img_path.as_posix(), svg)
                    cos_scan_orig = cosine(v_scan, phrase_vec[phrase])
                    cos_scan_svg = cosine(v_scan, v_svg_dec)
                except Exception as e:
                    cos_scan_orig = float("nan")
                    cos_scan_svg = float("nan")
                    print(f"  {fname:45s} {flip}: ERROR {e}")
                    continue
                results.append((phrase, fname, flip, cos_svg_orig, cos_scan_svg, cos_scan_orig))

    print(f"\n{'phrase':<32s} {'svg':<40s} {'flip':<5s} {'cos_svg_orig':>12s} {'cos_scan_svg':>12s} {'cos_scan_orig':>13s}")
    for r in sorted(results, key=lambda x: -x[5]):
        print(f"{r[0]:<32s} {r[1]:<40s} {r[2]:<5s} {r[3]:>12.4f} {r[4]:>12.4f} {r[5]:>13.4f}")


if __name__ == "__main__":
    main()
