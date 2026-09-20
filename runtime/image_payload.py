from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def to_train_image_object(image) -> Image.Image:
    return _to_pil(image)


def to_openai_image_url(image) -> dict:
    pil = _to_pil(image)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def decode_openai_image_url(url: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        raise ValueError("expected PNG data URL")
    raw = base64.b64decode(url[len(prefix):])
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)
