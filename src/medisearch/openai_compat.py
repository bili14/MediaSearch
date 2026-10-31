# -*- coding: utf-8 -*-
"""OpenAI-style fallback endpoint: POST /v1/embeddings (for old scripts that
cannot speak MCP).

Semantics (aligned with OpenAI /v1/embeddings so a base_url swap just works):
  request  {"model":"wemm-embedding-2b"(optional), "input": "text" | [elements],
            "dimensions": N (optional, <=2048, truncated then L2 re-normalized)}
  element:
    - string            -> text embedding
    - {"type":"image", "image": base64 | data:URL | local path} -> image
      embedding (multimodal extension; OpenAI proper does not embed images)
  videos are not accepted here (no natural OpenAI-style video input; the MCP
  tool media_search does search videos).

  response {"object":"list", "data":[{"object":"embedding","index":i,"embedding":[...]}],
            "model":"wemm-embedding-2b", "usage":{"prompt_tokens":0,"total_tokens":0}}

Always the local model layer (independent of backend mode): the REST API has no
text->vector endpoint; only the model layer can produce vectors.
GPU bed rule: a loaded model elsewhere -> 503 lm_bed_busy (no double-loading).
First call JIT-loads the model (~10-60s); the serve process idle-releases it.
"""
from __future__ import annotations

import json
import time

try:  # package mode
    from . import model_layer
except ImportError:  # script mode
    import model_layer

MODEL_NAME = "wemm-embedding-2b"


class _Item:
    def __init__(self):
        self.kind = "text"      # text | image
        self.value = None


def _parse_input(body: dict) -> tuple:
    """Returns (items, error). error not None = 400."""
    inp = body.get("input")
    if inp is None:
        return [], "400: input is required (string or list)"
    if isinstance(inp, str):
        items = [_Item()]
        items[0].kind, items[0].value = "text", inp
        return items, None
    if isinstance(inp, list):
        if not inp:
            return [], "400: input list must not be empty"
        items = []
        for it in inp:
            if isinstance(it, str):
                o = _Item()
                o.kind, o.value = "text", it
                items.append(o)
                continue
            if isinstance(it, dict):
                t = it.get("type")
                if t == "image":
                    ref = it.get("image")
                    if not ref:
                        return [], "400: image element missing 'image' field (base64/data URL/path)"
                    o = _Item()
                    o.kind, o.value = "image", ref
                    items.append(o)
                    continue
                return [], f"400: unsupported input element type: {t!r} (supported: str / {{type:image}})"
            return [], "400: input elements must be strings or {type:image} objects"
        return items, None
    return [], "400: input must be a string or a list"


def _dim(body: dict) -> int | None:
    d = body.get("dimensions", body.get("dim"))
    if d is None:
        return None
    d = int(d)
    if d < 1 or d > 2048:
        raise ValueError("400: dimensions must be in 1..2048 (WeMM full dim is 2048)")
    return d


def handle_embeddings(raw_body: bytes) -> tuple:
    """Returns (http_status, response_body)."""
    t0 = time.time()
    try:
        body = json.loads(raw_body.decode("utf-8") or "{}")
    except Exception:
        return 400, {"error": {"message": "request body is not valid JSON", "type": "invalid_request_error"}}

    try:
        dim = _dim(body)
    except ValueError as e:
        return 400, {"error": {"message": str(e), "type": "invalid_request_error"}}

    items, err = _parse_input(body)
    if err:
        return 400, {"error": {"message": err, "type": "invalid_request_error"}}

    try:
        ml = model_layer.get()
        data = []
        for i, it in enumerate(items):
            if it.kind == "text":
                vec = ml.embed_texts([it.value], dim=dim)[0]
            else:
                vec = ml.embed_image(it.value, dim=dim)
            data.append({"object": "embedding", "index": i, "embedding": vec})
    except model_layer.BedBusyError as e:
        return 503, {"error": {"message": str(e), "type": "lm_bed_busy",
                               "code": "lm_bed_busy",
                               "retry_hint": "retry once the local model bed is free"}}
    except model_layer.NoLibraryError:
        pass  # no library does not affect the embed endpoint (it never reads the library)
    except Exception as e:
        return 500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}}

    return 200, {
        "object": "list",
        "data": data,
        "model": body.get("model") or MODEL_NAME,
        "usage": {"prompt_tokens": 0, "total_tokens": 0,
                  "note": "WeMM is a visual-language embedding model; token accounting is not converted, always 0"},
        "_timing": {"load_and_embed_s": round(time.time() - t0, 1)},
    }
