# -*- coding: utf-8 -*-
r"""WeMM-Embedding-2B model layer - single instance shared by service and MCP.

Load recipe (verified, do NOT "improve" - every value came from a real OOM/
device-mismatch post-mortem):
  AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=bf16,
                            device_map="auto",
                            max_memory={0: "4GiB", "cpu": "64GiB"})
  - key must be int 0 (accelerate 1.15 does not accept "cuda:0")
  - 4GiB is the magic cap: Windows device_map=auto miscounts shared memory,
    without the cap the whole 5.2GB model lands on the GPU and OOMs;
    overflow layers spill to CPU automatically
  - inputs must be moved to the first layer's device (first layer on GPU)
  - one item per forward pass (batching -> token-length padding crash)

Model files auto-download (huggingface_hub.snapshot_download) when the
configured model_path is missing: tencent/WeMM-Embedding-2B (weights Apache-2.0;
the repo itself does NOT ship weights). HF_ENDPOINT env var is honored, so
CN users can point at a mirror. Offline: set model_path in config.json to a
local weights folder.

Bed rule: before loading, probe the local LM Studio API; a model with
state=loaded there -> BedBusyError (no double-loading, no fighting for VRAM).
API unreachable = LM Studio not running = GPU is free.

Console prints are ASCII only (cmd/GBK).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:  # package mode (pip install / python -m medisearch)
    from . import cfg
except ImportError:  # script mode (python src/medisearch/server.py)
    import cfg

# HF cache must point at the configured location BEFORE any transformers import.
os.environ.setdefault("HF_HOME", str(cfg.HF_HOME))
os.environ.setdefault("HF_MODULES_CACHE", str(cfg.HF_MODULES_CACHE))
os.environ.setdefault("TQDM_DISABLE", "1")  # keep the 618-line weight progress bar out of logs

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


class BedBusyError(RuntimeError):
    """LM Studio has a model loaded; WeMM must not take the GPU now."""


class NoLibraryError(RuntimeError):
    """No embedding library found anywhere (local search path)."""


_state = "idle"  # idle | loading | loaded
_model = None
_processor = None
_lock = threading.RLock()
_last_use = 0.0
_events = []  # small ring of recent happenings, shown by /status


def _log(msg: str) -> None:
    """Log to STDERR: in stdio mode stdout is the protocol pipe - even a
    background-thread print (idle watchdog release) must never leak there.
    serve mode: stderr is the console log, equally fine."""
    _events.append("%s %s" % (time.strftime("%H:%M:%S"), msg))
    if len(_events) > 12:
        _events.pop(0)
    print("[medisearch] " + msg, flush=True, file=sys.stderr)


def get_events() -> list:
    return list(_events)


def get_state() -> str:
    return _state


def vram_mb() -> int:
    if _state != "loaded":
        return 0
    try:
        return int(torch.cuda.memory_allocated(0) // 1048576)
    except Exception:
        return 0


def check_lm_bed() -> None:
    """Probe LM Studio. Raises BedBusyError if a model with state=loaded is on the bed.
    API unreachable (connection refused) means LM Studio is not running -> GPU is free."""
    global _model
    try:
        req = urllib.request.Request(cfg.LM_STUDIO_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return  # LM Studio not running / API down -> GPU is ours
    models = data.get("data", data) if isinstance(data, dict) else data
    for m in models or []:
        if isinstance(m, dict) and m.get("state") == "loaded":
            name = m.get("id") or m.get("name") or "unknown"
            _log("load refused: LM Studio bed occupied by %s" % name)
            raise BedBusyError(name)


def _ensure_model_files() -> None:
    """Make sure local weights exist; auto-download from HF when missing."""
    model_dir = Path(cfg.MODEL_DIR)
    marker = model_dir / "model.safetensors"
    if marker.exists() or (model_dir / "config.json").exists():
        return
    _log("model files missing at %s -> downloading %s (~5.2GB, first run only)"
         % (model_dir, cfg.MODEL_REPO))
    from huggingface_hub import snapshot_download

    t0 = time.time()
    try:
        snapshot_download(repo_id=cfg.MODEL_REPO, local_dir=str(model_dir), max_workers=8)
    except Exception as e:
        raise RuntimeError(
            "model download failed (%s: %s). Fix: check network / HF_ENDPOINT mirror, "
            "or point 'model_path' in config.json at an existing local weights folder "
            "(offline use)." % (type(e).__name__, e))
    _log("model downloaded in %.0fs" % (time.time() - t0))


def load() -> None:
    """JIT warm-up. Callers hold the model lock (ensure_loaded/encode_*)."""
    global _model, _processor, _state, _last_use
    if _state == "loaded":
        return
    with _lock:
        if _state == "loaded":
            return
        _state = "loading"
        check_lm_bed()  # may raise BedBusyError -> caller surfaces 503
        _ensure_model_files()
        _log("loading model (first use, warm-up ~10-60s)...")
        t0 = time.time()
        _processor = AutoProcessor.from_pretrained(cfg.MODEL_DIR, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            cfg.MODEL_DIR,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=dict(cfg.MODEL_MAX_MEMORY),
        ).eval()
        _state = "loaded"
        _last_use = time.time()
        _log("model loaded in %.1fs, %d tensors"
             % (time.time() - t0, sum(1 for _ in _model.state_dict())))


def release() -> None:
    """Drop the model, return VRAM+RAM. Safe when idle."""
    global _model, _processor, _state
    with _lock:
        if _state != "loaded":
            return
        del _model
        del _processor
        _model = None
        _processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _state = "idle"
        _log("model released (idle timeout)")


def ensure_loaded() -> None:
    """JIT: load on first use. Blocks until loaded. Raises BedBusyError if refused."""
    with _lock:
        load()


def _encode_messages(messages, dim=None):
    """Embed ONE sample (no batching, see module docstring). Returns L2-normalized
    (D,) tensor, optionally truncated+re-normalized to dim."""
    global _last_use
    with _lock:
        load()
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None
        inputs = _processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            **video_kwargs,
        )
        # first layer lives on GPU; CPU-resident inputs crash with device mismatch
        first_dev = next(_model.parameters()).device
        inputs = inputs.to(first_dev)
        with torch.inference_mode():
            emb = _model.embedding(**inputs).float()
        if dim:
            emb = F.normalize(emb[..., :dim], dim=-1)
        out = emb.cpu()
        _last_use = time.time()
    return out[0] if out.dim() > 1 else out


def msg_text(s: str):
    return [{"role": "user", "content": [{"type": "text", "text": s}]}]


def msg_image(p) -> list:
    return [{"role": "user", "content": [{"type": "image", "image": str(p)}]}]


def msg_video(p) -> list:
    return [{"role": "user", "content": [{"type": "video", "video": str(p)}]}]


# ---------------------------------------------------------------- idle watchdog
_watchdog_started = False


def start_watchdog() -> None:
    """Tick every 20s; release the model after settings.idle_release_min of no use
    (0 disables). Don't hog the owner's LM Studio bed."""
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    threading.Thread(target=_watchdog_tick, daemon=True, name="medisearch-idle-watchdog").start()


def _watchdog_tick() -> None:
    while True:
        time.sleep(20)
        try:
            try:
                from .libstore import load_settings
            except ImportError:
                from libstore import load_settings
            idle_min = float(load_settings().get("idle_release_min", 30))
        except Exception:
            idle_min = 30.0
        if idle_min <= 0:
            continue
        with _lock:
            if _state != "loaded":
                continue
            if time.time() - _last_use >= idle_min * 60:
                release()


# ---------------------------------------------------------------- facade (MCP side)
class ModelLayerFacade:
    """Object-style API used by the MCP backend / OpenAI-compat endpoint
    (mirrors the pre-merge LocalModel interface, now backed by the same
    process-wide singleton)."""

    def __init__(self) -> None:
        self._bed_occupied_at: str | None = None

    @property
    def loaded(self) -> bool:
        return _state == "loaded"

    def model_state(self) -> dict:
        if _state != "loaded":
            return {"model": "idle", "vram_mb": 0}
        try:
            vram = int(torch.cuda.memory_allocated(0) // 1048576) if torch.cuda.is_available() else 0
        except Exception:
            vram = -1
        return {"model": "loaded", "vram_mb": vram}

    def ensure_loaded(self) -> None:
        ensure_loaded()

    def release(self) -> None:
        release()

    def embed_texts(self, texts: list, dim: int | None = None) -> list:
        outs = []
        for t in texts:  # one item at a time (batching crash)
            outs.append(_encode_messages(msg_text(t), dim=dim).tolist())
        return outs

    def embed_image(self, image_ref: str, dim: int | None = None) -> list:
        ref = image_ref
        if image_ref.startswith("data:") or (
                len(image_ref) > 64 and "/" not in image_ref and ":" not in image_ref[:8]):
            # base64 (bare or data-URL) -> temp file (qwen_vl_utils likes paths most)
            import base64
            import tempfile
            raw = image_ref.split(",", 1)[1] if image_ref.startswith("data:") else image_ref
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(base64.b64decode(raw))
                tmp = tf.name
            try:
                return _encode_messages(msg_image(tmp), dim=dim).tolist()
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return _encode_messages(msg_image(ref), dim=dim).tolist()

    def embed_video(self, path: str, dim: int | None = None) -> list:
        return _encode_messages(msg_video(path), dim=dim).tolist()


_facade = ModelLayerFacade()


def get() -> ModelLayerFacade:
    return _facade
