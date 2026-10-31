# -*- coding: utf-8 -*-
"""MediaSearch central config - machine-free by design.

Resolution order (highest wins), per key:
  1. environment variables (WEMM_*)
  2. <data_dir>/config.json   (user file; copy config.json.example into place)
  3. built-in portable defaults

Built-in defaults NEVER contain absolute paths of any particular machine:
  - data dir : <repo>/data          when running from a source checkout
               ~/.medisearch/data   when installed into site-packages
  - models   : ~/.medisearch/models/WeMM-Embedding-2B
               (auto-downloaded from tencent/WeMM-Embedding-2B on first use;
                offline: point model_path at your local weights folder)
  - HF cache : ~/.medisearch/hf (HF_HOME) + .../hf/modules (HF_MODULES_CACHE)
               - override with WEMM_HF_HOME if your security software blocks
                 the default location.

Console output stays ASCII (cmd/GBK on Windows).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------- discovery


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path | None:
    """Repo checkout root when running from source / editable install,
    None when the package sits in site-packages."""
    p = _package_dir().parent            # .../src/medisearch -> .../src
    cand = p.parent                      # .../src -> repo root
    if (cand / "pyproject.toml").exists() and (cand / "src" / "medisearch" / "__init__.py").exists():
        return cand
    # flat editable install (repo/medisearch/...):
    cand2 = _package_dir().parent
    if (cand2 / "pyproject.toml").exists() and cand2.name != "site-packages":
        return cand2
    return None


def _home_root() -> Path:
    return Path(os.environ.get("WEMM_HOME") or (Path.home() / ".medisearch")).expanduser()


_REPO_ROOT = _repo_root()


def _user_config() -> dict:
    """Read <data_dir>/config.json (or $WEMM_CONFIG). Missing/invalid -> {}."""
    data_dir_default = _repo_root() / "data" if _repo_root() else _home_root() / "data"
    path = os.environ.get("WEMM_CONFIG")
    cand = Path(path).expanduser() if path else data_dir_default / "config.json"
    try:
        raw = cand.read_text(encoding="utf-8")
        val = json.loads(raw) if raw.strip() else {}
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def _pick(key: str, env: str, default, cast=None):
    """env > config.json > default."""
    env_val = os.environ.get(env)
    if env_val not in (None, ""):
        return cast(env_val) if cast else env_val
    val = _user_config().get(key)
    if val is not None:
        return cast(val) if cast else val
    return default


def _opt_path(val):
    if val in (None, ""):
        return None
    return Path(str(val)).expanduser()


# ---------------------------------------------------------------- resolved values

# --- data dir (embeddings.json / scopes.json / settings.json live here) ---
DATA: Path = (
    _opt_path(os.environ.get("WEMM_DATA_DIR"))
    or _opt_path(_user_config().get("data_dir"))
    or (_repo_root() / "data" if _repo_root() else _home_root() / "data")
)

EMBED_NEW = DATA / "embeddings.json"
SCOPES_FILE = DATA / "scopes.json"
SETTINGS_FILE = DATA / "settings.json"
CONFIG_FILE = (
    Path(os.environ["WEMM_CONFIG"]).expanduser()
    if os.environ.get("WEMM_CONFIG")
    else DATA / "config.json"
)

# --- model (weights + HF auto-download) ---
MODEL_REPO = _pick("model_repo", "WEMM_MODEL_REPO", "tencent/WeMM-Embedding-2B")
MODELS_DIR = (
    _opt_path(os.environ.get("WEMM_MODELS_DIR"))
    or _opt_path(_user_config().get("models_dir"))
    or _home_root() / "models"
)
MODEL_DIR = (
    _opt_path(os.environ.get("WEMM_MODEL_PATH"))
    or _opt_path(_user_config().get("model_path"))
    or MODELS_DIR / "WeMM-Embedding-2B"
)

# Load recipe: magic values verified on the acceptance machine - do NOT "improve":
# key must be int 0 (accelerate 1.15); 4GiB cap forces overflow layers to CPU
# (Windows miscounts shared memory; without it the whole 5.2GB model lands on
# the GPU and OOMs).
def _normalize_max_memory(val) -> dict:
    """JSON object keys are always strings ("0"), but accelerate 1.15's
    max_memory only accepts int device keys - int-ify on load, so a config
    written from config.json.example (string "0" key) loads cleanly."""
    if not val:
        return {0: "4GiB", "cpu": "64GiB"}
    out = {}
    for k, v in dict(val).items():
        try:
            k = int(k)
        except (TypeError, ValueError):
            pass  # non-numeric keys (e.g. "cpu") stay as-is
        out[k] = v
    return out


MODEL_MAX_MEMORY = _normalize_max_memory(_user_config().get("model_max_memory"))
MODEL_FULL_DIM = 2048  # stored vectors are always full-dim; truncation at search time

# --- HF cache (some security software blocks C: writes; keep it configurable) ---
HF_HOME = (
    _opt_path(os.environ.get("WEMM_HF_HOME"))
    or _opt_path(_user_config().get("hf_home"))
    or _home_root() / "hf"
)
HF_MODULES_CACHE = (
    _opt_path(os.environ.get("WEMM_HF_MODULES_CACHE"))
    or _opt_path(_user_config().get("hf_modules_cache"))
    or HF_HOME / "modules"
)

# Optional second library used when the primary one is empty (lib source "fallback").
FALLBACK_LIB = _opt_path(os.environ.get("WEMM_FALLBACK_LIB")) or _opt_path(
    _user_config().get("fallback_library"))

# --- service / probes ---
HOST = _pick("host", "WEMM_HOST", "127.0.0.1")
PORT = int(_pick("port", "WEMM_PORT", 8686, cast=int))
API_BASE = _pick("api_base", "WEMM_API_BASE", "http://127.0.0.1:8686").rstrip("/")
LM_STUDIO_URL = _pick("lm_bed_api", "WEMM_LM_BED_API", "http://127.0.0.1:1234/api/v0/models")

STATIC = _package_dir() / "static"

# --- file types that get embedded ---
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VID_EXTS = {".mp4", ".mkv", ".webm", ".mov"}

# Checkpoint: persist library every N embedded files during a build (atomic replace)
BUILD_CHECKPOINT = 5

# ---------------------------------------------------------------- defaults

DEFAULT_SETTINGS = {
    "topn": 5,               # results per search
    "dim": 2048,             # vector dim used for scoring (2048/1024/512/256)
    "min_score": 0.0,        # drop results below this cosine score
    "idle_release_min": 30,  # auto-release model after N idle minutes
}

# Seeded into data/scopes.json on first start. Repo checkout: the bundled
# samples folder (placeholder until assets ship); installed package: no seed.
if _repo_root():
    DEFAULT_SCOPES = [{
        "id": "samples",
        "path": str(_repo_root() / "samples"),
        "name": "MediaSearch samples",
        "enabled": True,
        "scanned_at": None,
        "count": 0,
    }]
else:
    DEFAULT_SCOPES = []

# ---------------------------------------------------------------- runtime helpers

_READ_ONLY = os.environ.get("WEMM_READ_ONLY", "").strip().lower() in ("1", "true", "yes", "on")


def is_read_only() -> bool:
    """Write refusal: set via env WEMM_READ_ONLY=1 or the CLI --read-only flag.
    Mutating endpoints (add/PUT scopes, remove-scope, embed-scope, PUT settings)
    refuse with 403; search/status keep working."""
    return _READ_ONLY or os.environ.get("WEMM_READ_ONLY", "").strip().lower() in ("1", "true", "yes", "on")


def set_read_only(on: bool) -> None:
    global _READ_ONLY
    _READ_ONLY = bool(on)


def pin_tmp() -> None:
    """Pin Python/C-level temp dirs into <data>/tmp BEFORE any torch import.

    Host sandboxes can make the system Temp unwritable (torch/tokenizers JIT
    precompile then dies with 'No usable temporary directory'); pinning into
    the project's own data dir makes any host environment work. No-op when the
    directory cannot be created (read-only media)."""
    try:
        tmp = DATA / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        import tempfile
        tempfile.tempdir = str(tmp)
        for k in ("TEMP", "TMP", "TMPDIR"):
            os.environ[k] = str(tmp)
    except Exception:
        pass
