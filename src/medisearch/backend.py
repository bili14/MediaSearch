# -*- coding: utf-8 -*-
"""Tool backends - the single branch point between in-process and HTTP execution.

Three modes (one per deployment shape, picked at startup):
  direct  the /mcp + /v1/embeddings faces of 'medisearch serve' run in the
          SAME process as the REST API: tools execute in-process against the
          shared library + model layer (no HTTP self-probe, no second process)
  api     stdio process ('medisearch mcp') talking to a running service:
          WEMM_API_BASE (default http://127.0.0.1:8686) pass-through
  local   stdio process with no service: read the library files directly and
          embed with the local model layer

Selection:
  serve process       -> init_direct()
  WEMM_BACKEND=auto   -> probe {API_BASE}/status: alive=api, else=local
  WEMM_BACKEND=api    -> forced api (probe errors are surfaced, not swallowed)
  WEMM_BACKEND=local  -> forced local

Library read order (local mode; the first existing file wins):
  1. $WEMM_DATA  (explicit library file)
  2. <data_dir>/embeddings.json
  3. 'fallback_library' from config.json / $WEMM_FALLBACK_LIB

/v1/embeddings (openai_compat) always uses the local model layer - the REST API
has no text->vector endpoint; only the model layer can produce vectors.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

try:  # package mode
    from . import cfg
except ImportError:  # script mode
    import cfg

API_BASE = os.environ.get("WEMM_API_BASE", "http://127.0.0.1:8686").rstrip("/")
API_TIMEOUT = float(os.environ.get("WEMM_API_TIMEOUT", "300"))  # first search JIT-loads the model
PROBE_TIMEOUT = float(os.environ.get("WEMM_PROBE_TIMEOUT", "2.0"))

# Explicit: a mode set by init_direct() / CLI; otherwise WEMM_BACKEND at call time
_MODE = None  # "direct" | None (resolve per call for stdio)

_lib_cache: dict = {}
_lib_lock = threading.Lock()

_last_probe_err: str | None = None


class NoLibrary(RuntimeError):
    pass


# ═══════════════ mode selection ═══════════════

def init_direct() -> None:
    """Called by 'medisearch serve': all tool execution happens in-process."""
    global _MODE
    _MODE = "direct"


def resolve_mode() -> str:
    if _MODE == "direct":
        return "direct"
    mode = os.environ.get("WEMM_BACKEND", "auto").strip().lower()
    if mode in ("api", "local"):
        return mode
    if _api_alive():
        return "api"
    return "local"


# Explicit no-proxy: host sandboxes may inject HTTP_PROXY into spawned
# children; loopback probes must always connect directly.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _api_alive() -> bool:
    global _last_probe_err
    req = urllib.request.Request(f"{API_BASE}/status", method="GET")
    try:
        with _OPENER.open(req, timeout=PROBE_TIMEOUT) as r:
            if getattr(r, "status", 200) != 200:
                _last_probe_err = f"HTTP {r.status}"
                return False
        _last_probe_err = None
        return True
    except Exception as e:
        _last_probe_err = f"{type(e).__name__}: {e}"
        return False


def _probe_note() -> str:
    mode = os.environ.get("WEMM_BACKEND", "auto").strip().lower()
    if mode == "local":
        return "local mode (WEMM_BACKEND=local forced): direct read of the local library"
    return (f"local mode: direct read of the local library (probe {API_BASE}/status failed: "
            f"{_last_probe_err or 'unknown'}; widen with WEMM_PROBE_TIMEOUT)")


def api_unavailable_note() -> str:
    return f"service unreachable ({API_BASE}/status, connection refused/timeout)"


# ═══════════════ api: HTTP pass-through ═══════════════

def _api_post(path: str, body: dict, timeout: float | None = None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with _OPENER.open(req, timeout=timeout or API_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _api_search(q: str, scope: str | None, topn: int) -> list:
    resp = _api_post("/search", {"q": q, "scope": scope or "all", "topn": topn})
    # service contract: object {q,scope,dim,ms,lib,results:[...]} (bare list tolerated)
    if isinstance(resp, dict):
        inner = resp.get("results")
        if isinstance(inner, list):
            return inner
        raise RuntimeError(f"/search response missing results list (keys={sorted(resp)})")
    if isinstance(resp, list):
        return resp
    raise RuntimeError(f"/search returned wrong type ({type(resp).__name__})")


def _api_status() -> dict:
    req = urllib.request.Request(f"{API_BASE}/status", method="GET")
    with _OPENER.open(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


# ═══════════════ local: direct library read ═══════════════

def library_path() -> str | None:
    for p in [os.environ.get("WEMM_DATA"), str(cfg.EMBED_NEW)] +              ([str(cfg.FALLBACK_LIB)] if cfg.FALLBACK_LIB is not None else []):
        if p and os.path.exists(p):
            return p
    return None


def load_library_local() -> tuple:
    p = library_path()
    if p is None:
        raise NoLibrary()
    with _lib_lock:
        st = os.stat(p)
        hit = _lib_cache.get(p)
        if hit and hit[0] == st.st_mtime:
            return p, hit[1]
        store = json.loads(open(p, encoding="utf-8").read())
        _lib_cache[p] = (st.st_mtime, store)
        return p, store


def _local_search(q: str, scope: str | None, topn: int) -> tuple:
    import numpy as np

    lib, store = load_library_local()
    items = [(p, e) for p, e in store.items()]
    if scope not in (None, "", "all"):
        items = [(p, e) for p, e in items if str(e.get("scope", "media")) == scope]
    if not items:
        return lib, []

    # mixed-dim libraries: group by the majority dim (insurance, not expected)
    from collections import Counter
    dim = Counter(str(e.get("dim", 2048)) for _, e in items).most_common(1)[0][0]
    dim_i = int(dim)
    items = [(p, e) for p, e in items if int(e.get("dim", 2048)) == dim_i]

    try:
        from . import model_layer
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import model_layer
    vecs = model_layer.get().embed_texts([q], dim=dim_i)
    qv = np.asarray(vecs[0], dtype=np.float32)
    V = np.array([np.asarray(e["vec"], dtype=np.float32) for _, e in items])
    cos = (V @ qv) / (np.linalg.norm(V, axis=1) * (np.linalg.norm(qv) + 1e-12))
    order = np.argsort(-cos)[:topn]
    results = []
    for i in order:
        p, e = items[i]
        results.append({
            "path": p,
            "score": round(float(cos[i]), 4),
            "modality": e.get("modality", "unknown"),
            "scope": e.get("scope", "media"),
        })
    return lib, results


# ═══════════════ direct: in-process (serve face 2) ═══════════════

def _direct_search(q: str, scope: str | None, topn: int) -> list:
    try:
        from . import libstore
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import libstore
    s = libstore.load_settings()
    topn = max(1, min(int(topn), 50)) or s["topn"]
    dim = int(s["dim"])
    min_score = s["min_score"]
    scope = scope or "all"
    store, _src = libstore.load_library()
    if not store:
        return []
    try:
        from . import model_layer
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import model_layer
    qvec = model_layer.get().embed_texts([q], dim=dim)[0]
    return libstore.search(store, qvec, scope, None if scope == "all" else scope,
                           dim, topn, min_score)


def _direct_status() -> dict:
    try:
        from . import libstore, model_layer
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import libstore, model_layer
    store, source = libstore.load_library()
    s = libstore.load_settings()
    return {
        "backend": "direct",
        "model": model_layer.get_state(),
        "vram_mb": model_layer.vram_mb(),
        "items": len(store),
        "lib": source,
        "settings": s,
        "building": libstore.build_status(),
        "events": model_layer.get_events(),
        "read_only": cfg.is_read_only(),
    }


# ═══════════════ tool entries (mcp_proto calls only these) ═══════════════

def media_search(q: str, scope: str | None = None, topn: int = 5) -> dict:
    topn = max(1, min(int(topn), 50))
    mode = resolve_mode()
    if mode == "direct":
        results = _direct_search(q, scope, topn)
        return {"backend": "direct", "query": q, "scope": scope or "all",
                "results": results}
    if mode == "api":
        results = _api_search(q, scope, topn)
        return {"backend": "api", "api": API_BASE, "query": q, "results": results}
    try:
        lib, results = _local_search(q, scope, topn)
    except NoLibrary:
        return {"backend": "local", "query": q, "results": [],
                "error": "no embedding library found ($WEMM_DATA / data/embeddings.json / "
                          "fallback_library all missing)"}
    out = {"backend": "local", "library": lib, "query": q,
           "scope": scope or "all", "topn": topn, "results": results}
    if _last_probe_err:
        out["probe_fail"] = f"{API_BASE}/status -> {_last_probe_err}"
    if not results:
        out["note"] = f"0 items in scope (scope={scope or 'all'} has no matching media)"
    return out


def media_status() -> dict:
    mode = resolve_mode()
    if mode == "direct":
        return _direct_status()
    if mode == "api":
        try:
            st = _api_status()
            st["backend"] = "api"
            st["api"] = API_BASE
            return st
        except Exception as e:
            return {"backend": "api", "api": API_BASE,
                    "error": f"{api_unavailable_note()}: {e}"}
    try:
        from . import model_layer
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import model_layer
    lib = library_path()
    items, dim = 0, 2048
    if lib:
        try:
            _, store = load_library_local()
            items = len(store)
            if store:
                dim = int(next(iter(store.values())).get("dim", 2048))
        except Exception:
            items = -1
    ml = model_layer.get()
    out = {
        "backend": "local",
        "library": lib,
        "items": items,
        "dim": dim,
        **ml.model_state(),
        "note": _probe_note(),
    }
    return out


def media_build(scope=None) -> dict:
    """Incremental build of registered scopes (new/changed files by mtime).
    Available in direct mode (in-process) and api mode (service online).
    local (standalone) mode has no build - it says so plainly."""
    mode = resolve_mode()
    if mode == "direct":
        try:
            from . import libstore
        except ImportError as _e:
            import sys as _sys
            print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
            import libstore
        try:
            info = libstore.start_build(scope)
            return {"backend": "direct", **info}
        except libstore.ReadOnlyRefused as e:
            return {"backend": "direct", "error": str(e)}
        except KeyError as e:
            return {"backend": "direct", "error": f"unknown scope {e.args[0] if e.args else e}"}
    if mode == "api":
        body: dict = {}
        if scope not in (None, "", []):
            body["scope"] = scope
        resp = _api_post("/embed-scope", body)
        if not isinstance(resp, dict):
            return {"backend": "api", "api": API_BASE,
                    "error": f"/embed-scope returned non-object ({type(resp).__name__})"}
        return {"backend": "api", "api": API_BASE, **resp}
    return {
        "backend": "local",
        "error": ("media_build needs a running service (or direct mode): "
                  "start 'medisearch serve' and retry with WEMM_BACKEND=auto/api, "
                  "or call it through the serve process"),
    }


def media_remove_scope(scope_id: str, delete_vectors: bool = True) -> dict:
    """Unregister a scope. Safety contract: NEVER deletes files on disk
    (files_deleted is always 0); delete_vectors removes only the vector records."""
    mode = resolve_mode()
    if mode == "direct":
        try:
            from . import libstore
        except ImportError as _e:
            import sys as _sys
            print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
            import libstore
        try:
            res = libstore.remove_scope(scope_id, delete_vectors)
            return {"backend": "direct", **res}
        except KeyError as e:
            return {"backend": "direct", "error": f"unknown scope {e.args[0] if e.args else e}"}
        except libstore.ReadOnlyRefused as e:
            return {"backend": "direct", "error": str(e)}
    if mode == "api":
        resp = _api_post("/remove-scope", {"id": scope_id, "delete_vectors": bool(delete_vectors)})
        if not isinstance(resp, dict):
            return {"backend": "api", "api": API_BASE,
                    "error": f"/remove-scope returned non-object ({type(resp).__name__})"}
        out = {"backend": "api", "api": API_BASE, **resp}
        out.setdefault("files_deleted", 0)
        out.setdefault("note", "registry + vector records only; source files on disk are never deleted")
        return out
    return {
        "backend": "local",
        "error": ("media_remove_scope needs a running service (or direct mode): "
                  "start 'medisearch serve' and retry with WEMM_BACKEND=auto/api "
                  "(it only removes the registry row + vector records, never files on disk)"),
    }
