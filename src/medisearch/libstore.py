# -*- coding: utf-8 -*-
"""Library store: data/embeddings.json (primary) with an optional fallback
library (config "fallback_library" / WEMM_FALLBACK_LIB), plus scopes.json /
settings.json and the incremental build worker.

Data schema (team standard):
  data/embeddings.json  { "<abs_path>": {"modality","dim","vec","scope","mtime"} }
  data/scopes.json      [ {id, path, name, enabled, scanned_at, count} ]
  data/settings.json    { topn, dim, min_score, idle_release_min }

Notes:
  * old-library items without scope/mtime tags get scope derived by folder
    prefix, mtime read from disk
  * builds take an advisory lockfile + atomic replace so a crashed build never
    leaves a half-written library (readers see either the old or the new file)
  * builds never re-embed files whose mtime is unchanged (incremental)
  * read-only mode (cfg.is_read_only): every mutating operation refuses with
    ReadOnlyRefused (the service maps it to HTTP 403)
"""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

try:  # package mode
    from . import cfg, model_layer
except ImportError:  # script mode
    import cfg, model_layer

# ---------------------------------------------------------------- portable lock
try:
    import msvcrt  # Windows

    # In-process front gate: two threads of THIS process must never issue
    # msvcrt.locking on the same 1-byte range at the same time (same-region
    # re-lock within one process raises OSError/EDEADLK, verified by a
    # two-handle C-style comparison). The guard serializes our threads;
    # cross-process contention is settled by the non-blocking attempts +
    # backoff below. POSIX (flock) needs neither and is left untouched.
    _MSVCRT_GUARD = threading.Lock()
    _LOCK_RETRY_SLEEP_S = 0.2
    _LOCK_MAX_ATTEMPTS = 25

    def _lock(fh):
        # msvcrt.locking(..., 0) locks ZERO bytes = a no-op (two concurrent
        # 0-byte LK_LOCKs both succeed - the lock was an illusion, and two
        # services on the same data dir could lose vectors last-writer-wins).
        # Lock a real 1-byte range: seed the empty .lock file with 1 byte so
        # the range exists, then take it NON-blocking with backoff, so a
        # held lock (same process or another process) can never surface as
        # a raw Errno 36/OSError: contenders sleep 0.2s x 25 and then raise
        # a clear RuntimeError instead. Nothing is written on failure.
        # Seed an empty .lock file with 1 byte via an APPEND write
        # (never read the locked range: while the holder locks byte 0,
        # even fh.read(1) raises EACCES on Windows). A cross-process
        # race may make it 2 bytes; byte 0 stays the shared target.
        try:
            if os.path.getsize(fh.name) == 0:
                fh.write(b"\0")
                fh.flush()
        except OSError:
            pass
        try:
            fh.seek(0)
        except OSError as e:
            raise RuntimeError(
                "could not seek the library lock file: %s" % e) from e
        last = None
        for i in range(_LOCK_MAX_ATTEMPTS):
            # The guard is held ONLY for the msvcrt call: while a contender
            # sleeps, the in-process holder must still be able to _unlock
            # (a guard held across the whole backoff window would starve the
            # releaser and the contender would time out for no reason).
            with _MSVCRT_GUARD:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError as ee:
                    last = ee
            time.sleep(_LOCK_RETRY_SLEEP_S)
        raise RuntimeError(
                "could not acquire the library lock on %s after ~%ds "
                "(a service or thread holds it; last error: %s). "
                "Retry a moment later - nothing was written." % (
                    getattr(fh, "name", "?"),
                    _LOCK_RETRY_SLEEP_S * _LOCK_MAX_ATTEMPTS, last))

    def _unlock(fh):
        with _MSVCRT_GUARD:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

except ImportError:
    try:
        import fcntl  # POSIX

        def _lock(fh):
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        def _unlock(fh):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

    except ImportError:
        def _lock(fh):  # best effort: no advisory locking available
            return

        def _unlock(fh):
            return


class ReadOnlyRefused(RuntimeError):
    """Write attempted while read-only mode (WEMM_READ_ONLY=1 / --read-only)."""


def _require_writable() -> None:
    if cfg.is_read_only():
        raise ReadOnlyRefused(
            "read-only mode (WEMM_READ_ONLY=1): this operation writes to the library. "
            "Restart without read-only to allow writes.")


# ---------------------------------------------------------------- settings
def load_settings() -> dict:
    _ensure_data_dir()
    s = dict(cfg.DEFAULT_SETTINGS)
    if cfg.SETTINGS_FILE.exists():
        try:
            s.update(json.loads(cfg.SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(patch: dict) -> dict:
    _require_writable()
    s = load_settings()
    s.update(patch)
    _ensure_data_dir()
    _atomic_write(cfg.SETTINGS_FILE, json.dumps(s, ensure_ascii=False, indent=2))
    return s


# ---------------------------------------------------------------- scopes
def _ensure_data_dir() -> None:
    cfg.DATA.mkdir(parents=True, exist_ok=True)


def load_scopes() -> list:
    _ensure_data_dir()
    if not cfg.SCOPES_FILE.exists():
        if cfg.DEFAULT_SCOPES:
            save_scopes(cfg.DEFAULT_SCOPES)
        else:
            save_scopes([])
        return [dict(r) for r in cfg.DEFAULT_SCOPES]
    try:
        return json.loads(cfg.SCOPES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return [dict(r) for r in cfg.DEFAULT_SCOPES]


def save_scopes(rows: list) -> list:
    _atomic_write(cfg.SCOPES_FILE, json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def add_scope(path: str, name: str | None = None) -> dict:
    _require_writable()
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise ValueError("folder does not exist: %s" % path)
    rows = load_scopes()
    dups = [r for r in rows if Path(r["path"]).expanduser().resolve() == p]
    if dups:
        raise ValueError("scope already registered: %s" % dups[0]["id"])
    # reject nested scopes (dedupe of overlapping files is out of scope)
    for r in rows:
        rp = Path(r["path"]).expanduser().resolve()
        if rp == p or rp in p.parents or p in rp.parents:
            raise ValueError("refused: nested inside existing scope %s (overlapping dedupe not built yet)" % r["id"])
    base = Path(name or p.name).name.strip() or p.name
    sid, i = base, 2
    while any(r["id"] == sid for r in rows):
        sid = "%s-%d" % (base, i)
        i += 1
    count = count_files(p)
    row = {"id": sid, "path": str(p), "name": name or base,
           "enabled": True, "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"), "count": count}
    rows.append(row)
    save_scopes(rows)
    _lib_cache.clear()
    return row


def set_scope_enabled(sid: str, enabled: bool) -> dict:
    _require_writable()
    rows = load_scopes()
    for r in rows:
        if r["id"] == sid:
            r["enabled"] = bool(enabled)
            save_scopes(rows)
            return r
    raise KeyError("unknown scope %s" % sid)


def _write_library_replace(store: dict) -> None:
    """Full-replace write (remove-scope only). Locked + atomic.
    Distinct from _checkpoint_store (incremental merge, never drops keys)."""
    _require_writable()
    lk = open(cfg.EMBED_NEW.with_suffix(".lock"), "a+b")
    try:
        _lock(lk)
        _atomic_write(cfg.EMBED_NEW, json.dumps(store, ensure_ascii=False))
    finally:
        _unlock(lk)
        lk.close()
    _lib_cache.clear()
    invalidate_matrix()


def remove_scope(sid: str, delete_vectors: bool = True) -> dict:
    """Remove a registered scope; delete_vectors=True also prunes its vectors.

    Safety contract: only the two data files are written
    (scopes.json / embeddings.json); files on disk are NEVER deleted
    (files_deleted is always 0).
    """
    _require_writable()
    rows = load_scopes()
    hit = [r for r in rows if r["id"] == sid]
    if not hit:
        raise KeyError("unknown scope %s" % sid)
    scope_path = str(Path(hit[0]["path"]).expanduser().resolve()).lower() + os.sep
    rows = [r for r in rows if r["id"] != sid]
    save_scopes(rows)
    removed = 0
    if delete_vectors:
        lib, _src = load_library()
        pruned = {k: v for k, v in lib.items()
                  if item_scope(v, k) != sid and not k.lower().startswith(scope_path)}
        removed = len(lib) - len(pruned)
        _write_library_replace(pruned)
    else:
        _lib_cache.clear()
        invalidate_matrix()
    return {"id": sid, "registry_removed": True, "vectors_removed": removed,
            "files_deleted": 0, "note": "source files untouched"}


def count_files(folder: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in (cfg.IMG_EXTS | cfg.VID_EXTS):
                n += 1
    return n


def scan_files(folder: Path) -> list:
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in (cfg.IMG_EXTS | cfg.VID_EXTS):
                out.append(Path(root) / f)
    return out


# ---------------------------------------------------------------- library
_lib_cache: dict = {}  # sig -> {"s": store, "n": has_items}


def _candidate_libraries() -> list:
    cands = [(cfg.EMBED_NEW, "new")]
    if cfg.FALLBACK_LIB is not None:
        cands.append((cfg.FALLBACK_LIB, "fallback"))
    return cands


def load_library() -> tuple:
    """(store, source)  source in {"new","fallback","none"}.
    Primary file wins when it exists and holds items; else the optional
    fallback library."""
    global _lib_cache
    for cand, tag in _candidate_libraries():
        if cand.exists() and cand.stat().st_size > 2:
            sig = (cand.stat().st_mtime_ns, cand.stat().st_size)
            hit = _lib_cache.get(sig)
            if hit is not None:
                store = hit["s"]
                if hit.get("n"):
                    return store, tag
            try:
                store = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                _lib_cache[sig] = {"s": {}, "n": False}
                continue
            if isinstance(store, dict) and store:
                _lib_cache[sig] = {"s": store, "n": True}
                return _normalize(store), tag
            _lib_cache[sig] = {"s": {}, "n": False}
    return {}, "none"


def _normalize(store: dict) -> dict:
    """Fill missing scope/mtime tags on legacy-library items (in place, cheap)."""
    scopes = load_scopes()
    for key, rec in store.items():
        if "scope" not in rec:
            rec["scope"] = derive_scope(key, scopes)
        if "mtime" not in rec:
            try:
                rec["mtime"] = int(Path(key).stat().st_mtime)
            except OSError:
                rec["mtime"] = 0
    return store


def derive_scope(abs_path: str, scopes: list) -> str:
    for r in scopes:
        rp = str(Path(r["path"]).expanduser().resolve()).lower()
        if abs_path.lower().startswith(rp + os.sep):
            return r["id"]
    return "unknown"


def item_scope(rec: dict, abs_path: str) -> str:
    return rec.get("scope") or "unknown"


# ---------------------------------------------------------------- matrix (numpy, cached per dim)
_mat_cache: dict = {}


def invalidate_matrix() -> None:
    _mat_cache.clear()


def matrix(dim: int, store: dict):
    """(V normalized (N,D), keys, item_scopes). Recomputed only when library changed."""
    if dim not in _mat_cache:
        _mat_cache[dim] = None
    hit = _mat_cache[dim]
    if hit is not None and hit[2] is store:  # same store object -> same contents
        return hit[0], hit[1], hit[3]
    keys = list(store)
    scopes = [item_scope(store[k], k) for k in keys]
    D = min(dim, cfg.MODEL_FULL_DIM)
    V = np.empty((len(keys), D), dtype=np.float32)
    for i, k in enumerate(keys):
        vec = np.asarray(store[k]["vec"][:D], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        V[i] = vec / n if n > 0 else 0.0
    _mat_cache[dim] = (V, keys, store, scopes)
    return V, keys, scopes


def search(store: dict, qvec, scope, scopes_list, dim: int, topn: int, min_score: float) -> list:
    D = min(dim, cfg.MODEL_FULL_DIM)
    qv = np.asarray(qvec, dtype=np.float32)[:D]
    qn = float(np.linalg.norm(qv))
    qv = qv / qn if qn > 0 else qv
    V, keys, iscp = matrix(D, store)
    scores = V @ qv
    sel = []
    if scope == "all" or not scopes_list:
        sel = [i for i in range(len(keys)) if scores[i] >= min_score]
        sel.sort(key=lambda i: -scores[i])
    else:
        # scope may arrive as a single id string ("samples") or a list
        # (["samples", "downloads"]): set(string) would split it into a
        # character set and the filter would match nothing (audit S23).
        want = set(scopes_list) if isinstance(scopes_list, (list, tuple)) else {scopes_list}
        sel = [i for i in range(len(keys)) if iscp[i] in want and scores[i] >= min_score]
        sel.sort(key=lambda i: -scores[i])
    out = []
    for i in sel[:topn]:
        rec = store[keys[i]]
        out.append({
            "path": keys[i],
            "score": round(float(scores[i]), 4),
            "modality": rec.get("modality", "?"),
            "scope": iscp[i],
        })
    return out


# ---------------------------------------------------------------- incremental build
BUILD = {"state": "none", "scope": None, "done": 0, "total": 0, "cur": None, "err": None, "started": None, "finished": None}
_build_lock = threading.Lock()


def build_status() -> dict:
    return dict(BUILD)


def start_build(scope=None) -> dict:
    """scope: id, "all", or None (= all enabled scopes). Rejects if already building."""
    if cfg.is_read_only():
        raise ReadOnlyRefused(
            "read-only mode (WEMM_READ_ONLY=1): library build writes to the library. "
            "Restart without read-only to allow builds.")
    with _build_lock:
        if BUILD["state"] in ("running", "queued"):
            # queued included: the window between 'queued' and the worker thread
            # flipping to 'running' must also reject a second start_build
            # (UI double-click / agent retry -> two concurrent builds).
            raise RuntimeError("build already running (%d/%d)" % (BUILD["done"], BUILD["total"]))
        target = _target_scopes(scope)
        if not target:
            raise RuntimeError("no enabled scope to build (add a folder first)")
    rows = {r["id"]: r for r in load_scopes()}
    store, _src = load_library()
    total_pending = 0
    for sid in target:
        total_pending += len(pending_files(Path(rows[sid]["path"]), sid, store)[0])
    BUILD.update(state="queued", scope=scope or "all", done=0, total=total_pending,
                 cur=None, err=None, started=time.strftime("%H:%M:%S"))
    t = threading.Thread(target=_build_job, args=(target, rows), daemon=True, name="embed-build")
    t.start()
    return {"started": True, "total": total_pending, "scope": scope or "all"}


def _target_scopes(scope) -> list:
    rows = load_scopes()
    if scope in (None, "all"):
        return [r["id"] for r in rows if r.get("enabled")]
    wanted = scope if isinstance(scope, list) else [scope]
    ids = [r["id"] for r in rows if r["id"] in wanted and r.get("enabled")]
    if not ids:
        raise KeyError("unknown scope %s" % (wanted,))
    return ids


def pending_files(folder: Path, sid: str, store: dict):
    """(to_embed, to_prune).
    to_embed: files not in store, or mtime changed since embedding
    to_prune: store keys tagged with this scope whose file vanished."""
    out = []
    for f in scan_files(folder):
        key = str(f)
        rec = store.get(key)
        try:
            mtime = int(f.stat().st_mtime)
        except OSError:
            continue
        if rec is None:
            out.append((f, "image" if f.suffix.lower() in cfg.IMG_EXTS else "video", mtime, sid))
        elif int(rec.get("mtime", 0)) != mtime:
            out.append((f, rec.get("modality", "image"), mtime, sid))
    pruned = [k for k, rec in store.items()
              if item_scope(rec, k) == sid and not Path(k).exists()]
    return out, pruned


def _checkpoint_store(mine: dict) -> None:
    """Short exclusive window: read whatever is on disk NOW, merge our records,
    write back atomically. Concurrent writers never lose data."""
    lk = open(cfg.EMBED_NEW.with_suffix(".lock"), "a+b")
    try:
        _lock(lk)
        txt = cfg.EMBED_NEW.read_text(encoding="utf-8") if cfg.EMBED_NEW.exists() else ""
        fresh = json.loads(txt) if txt.strip() else {}
        if not isinstance(fresh, dict):
            fresh = {}
        fresh.update(mine)
        _atomic_write(cfg.EMBED_NEW, json.dumps(fresh, ensure_ascii=False))
    except json.JSONDecodeError:
        _atomic_write(cfg.EMBED_NEW, json.dumps(mine, ensure_ascii=False))
    finally:
        _unlock(lk)
        lk.close()
    _lib_cache.clear()
    invalidate_matrix()


def _build_job(targets: list, rows: dict) -> None:
    global BUILD
    BUILD["state"] = "running"
    try:
        # JIT + LM bed check; BedBusyError -> build fails clean, state -> error
        model_layer.ensure_loaded()
        ml = model_layer.get()
        store, _src = load_library()
        jobs: list = []
        for sid in targets:
            pend, _prune = pending_files(Path(rows[sid]["path"]), sid, store)
            jobs.extend(pend)
        # prune vanished files of the target scopes (no embed needed)
        for sid in targets:
            _p, prune = pending_files(Path(rows[sid]["path"]), sid, store)
            for k in prune:
                store.pop(k, None)
        total = len(jobs)
        BUILD["total"], BUILD["done"] = total, 0
        if total == 0:
            BUILD.update(state="done", finished=time.strftime("%H:%M:%S"))
            return  # nothing to embed; pruned keys persist on the next checkpoint
        mine: dict = {}
        for idx, (f, mod, mtime, sid) in enumerate(jobs, 1):
            BUILD["done"], BUILD["cur"] = idx, f.name
            key = str(f)
            if mod == "image":
                vec = ml.embed_image(key)
            else:
                vec = ml.embed_video(key)
            mine[key] = {"modality": mod, "dim": 2048,
                         "vec": vec, "scope": sid, "mtime": mtime}
            store[key] = mine[key]
            if idx % cfg.BUILD_CHECKPOINT == 0 or idx == total:
                _checkpoint_store(mine)
        # refresh scope records; re-read to keep rows added mid-build (no lost update)
        fresh = {r["id"]: r for r in load_scopes()}
        for sid in targets:
            fresh[sid]["scanned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            fresh[sid]["count"] = count_files(Path(fresh[sid]["path"]))
        save_scopes(list(fresh.values()))
        BUILD.update(state="done", finished=time.strftime("%H:%M:%S"))
    except model_layer.BedBusyError as e:
        BUILD.update(state="error", err="LM bed occupied: %s" % e)
    except ReadOnlyRefused as e:
        BUILD.update(state="error", err=str(e))
    except Exception as e:  # keep build state visible in /status
        BUILD.update(state="error", err="%s: %s" % (type(e).__name__, e))


# ---------------------------------------------------------------- atomic json write
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
