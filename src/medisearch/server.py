# -*- coding: utf-8 -*-
r"""MediaSearch service - FastAPI on one port, three faces:

  face 1  web UI + REST API (what the pre-merge 8686 process did)
    GET  /                 single-page UI (search box / scope checks / param panel / build bar)
    POST /search           {q, scope?, topn?, dim?, min_score?} -> [{path,score,modality,scope}]
    GET  /status           {model, vram_mb, items, lib, building, settings, events}
    POST /embed-scope      {scope?} -> start incremental build (new/changed files, by mtime)
    GET/PUT /scopes       scope list / toggle enabled
    POST /add-scope       {path, name?} -> register a folder
    POST /remove-scope    {id, delete_vectors?} -> unregister (vectors optional; files NEVER deleted)
    GET/PUT /settings     read / write data/settings.json (take effect immediately)
    GET  /file?path=...  serve a library image for thumbnails (library members only)
  face 2  MCP streamable-HTTP
    POST /mcp              JSON-RPC (notifications -> 202; GET -> 405; SSE when Accept wants it)
    GET  /healthz          liveness
  face 3  OpenAI-style fallback
    POST /v1/embeddings   text/image -> vector (always the local model layer)

Model policy: NOT loaded at startup. First /search / build /v1/embeddings
JIT-loads (warm-up ~10-60s; weights auto-download on first run); the watchdog
releases it after settings.idle_release_min of idle time (don't hog a local
LM Studio bed). One process, one port - no second process needed.

Read-only mode: env WEMM_READ_ONLY=1 (or --read-only) -> all mutating
endpoints refuse with 403 (search/status stay usable).

Console output is ASCII only (cmd/GBK).
"""
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:  # package mode (pip install / python -m medisearch)
    from . import cfg, libstore, model_layer, mcp_proto
except ImportError:  # script mode (python src/medisearch/server.py)
    import cfg, libstore, model_layer, mcp_proto


def _create_app() -> FastAPI:
    app = FastAPI(title="MediaSearch", version="0.1")

    # ---------------------------------------------------------------- payloads
    class SearchIn(BaseModel):
        q: str
        scope: str | list | None = None
        topn: int | None = None
        dim: int | None = None
        min_score: float | None = None

    class EmbedIn(BaseModel):
        scope: str | list | None = None

    class AddScopeIn(BaseModel):
        path: str
        name: str | None = None

    class SetScopeIn(BaseModel):
        id: str
        enabled: bool

    class RemoveScopeIn(BaseModel):
        id: str
        delete_vectors: bool | None = None

    class SettingsIn(BaseModel):
        topn: int | None = None
        dim: int | None = None
        min_score: float | None = None
        idle_release_min: float | None = None
        gpu_cap_gib: float | None = None

    # ---------------------------------------------------------------- face 1: web + API
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(cfg.STATIC / "index.html", media_type="text/html; charset=utf-8")

    @app.post("/search")
    def search(body: SearchIn):
        q = (body.q or "").strip()
        if not q:
            raise HTTPException(400, "empty query")
        s = libstore.load_settings()
        # same clamp as the MCP face (1..50): REST /search had no cap,
        # topn=999999 would return the whole library + hold the model lock.
        topn = max(1, min(int(body.topn or s["topn"]), 500))
        dim = int(body.dim or s["dim"])
        min_score = s["min_score"] if body.min_score is None else body.min_score
        scope = body.scope or "all"

        store, source = libstore.load_library()
        if not store:
            raise HTTPException(
                409,
                "no embedding library. add a folder and build it, or set "
                "'fallback_library' in config.json to an existing library file.")

        t0 = time.time()
        try:
            qvec = model_layer.get().embed_texts([q], dim=dim)[0]
        except model_layer.BedBusyError as e:
            raise HTTPException(
                503, "local model bed occupied by %s - MediaSearch will not fight for it. "
                     "Release it over there (or let it idle-release) and retry." % e)
        res = libstore.search(store, qvec, scope, None if scope == "all" else scope,
                              dim, topn, min_score)
        return {"q": q, "scope": scope, "dim": dim, "ms": int((time.time() - t0) * 1000),
                "lib": source, "results": res}

    @app.get("/status")
    def status():
        store, source = libstore.load_library()
        s = libstore.load_settings()
        return {
            "model": model_layer.get_state(),
            "vram_mb": model_layer.vram_mb(),
            "vram_total_mb": model_layer.vram_total_mb(),
            "items": len(store),
            "lib": source,                      # new | fallback | none
            "settings": s,
            "building": libstore.build_status(),
            "events": model_layer.get_events(),
            "read_only": cfg.is_read_only(),
        }

    @app.post("/embed-scope")
    def embed_scope(body: EmbedIn):
        try:
            info = libstore.start_build(body.scope)
        except libstore.ReadOnlyRefused as e:
            raise HTTPException(403, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))
        return info

    @app.get("/scopes")
    def get_scopes():
        return {"scopes": libstore.load_scopes()}

    @app.put("/scopes")
    def put_scope(body: SetScopeIn):
        try:
            r = libstore.set_scope_enabled(body.id, body.enabled)
        except libstore.ReadOnlyRefused as e:
            raise HTTPException(403, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))
        return r

    @app.post("/add-scope")
    def add_scope(body: AddScopeIn):
        try:
            r = libstore.add_scope(body.path, body.name)
        except libstore.ReadOnlyRefused as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return r

    @app.post("/remove-scope")
    def remove_scope_ep(body: RemoveScopeIn):
        """Only the registry + vector records are removed; source files are NEVER deleted."""
        try:
            dv = True if body.delete_vectors is None else bool(body.delete_vectors)
            return libstore.remove_scope(body.id, dv)
        except libstore.ReadOnlyRefused as e:
            raise HTTPException(403, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/settings")
    def get_settings():
        return libstore.load_settings()

    @app.put("/settings")
    def put_settings(body: SettingsIn):
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        if body.dim is not None and body.dim not in (2048, 1024, 512, 256, 128):
            raise HTTPException(400, "dim must be 2048/1024/512/256/128")
        if body.gpu_cap_gib is not None and not (0 <= body.gpu_cap_gib <= 64):
            raise HTTPException(400, "gpu_cap_gib must be 1..64")
        try:
            return libstore.save_settings(patch)
        except libstore.ReadOnlyRefused as e:
            raise HTTPException(403, str(e))

    @app.get("/file")
    def file(path: str):
        """Thumbnail access: only files that are members of the current library."""
        store, _src = libstore.load_library()
        p = Path(path)
        key = str(p.resolve()) if p.is_absolute() else None
        hit = None
        if key in store:
            hit = store[key]
        if hit is None:
            raise HTTPException(404, "not a library member")
        if not p.is_file():
            raise HTTPException(410, "file vanished from disk")
        MIMES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
                 ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
                 ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
        mt = MIMES.get(p.suffix.lower())
        if mt is None:
            raise HTTPException(415, "unsupported media type")
        return FileResponse(p, media_type=mt)

    @app.post("/reveal")
    def reveal(body: dict):
        """Open Explorer with the file selected (Win10/11). Library members only."""
        import ctypes
        p = Path((body or {}).get("path", ""))
        if not p.is_absolute():
            raise HTTPException(400, "path required")
        store, _src = libstore.load_library()
        if str(p.resolve()) not in store:
            raise HTTPException(404, "not a library member")
        if not p.is_file():
            raise HTTPException(410, "file vanished from disk")
        # official shell API: open folder window with the item selected (Win10/11)
        # NOTE: argtypes/restype are mandatory on 64-bit Python -- without them the
        # PIDL pointer is truncated to 32-bit and the call segfaults the process.
        shell32 = ctypes.windll.shell32
        shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
        shell32.ILCreateFromPathW.restype = ctypes.c_void_p
        shell32.SHOpenFolderAndSelectItems.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        shell32.SHOpenFolderAndSelectItems.restype = ctypes.HRESULT
        shell32.ILFree.argtypes = [ctypes.c_void_p]
        # SHOpenFolderAndSelectItems needs COM initialized on this thread
        # (FastAPI threadpool threads start uninitialized).
        ole32 = ctypes.windll.ole32
        coinited = ole32.CoInitializeEx(None, 0x2) >= 0  # COINIT_APARTMENTTHREADED
        try:
            pidl = shell32.ILCreateFromPathW(str(p))
            if not pidl:
                raise HTTPException(500, "cannot resolve path")
            try:
                if shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0) != 0:
                    raise HTTPException(500, "open folder failed")
            finally:
                shell32.ILFree(pidl)
        finally:
            if coinited:
                ole32.CoUninitialize()
        return {"ok": True, "path": str(p)}

    # ---------------------------------------------------------------- face 2: MCP streamable-HTTP
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "medisearch", "faces": ["ui", "api", "mcp", "v1"]}

    def _want_sse(req: Request) -> bool:
        acc = (req.headers.get("Accept") or "").lower()
        return "text/event-stream" in acc and "application/json" not in acc

    @app.get("/mcp", include_in_schema=False)
    def mcp_get():
        # spec 2025-03-26+: a server without GET support must answer 405
        return Response(
            status_code=405,
            content='{"jsonrpc":"2.0","id":null,"error":{"code":-32000,'
                    '"message":"GET not supported (streamable-HTTP is POST-only)"}}',
            media_type="application/json",
            headers={"Allow": "POST"})

    @app.post("/mcp", include_in_schema=False)
    async def mcp_post(req: Request):
        raw = await req.body()
        import json as _json
        try:
            msg = _json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            return _json_response(mcp_proto._err(None, -32700, "Parse error"), 400)
        if msg is None:
            return _json_response({"error": "empty body"}, 400)
        if isinstance(msg, list):
            responses = mcp_proto.handle_batch(msg)
        else:
            responses = [mcp_proto.handle_message(msg)]
        responses = [r for r in responses if r is not None]
        if not responses:
            return Response(status_code=202)  # all notifications
        if _want_sse(req):
            body = "".join(
                "event: message\ndata: %s\n\n" % _json.dumps(r, ensure_ascii=False)
                for r in responses).encode("utf-8")
            return Response(content=body, media_type="text/event-stream")
        payload = responses[0] if len(responses) == 1 else responses
        return _json_response(payload, 200)

    # ---------------------------------------------------------------- face 3: OpenAI-style /v1/embeddings
    @app.post("/v1/embeddings", include_in_schema=False)
    async def v1_embeddings(req: Request):
        try:
            from . import openai_compat
        except ImportError as _e:
            import sys as _sys
            print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
            import openai_compat
        status, body = openai_compat.handle_embeddings(await req.body())
        return _json_response(body, status)

    return app


def _json_response(payload, status: int = 200):
    import json as _json
    body = _json.dumps(payload, ensure_ascii=False)
    return Response(content=body, status_code=status, media_type="application/json")


# MCP tool dispatch runs in-process against this same library + model layer.
def wire_direct_backend() -> None:
    """Register the in-process tool implementations (no HTTP self-probe)."""
    try:
        from . import backend
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import backend
    backend.init_direct()


def main(host: str | None = None, port: int | None = None, read_only: bool = False,
         open_browser: bool = True) -> None:
    if read_only:
        cfg.set_read_only(True)
    model_layer.start_watchdog()
    wire_direct_backend()
    app = _create_app()
    host = host or cfg.HOST
    port = port if port is not None else cfg.PORT
    store, source = libstore.load_library()
    # port fallback: configured port taken (other app or another instance)
    # -> walk up to a free one instead of crashing; pin via settings "port"
    #    or env WEMM_PORT (MCP users: keep it pinned for stable config)
    import socket as _socket
    import sys as _sys
    _p0 = port
    for _ in range(20):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            try:
                _s.bind((host, port))
                break
            except OSError:
                port += 1
    else:
        raise SystemExit("no free port in %d..%d" % (_p0, port))
    if port != _p0:
        print("port %d busy -> using %d (pin via settings.json \"port\" or WEMM_PORT)"
              % (_p0, port), file=_sys.stderr)

    print("=" * 60)
    print("MediaSearch service on http://%s:%d/" % (host, port))
    print("faces: web UI + REST API + MCP /mcp + /v1/embeddings + /healthz (one port)")
    print("library: %s (%d items)  model: idle (JIT on first use, auto-download on first run)"
          % (source, len(store)))
    print("params: %s" % libstore.load_settings())
    print("read_only: %s   idle release: %s min" %
          (cfg.is_read_only(), libstore.load_settings()["idle_release_min"]))
    print("=" * 60)
    if open_browser:
        def _open():
            import webbrowser
            time.sleep(3)
            webbrowser.open("http://%s:%d/" % (host, port))
        threading.Thread(target=_open, daemon=True).start()
    # access_log off: /file?path=... requests would otherwise log every user's
    # local media paths into the service window (privacy leak, audit S16).
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
