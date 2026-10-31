# -*- coding: utf-8 -*-
"""MCP protocol layer (JSON-RPC 2.0, transport-agnostic).

stdio and streamable-HTTP transports share this one implementation.
The transports only hand JSON messages in and responses out; all MCP
semantics live here.

Protocol stance (MCP 2025-06-18):
- server implements the tools capability only (advertise capabilities={tools:{}});
  resources/prompts not implemented -> -32601 for those methods
- stateless: no Mcp-Session-Id; notifications (no id) get no reply (stdio)
  or a 202 (HTTP, handled by the transport)
- GET on the MCP endpoint -> 405 (allowed by spec: server does not support GET)

Read-only switch: env WEMM_READ_ONLY=1|true|yes|on or CLI --read-only.
When read-only, tools/list exposes only media_search / media_status, and the
write tools refuse with a clear error even when called by name (guards against
hosts that cached the older tool list).
"""
from __future__ import annotations

import json
import os

SERVER_INFO = {"name": "medisearch-mcp", "version": "0.1.0"}
PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18"]
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

READ_ONLY = os.environ.get("WEMM_READ_ONLY", "").strip().lower() in ("1", "true", "yes", "on")
READ_ONLY_TOOLS = ("media_search", "media_status")
READ_ONLY_ERR = (
    "Read-only mode (WEMM_READ_ONLY=1 or --read-only): this tool is disabled - "
    "read-only access exposes media_search / media_status only. "
    "Connect a non-read-only server for media_build / media_remove_scope."
)


def set_read_only(on: bool = True) -> None:
    """Called by the CLI after parsing --read-only; env form is active at import."""
    global READ_ONLY
    READ_ONLY = bool(on)


TOOL_DEFS = [
    {
        "name": "media_search",
        "description": (
            "Fuzzy content search over your media (images/videos - the content, not file names). "
            "WeMM-Embedding-2B multimodal embeddings + cosine topN. "
            "Returns topN {path, score, modality, scope} rows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Fuzzy phrase describing what is on screen (e.g. a girl sitting on a windowsill, a seagull)"},
                "scope": {
                    "type": "string",
                    "description": "Scope filter: 'all' or one registered scope id (see media_status / GET /scopes)",
                },
                "topn": {"type": "integer", "description": "Result count, default 5, max 50", "minimum": 1, "maximum": 50},
            },
            "required": ["q"],
        },
    },
    {
        "name": "media_status",
        "description": (
            "Service state: backend (direct = in-process / api = HTTP pass-through / local = local library), "
            "model loaded|idle, library path + item count, vector dim, build progress (building)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "media_build",
        "description": (
            "Incremental library build: embed new/changed files of registered scopes "
            "(images ~1.2s each on a typical GPU); poll media_status (building.done/total) for progress. "
            "Returns {started, total, scope}; scope omitted = all enabled scopes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Build scope: a single string (a scope id) or an array of scope ids; "
                        "omit = all enabled scopes"
                    ),
                },
            },
        },
    },
    {
        "name": "media_remove_scope",
        "description": (
            "Unregister a scope (removes the registry row and, by default, its vector records). "
            "Safety contract: files on disk are NEVER deleted (files_deleted is always 0). "
            "Returns {id, registry_removed, vectors_removed, files_deleted, note}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Scope id to remove"},
                "delete_vectors": {
                    "type": "boolean",
                    "default": True,
                    "description": "Also remove the scope's vector records, default true; false = registry only",
                },
            },
            "required": ["id"],
        },
    },
]


def _ok(msg_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _err_content(msg: str) -> dict:
    """Param/exception class error -> MCP result (isError, text explanation or JSON)."""
    return {"content": [{"type": "text", "text": msg}], "isError": True}


def _ok_content(res) -> dict:
    """backend entry result -> MCP result (dict carrying 'error' also gets isError)."""
    out = {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}]}
    if isinstance(res, dict) and res.get("error"):
        out["isError"] = True
    return out


def _tools_call(arguments: dict) -> dict:
    """Returns the MCP tools/call result object (content array)."""
    try:
        from . import backend
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import backend

    name = arguments.get("name")
    params = arguments.get("arguments") or {}

    # read-only double guard: even a host that cached the old tool list cannot trigger writes
    if READ_ONLY and name in ("media_build", "media_remove_scope"):
        return _err_content(READ_ONLY_ERR)

    if name == "media_search":
        q = params.get("q")
        if not isinstance(q, str) or not q.strip():
            return _err_content("bad params: q is required and must be a non-empty string")
        topn = params.get("topn", 5)
        try:
            res = backend.media_search(q, params.get("scope"), int(topn))
        except Exception as e:
            return _err_content(json.dumps(
                {"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return _ok_content(res)

    if name == "media_status":
        try:
            res = backend.media_status()
        except Exception as e:
            return _err_content(json.dumps(
                {"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return _ok_content(res)

    if name == "media_build":
        # scope: omit / single string / string array
        scope = params.get("scope")
        if scope is not None:
            if isinstance(scope, str):
                scope = scope.strip() or None
            elif isinstance(scope, list) and scope and all(isinstance(s, str) for s in scope):
                scope = [s.strip() for s in scope if s.strip()] or None
            elif isinstance(scope, list) and not scope:
                scope = None
            else:
                return _err_content('bad params: scope must be omitted, a string, or an array of strings (e.g. ["downloads"])')
        try:
            res = backend.media_build(scope)
        except Exception as e:
            return _err_content(json.dumps(
                {"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return _ok_content(res)

    if name == "media_remove_scope":
        sid = params.get("id")
        if not isinstance(sid, str) or not sid.strip():
            return _err_content("bad params: id is required and must be a non-empty string")
        dv = params.get("delete_vectors", True)
        if not isinstance(dv, bool):
            return _err_content("bad params: delete_vectors must be a boolean (default true)")
        try:
            res = backend.media_remove_scope(sid.strip(), dv)
        except Exception as e:
            return _err_content(json.dumps(
                {"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return _ok_content(res)

    return _err_content("unknown tool: %r (only media_search / media_status / media_build / media_remove_scope exist)" % name)


def handle_message(msg) -> dict | None:
    """One JSON-RPC message -> response dict; notifications / no-reply -> None."""
    if not isinstance(msg, dict):
        return _err(None, -32600, "Invalid Request (not a JSON object)")

    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # no method = a response to a server->client request (we send none; ignore)
    if method is None:
        return None

    if method == "initialize":
        pv = params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION)
        negotiated = pv if pv in PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        return _ok(msg_id, {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if method == "ping":
        return _ok(msg_id, {})

    if msg_id is None:
        # notification (notifications/initialized, cancelled, ...): no reply
        return None

    if method == "tools/list":
        tools = [t for t in TOOL_DEFS if t["name"] in READ_ONLY_TOOLS] if READ_ONLY else TOOL_DEFS
        return _ok(msg_id, {"tools": tools})

    if method == "tools/call":
        args = params if isinstance(params, dict) else {}
        return _ok(msg_id, _tools_call(args))

    if method in ("resources/list", "prompts/list", "resources/read", "prompts/get"):
        return _err(msg_id, -32601, "method not implemented (this server only provides tools): %s" % method)

    return _err(msg_id, -32601, "unknown method: %s" % method)


def handle_batch(msgs: list) -> list:
    """Batch message -> response list (None notifications filtered out)."""
    out = []
    for m in msgs:
        r = handle_message(m)
        if r is not None:
            out.append(r)
    return out
