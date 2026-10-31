# -*- coding: utf-8 -*-
"""MCP stdio transport ('medisearch mcp').

Harnesses spawn this process and speak JSON-RPC over stdin/stdout:
  initialize -> tools/list -> tools/call (media_search / media_status / ...)

The streamable-HTTP transport is part of 'medisearch serve' (one process,
one port - see server.py); stdio is the standalone face.

Stdout is the protocol pipe, so logs go to stderr; both are reconfigured to
UTF-8 (Windows console defaults to the system locale: Chinese query text
would corrupt before it even reached json.loads).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time

try:  # package mode
    from . import cfg
except ImportError:  # script mode
    import cfg


def log(msg: str) -> None:
    print(f"[medisearch-mcp] {msg}", file=sys.stderr, flush=True)


def run_stdio() -> None:
    # Encoding triple-set: stdin + stdout + stderr (see module docstring).
    for _s in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8")
            except Exception:
                pass

    try:
        from . import mcp_proto
    except ImportError as _e:
        import sys as _sys
        print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
        import mcp_proto
    log("stdio mode started (pid=%d); backend=%s, probe %s"
        % (os.getpid(), os.environ.get("WEMM_BACKEND", "auto"), cfg.API_BASE))

    def release_best_effort():
        try:
            try:
                from . import model_layer
            except ImportError as _e:
                import sys as _sys
                print("[medisearch] package-style import failed: %s; trying source-tree style" % _e, file=_sys.stderr)
                import model_layer
            # the release log print must not leak into the protocol pipe either
            _o = sys.stdout
            sys.stdout = sys.stderr
            try:
                model_layer.get().release()
            finally:
                sys.stdout = _o
        except Exception:
            pass

    atexit.register(release_best_effort)

    out = sys.stdout  # protocol sink (the reconfigured stdout object)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            resp = mcp_proto._err(None, -32700, "Parse error")
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
            continue
        # Guard window: python-level prints that happen while handling this
        # request (model JIT load "loading model..." / build progress) must not
        # leak into the protocol pipe - point sys.stdout at stderr for the
        # duration; protocol responses use the saved `out` object and are
        # unaffected (same trick the pre-merge MCP server used around wemm.load).
        sys.stdout = sys.stderr
        try:
            responses = mcp_proto.handle_batch(msg) if isinstance(msg, list) \
                else [mcp_proto.handle_message(msg)]
        finally:
            sys.stdout = out
        for r in responses:
            if r is not None:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
    log("stdio EOF, exiting")
