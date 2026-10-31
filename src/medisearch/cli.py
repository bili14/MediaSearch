# -*- coding: utf-8 -*-
"""MediaSearch command line: one entry point, two faces.

  medisearch serve [--host H] [--port P] [--read-only] [--no-browser]
      all-in-one process on ONE port (default 127.0.0.1:8686):
      web UI (/) + REST API (/search /status /embed-scope /scopes /add-scope
      /remove-scope /settings /file) + MCP streamable-HTTP (/mcp /healthz)
      + OpenAI-style /v1/embeddings. No second process needed.

  medisearch mcp [--read-only]
      MCP stdio transport for harnesses that spawn a child process.
      Backend: WEMM_BACKEND=auto (default) probes a running service and
      falls back to the local library; =api forces pass-through;
      =local forces local.

Read-only: WEMM_READ_ONLY=1 (env) or --read-only refuses all writes
(scope add/PUT/remove, build, settings PUT) with 403 / tool errors.
"""
from __future__ import annotations

import argparse
import sys


def _pin_and_env() -> None:
    try:
        from . import cfg
    except ImportError:
        import cfg
    cfg.pin_tmp()


def cmd_serve(args) -> int:
    _pin_and_env()
    try:
        from . import cfg, server
    except ImportError:
        import cfg, server

    if args.read_only:
        cfg.set_read_only(True)
        try:
            from . import mcp_proto
        except ImportError:
            import mcp_proto
        mcp_proto.set_read_only(True)
    server.main(host=args.host, port=args.port,
                read_only=args.read_only, open_browser=not args.no_browser)
    return 0


def cmd_mcp(args) -> int:
    _pin_and_env()
    if args.read_only:
        try:
            from . import cfg, mcp_proto
        except ImportError:
            import cfg, mcp_proto
        cfg.set_read_only(True)
        mcp_proto.set_read_only(True)
    try:
        from . import mcp_server
    except ImportError:
        import mcp_server
    mcp_server.run_stdio()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="medisearch",
                                 description="MediaSearch - local multimodal media search (WeMM-Embedding-2B)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve",
                       help="all-in-one service: web UI + REST API + MCP /mcp + /v1/embeddings on one port")
    s.add_argument("--host", default=None, help="bind address (default: 127.0.0.1 or WEMM_HOST)")
    s.add_argument("--port", type=int, default=None, help="port (default: 8686 or WEMM_PORT)")
    s.add_argument("--read-only", action="store_true",
                   help="refuse writes (WEMM_READ_ONLY=1 equivalent)")
    s.add_argument("--no-browser", action="store_true",
                   help="do not auto-open the UI in a web browser")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("mcp", help="MCP stdio transport (spawned by harnesses)")
    m.add_argument("--read-only", action="store_true",
                   help="expose search/status tools only")
    m.set_defaults(func=cmd_mcp)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
