# -*- coding: utf-8 -*-
"""MediaSearch: local multimodal (image/video) content search.

Powered by WeMM-Embedding-2B (tencent/WeMM-Embedding-2B on Hugging Face,
Apache-2.0 weights; code license MIT - see LICENSE / README).

Entry points:
  medisearch serve   all-in-one service on one port (web UI + REST API + MCP /mcp + /v1/embeddings)
  medisearch mcp     MCP stdio transport (spawned by harnesses)
"""

__version__ = "0.1.0"
