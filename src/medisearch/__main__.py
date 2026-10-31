# -*- coding: utf-8 -*-
"""python -m medisearch == medisearch (console script)."""
import sys

try:
    from .cli import main
except ImportError:
    from cli import main

if __name__ == "__main__":
    sys.exit(main())
