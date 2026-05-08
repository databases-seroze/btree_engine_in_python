"""
Entry point for `python -m py_btree_engine`.

Usage:
    python -m py_btree_engine <dbdir>
    python -m py_btree_engine .
"""

import sys
from src.repl import run

if len(sys.argv) != 2:
    print("Usage: python -m py_btree_engine <dbdir>", file=sys.stderr)
    sys.exit(1)

run(sys.argv[1])
