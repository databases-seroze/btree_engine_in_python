"""
Engine — the top-level interface for the B+Tree database.

The Engine owns a BPlusPager and a BPlusTree and translates between the
caller's Python rows (lists of values) and the raw bytes stored in the tree.

Usage
-----
In-memory (no persistence):

    from src.engine import Engine

    db = Engine()
    db.put(1, [1, "alice", 30])
    db.put(2, [2, "bob",   25])

    row = db.get(1)           # [1, "alice", 30]
    db.delete(1)

    for key, row in db.scan(1, 100):
        print(key, row)

    db.close()

File-backed (survives process restart):

    with Engine("users.db", order=100) as db:
        db.put(1, [1, "alice", 30])
        db.flush()

    # Later:
    with Engine("users.db") as db:   # order restored from file
        print(db.get(1))             # [1, "alice", 30]

Parameters
----------
filepath : str or None
    Path to the database file.  Pass None (default) for a pure in-memory
    engine that is discarded when close() is called.
order : int
    Max keys per B+Tree node before a split.  Only used when creating a
    *new* file; reopening an existing file always uses the order stored in
    the file's meta page.  Defaults to 100 (reasonable for 4 KB pages).
"""

from src.bplus_pager  import BPlusPager
from src.bplus_tree   import BPlusTree
from src.cursor       import Cursor
from src.record       import encode_record, decode_record
from src.transaction  import Transaction


class Engine:
    """
    High-level database interface.

    Rows are Python lists whose elements may be int, str, or None — the
    same types supported by record.py.  Keys are unsigned 32-bit integers.
    """

    def __init__(self, filepath: str | None = None, order: int = 100):
        self._pager = BPlusPager(filepath, order=order)
        self._tree  = BPlusTree(self._pager)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, key: int, row: list) -> None:
        """
        Insert or update a row.

        Parameters
        ----------
        key : int
            Unsigned 32-bit integer key.
        row : list
            List of column values (int | str | None).
        """
        self._tree.insert(key, encode_record(row))

    def get(self, key: int) -> list | None:
        """
        Return the row stored under key, or None if the key is absent.
        """
        raw = self._tree.search(key)
        return decode_record(raw) if raw is not None else None

    def delete(self, key: int) -> bool:
        """
        Remove key from the database.

        Returns True if the key existed and was removed, False if absent.
        """
        return self._tree.delete(key)

    def scan(self, start: int = 0, end: int = 0xFFFF_FFFF) -> 'Cursor':
        """
        Return a Cursor positioned at the first row with key >= start.

        The cursor yields (key, row) pairs for all keys in [start, end]
        in ascending order.  It supports forward iteration via next(),
        backward iteration via prev(), and reset() to go back to start.

        Parameters
        ----------
        start : int  lower bound (inclusive), default 0
        end   : int  upper bound (inclusive), default 2^32-1
        """
        return Cursor(self._tree, start, end)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def begin(self) -> 'Transaction':
        """
        Open a new transaction.

        All put() / delete() calls on the returned Transaction object are
        buffered until commit() is called.  Reads (get, scan) reflect the
        transaction's own staged writes via an in-memory overlay.

        Use as a context manager for automatic commit / rollback:

            with db.begin() as txn:
                txn.put(1, [1, "alice"])
                txn.delete(2)
            # committed on success, rolled back on exception
        """
        return Transaction(self)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all cached pages to disk (no-op for in-memory engine)."""
        self._tree.flush()

    def close(self) -> None:
        """Flush and release all resources."""
        self._tree.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'Engine':
        return self

    def __exit__(self, *_) -> None:
        self.close()
