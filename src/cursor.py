"""
Cursor — stateful iterator over a key range in the B+Tree.

The cursor is positioned AT the current entry (not before it like a
traditional Python iterator).  Reading key / value is free; calling
next() or prev() moves to the adjacent entry.

Forward iteration (next) is O(1) amortised — it follows the leaf
linked list directly.

Backward iteration (prev) is O(k) where k is the number of entries
between start and the current position, because the leaf pages are
singly-linked (forward only).  prev() re-uses BPlusTree.range_scan
internally to find the preceding key.

Usage
-----
    cursor = db.scan(10, 50)            # positioned at first key >= 10

    while cursor.key is not None:
        print(cursor.key, cursor.value)
        cursor.next()

    # or as a for loop (consumes the cursor from current position):
    for key, row in db.scan(10, 50):
        print(key, row)

    # move backward:
    cursor = db.scan(1, 100)
    cursor.next()   # key=2
    cursor.prev()   # key=1 again
    cursor.reset()  # back to the very first key in range

Note: modifying the tree (insert / delete) while a cursor is open
produces undefined behaviour.  Close and re-open the cursor after
writes.
"""

from src.record import decode_record


class Cursor:
    """
    Stateful forward/backward iterator over a contiguous key range.

    Parameters
    ----------
    tree      : BPlusTree  — the tree to iterate over
    start_key : int        — inclusive lower bound
    end_key   : int        — inclusive upper bound
    """

    def __init__(self, tree, start_key: int, end_key: int):
        self._tree  = tree
        self._start = start_key
        self._end   = end_key

        # Current position: (leaf page, index into leaf._entries())
        self._leaf  = tree._find_leaf(start_key)
        self._idx   = self._first_valid_idx()

        # Decoded current entry — None when exhausted
        self._key:   int  | None  = None
        self._value: list | None  = None

        # Advance to the first entry that is inside the range
        self._advance()

    # ------------------------------------------------------------------
    # Current position
    # ------------------------------------------------------------------

    @property
    def key(self) -> int | None:
        """Current key, or None if the cursor is exhausted."""
        return self._key

    @property
    def value(self) -> list | None:
        """Current decoded row, or None if the cursor is exhausted."""
        return self._value

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next(self) -> bool:
        """
        Advance to the next entry.

        Returns True if the cursor landed on a valid entry inside the
        range, False if the range is exhausted.
        """
        if self._key is None:
            return False
        self._idx += 1
        return self._advance()

    def prev(self) -> bool:
        """
        Move to the previous entry.

        Because leaf pages are singly-linked (forward only), prev()
        re-walks the range [start, current_key - 1] to find the
        preceding key.  This is O(k) in the number of entries before
        the current position.

        Returns True if there was a previous entry, False if the cursor
        is already at the first entry in the range.
        """
        if self._key is None or self._key <= self._start:
            return False

        # Collect everything in [start, current_key - 1]
        before = self._tree.range_scan(self._start, self._key - 1)
        if not before:
            return False

        prev_key, prev_raw = before[-1]

        # Reposition the cursor at that entry
        self._leaf = self._tree._find_leaf(prev_key)
        entries    = self._leaf._entries()
        self._idx  = next(i for i, (k, _) in enumerate(entries) if k == prev_key)
        self._key   = prev_key
        self._value = decode_record(prev_raw)
        return True

    def reset(self) -> bool:
        """
        Reposition the cursor at the first entry in the range.

        Returns True if the range contains at least one entry,
        False if it is empty.
        """
        self._leaf = self._tree._find_leaf(self._start)
        self._idx  = self._first_valid_idx()
        return self._advance()

    # ------------------------------------------------------------------
    # Iterator protocol  (for key, row in cursor: ...)
    # ------------------------------------------------------------------

    def __iter__(self) -> 'Cursor':
        return self

    def __next__(self) -> tuple[int, list]:
        if self._key is None:
            raise StopIteration
        result = (self._key, self._value)
        self.next()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _first_valid_idx(self) -> int:
        """
        Return the index of the first entry in the current leaf that
        has key >= start.  Returns len(entries) if all entries are
        below start (advance() will then move to the next leaf).
        """
        for i, (k, _) in enumerate(self._leaf._entries()):
            if k >= self._start:
                return i
        return len(self._leaf._entries())

    def _advance(self) -> bool:
        """
        Walk forward from (self._leaf, self._idx) until we find an entry
        inside [start, end], or exhaust the range.

        Sets self._key / self._value on success.
        Clears them and returns False when the range is exhausted.
        """
        while self._leaf is not None:
            entries = self._leaf._entries()

            if self._idx < len(entries):
                k, raw = entries[self._idx]

                if k > self._end:
                    # Passed the right bound — done
                    break

                if k >= self._start:
                    self._key   = k
                    self._value = decode_record(raw)
                    return True

                # Still before start — skip
                self._idx += 1
                continue

            # Exhausted this leaf — jump to the right sibling
            if self._leaf.next_page_id is None:
                break
            self._leaf = self._tree._pager.get_page(self._leaf.next_page_id)
            self._idx  = 0

        self._key   = None
        self._value = None
        return False
