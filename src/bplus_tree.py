"""
BPlusTree — the public interface for the B+Tree storage engine.

Public API
----------
    tree.insert(key, record)           insert / update a key
    tree.search(key)                   return record bytes or None
    tree.range_scan(start, end)        return [(key, record), ...] inclusive
    tree.flush()                       persist all cached pages to disk
    tree.close()                       flush and close the underlying file

Typical usage
-------------
Create a new on-disk tree:

    from src.bplus_pager import BPlusPager
    from src.bplus_tree  import BPlusTree

    pager = BPlusPager("users.db", order=100)
    tree  = BPlusTree(pager)
    tree.insert(1, encode_record(["alice", 30]))
    tree.flush()
    pager.close()

Reopen an existing tree:

    pager = BPlusPager("users.db")   # order is restored from file
    tree  = BPlusTree(pager)
    print(tree.search(1))            # b'...'

Design notes
------------
* Keys are unsigned 32-bit integers.
* Values are arbitrary bytes (use record.py to encode / decode rows).
* `order` controls the max keys per node; after an insert pushes a node
  over this limit the node is split immediately.
* Leaf splits COPY the middle key up (it stays as the first entry of the
  right leaf — required for correct routing).
* Index splits PUSH the middle key up (it leaves both child nodes).
* When the root splits, a new root IndexPage is created automatically and
  pager.root_page_id is updated so the file tracks the new root.
"""

from src.bplus_page  import LeafPage, IndexPage
from src.bplus_pager import BPlusPager


class BPlusTree:
    """
    B+Tree backed by a BPlusPager.

    Parameters
    ----------
    pager : BPlusPager, optional
        Supply your own pager (file-backed or in-memory).  If omitted an
        in-memory pager is created automatically.
    order : int
        Max keys per node.  Only used when *creating* a new in-memory pager
        (i.e. when pager=None).  When a pager is supplied, its own stored
        order takes precedence so that reopening a file always uses the
        same split threshold as when the file was created.
    """

    def __init__(self, pager: BPlusPager | None = None, order: int = 4):
        if pager is None:
            pager = BPlusPager(order=order)
        self._pager = pager
        # Always use the pager's order (important for file-backed reopens).
        self._order = pager._order

        if pager.root_page_id is not None:
            # Reopening an existing tree — root is already known.
            self._root_id = pager.root_page_id
        else:
            # Brand-new tree — create the first (empty) leaf as root.
            root = pager.new_leaf_page()
            self._root_id          = root.page_id
            pager.root_page_id     = self._root_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, key: int):
        """Return record bytes for key, or None if the key is absent."""
        leaf = self._find_leaf(key)
        return leaf.search(key)

    def insert(self, key: int, record: bytes):
        """Insert or update key → record.  Splits nodes as needed."""
        result = self._insert(self._root_id, key, record)
        if result is not None:
            # Root was split — promote a new root above both halves.
            push_up_key, new_page_id = result
            new_root           = self._pager.new_index_page()
            new_root.keys      = [push_up_key]
            new_root.children  = [self._root_id, new_page_id]
            self._root_id      = new_root.page_id
            self._pager.root_page_id = self._root_id   # keep pager in sync

    def range_scan(self, start_key: int, end_key: int) -> list:
        """
        Return [(key, record), ...] for all keys in [start_key, end_key].

        Uses the leaf linked list so only the relevant pages are visited.
        """
        leaf    = self._find_leaf(start_key)
        results = []

        while leaf is not None:
            for key, record in leaf._entries():
                if key > end_key:
                    return results
                if key >= start_key:
                    results.append((key, record))

            if leaf.next_page_id is None:
                break
            leaf = self._pager.get_page(leaf.next_page_id)

        return results

    def flush(self):
        """Write all cached pages and the meta page to disk (no-op for in-memory pager)."""
        self._pager.flush()

    def close(self):
        """Flush and close the underlying file."""
        self._pager.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_leaf(self, key: int) -> LeafPage:
        """Traverse index pages from root down to the correct leaf."""
        page = self._pager.get_page(self._root_id)
        while isinstance(page, IndexPage):
            child_id = page.find_child(key)
            page     = self._pager.get_page(child_id)
        return page

    def _insert(self, page_id: int, key: int, record: bytes):
        """
        Recursively insert into the subtree rooted at page_id.

        Returns (push_up_key, new_page_id) if a split occurred at this
        level, or None if no split was needed.
        """
        page = self._pager.get_page(page_id)

        # ---- Leaf node ------------------------------------------------
        if isinstance(page, LeafPage):
            page.insert(key, record)

            if not page.is_full():
                return None

            # Leaf exceeded capacity — split it.
            right     = self._pager.new_leaf_page()
            split_key = page.split(right)       # left stays in page
            return (split_key, right.page_id)   # split_key copied up

        # ---- Index node -----------------------------------------------
        child_id = page.find_child(key)
        result   = self._insert(child_id, key, record)

        if result is None:
            return None

        # A child split — absorb the new key into this index page.
        split_key, new_child_id = result
        page.insert_key(split_key, new_child_id)

        if not page.is_full():
            return None

        # Index page also exceeded capacity — split it.
        right_index = self._pager.new_index_page()
        push_up_key = page.split(right_index)   # middle key pushed up
        return (push_up_key, right_index.page_id)
