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

    def delete(self, key: int) -> bool:
        """
        Remove key from the tree.

        Returns True if the key existed and was removed, False if it was
        not present.

        Algorithm
        ---------
        1. Walk from root to the target leaf, recording the full path
           (each node's page_id and its position inside its parent).
        2. Delete the key from the leaf.
        3. Walk the path back upward.  At each level:
             a. If the node is still at or above its minimum occupancy
                (max_keys // 2), stop — nothing above needs fixing.
             b. Otherwise try to *borrow* a key from a sibling:
                  - Right sibling has a spare  → borrow its leftmost key;
                    update the parent separator.
                  - Left sibling has a spare   → borrow its rightmost key;
                    update the parent separator.
             c. If neither sibling can spare a key, *merge* this node with
                one sibling, removing the separator key from the parent.
                The parent may now underflow — continue the loop.
        4. After the loop, check whether the root has become an IndexPage
           with 0 keys.  If so, its only remaining child becomes the new
           root (tree shrinks by one level).

        Minimum occupancy
        -----------------
        min_keys = max_keys // 2   (floor-division, matching the split rule)
        The root is exempt — it may hold as few as 1 key (IndexPage) or
        0 keys (empty LeafPage).
        """
        path    = self._find_path(key)
        leaf_id = path[-1][0]
        leaf    = self._pager.get_page(leaf_id)

        if not leaf.delete_key(key):
            return False

        self._fix_underflow(path)
        return True

    # ------------------------------------------------------------------
    # Delete internals
    # ------------------------------------------------------------------

    def _find_path(self, key: int) -> list:
        """
        Walk from root to the leaf that should contain key.

        Returns a list of (page_id, child_idx) pairs where child_idx is
        the position of this page inside its parent's children list.
        The root entry has child_idx = None.
        """
        path = [(self._root_id, None)]
        page = self._pager.get_page(self._root_id)

        while isinstance(page, IndexPage):
            # Mirror the routing logic in find_child, but also record the index.
            child_idx = len(page.children) - 1   # default: rightmost
            for i, k in enumerate(page.keys):
                if key < k:
                    child_idx = i
                    break
            child_id = page.children[child_idx]
            path.append((child_id, child_idx))
            page = self._pager.get_page(child_id)

        return path

    def _fix_underflow(self, path: list):
        """
        Walk the path bottom-up and fix any underflowing nodes.

        Stops as soon as a node is at or above its minimum occupancy, or
        a borrow (rather than a merge) resolves the underflow at some level.
        After the loop, checks whether the root should collapse.
        """
        # depth 0 is the root — we never check the root for underflow here;
        # root collapse is handled separately below.
        for depth in range(len(path) - 1, 0, -1):
            page_id, child_idx = path[depth]
            page     = self._pager.get_page(page_id)
            min_keys = page.max_keys // 2

            if page.num_keys() >= min_keys:
                return   # fine here — nothing above can be affected

            parent_id, _ = path[depth - 1]
            parent        = self._pager.get_page(parent_id)

            did_merge = self._borrow_or_merge(page, child_idx, parent)
            if not did_merge:
                return   # borrow: parent key count unchanged, done

            # merge: parent lost one key — continue upward

        # Check whether the root has been emptied by a merge
        root = self._pager.get_page(self._root_id)
        if isinstance(root, IndexPage) and root.num_keys() == 0:
            # The root had exactly 1 key, whose two children were just merged.
            # The surviving child becomes the new root.
            self._root_id            = root.children[0]
            self._pager.root_page_id = self._root_id

    def _borrow_or_merge(self, page, child_idx: int, parent) -> bool:
        """
        Attempt to fix an underflowing node.

        Tries right-borrow then left-borrow; falls back to a merge.

        Returns False if a borrow succeeded (parent key count unchanged).
        Returns True  if a merge was performed (parent lost one key).
        """
        is_leaf  = isinstance(page, LeafPage)
        min_keys = page.max_keys // 2

        has_right = child_idx < len(parent.children) - 1
        has_left  = child_idx > 0

        # ---- try borrow from right sibling ----------------------------
        if has_right:
            right = self._pager.get_page(parent.children[child_idx + 1])
            if right.num_keys() > min_keys:
                self._borrow_from_right(page, right, child_idx, parent, is_leaf)
                return False

        # ---- try borrow from left sibling -----------------------------
        if has_left:
            left = self._pager.get_page(parent.children[child_idx - 1])
            if left.num_keys() > min_keys:
                self._borrow_from_left(page, left, child_idx, parent, is_leaf)
                return False

        # ---- merge (prefer right, fall back to left) ------------------
        if has_right:
            right = self._pager.get_page(parent.children[child_idx + 1])
            self._merge_pages(page, right, child_idx, parent, is_leaf)
        else:
            left = self._pager.get_page(parent.children[child_idx - 1])
            self._merge_pages(left, page, child_idx - 1, parent, is_leaf)

        return True

    def _borrow_from_right(self, page, right, child_idx: int, parent,
                           is_leaf: bool):
        """
        Rotate the right sibling's leftmost key into page.

        Leaf  — the borrowed key moves directly; the parent separator is
                updated to the right sibling's new first key.
        Index — the parent separator rotates DOWN into page; the right
                sibling's first key rotates UP to the parent.
        """
        if is_leaf:
            entries              = right._entries()
            key_in, rec_in       = entries[0]
            right._rebuild(entries[1:])
            page.insert(key_in, rec_in)
            parent.keys[child_idx] = right._entries()[0][0]   # new first key of right
        else:
            sep                  = parent.keys[child_idx]
            page.keys.append(sep)
            page.children.append(right.children[0])
            parent.keys[child_idx] = right.keys[0]
            right.keys.pop(0)
            right.children.pop(0)

    def _borrow_from_left(self, page, left, child_idx: int, parent,
                          is_leaf: bool):
        """
        Rotate the left sibling's rightmost key into page.

        Leaf  — the borrowed key moves directly; the parent separator is
                updated to the borrowed key (= page's new first key).
        Index — the parent separator rotates DOWN into page; the left
                sibling's last key rotates UP to the parent.
        """
        if is_leaf:
            entries              = left._entries()
            key_in, rec_in       = entries[-1]
            left._rebuild(entries[:-1])
            page.insert(key_in, rec_in)
            parent.keys[child_idx - 1] = key_in   # new first key of page
        else:
            sep                    = parent.keys[child_idx - 1]
            page.keys.insert(0, sep)
            page.children.insert(0, left.children[-1])
            parent.keys[child_idx - 1] = left.keys[-1]
            left.keys.pop(-1)
            left.children.pop(-1)

    def _merge_pages(self, left, right, sep_idx: int, parent,
                     is_leaf: bool):
        """
        Absorb right into left, then remove the separator from parent.

        Leaf  — all entries of right move to left; the leaf linked list is
                restitched.  The separator is simply removed (it was only a
                routing copy, so it stays nowhere).
        Index — the separator is pulled DOWN into left (it's a push-up key,
                so it must live somewhere); right's keys and children are
                appended to left.
        """
        if is_leaf:
            left._rebuild(left._entries() + right._entries())
            left.next_page_id = right.next_page_id
        else:
            sep           = parent.keys[sep_idx]
            left.keys     = left.keys + [sep] + right.keys
            left.children = left.children + right.children

        parent.keys.pop(sep_idx)
        parent.children.pop(sep_idx + 1)

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
