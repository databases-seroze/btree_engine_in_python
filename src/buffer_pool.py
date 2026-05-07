"""
BufferPool — LRU page cache with dirty-page tracking and pin/unpin.

Sits between BPlusPager and the raw file I/O layer.  The pager calls
load_page / store_page; the buffer pool decides what lives in memory.

Design
------
* Fixed capacity (default 256 frames).
* LRU eviction order tracked by a doubly-linked dict (collections.OrderedDict).
* Dirty bit — set by mark_dirty(); cleared after a page is written to disk.
* Pin count — a pinned page is never evicted.  Any caller that holds a page
  reference should pin it, then unpin when done.  The pager pins each page
  it hands to the tree and unpins it when the tree is no longer using it.
  For simplicity the current implementation pins on get, unpins on explicit
  unpin() or on flush/eviction.

Frame layout (internal dict value)
-----------------------------------
Each frame is a plain dict:
    {
        "page":  LeafPage | IndexPage,
        "dirty": bool,
        "pins":  int,     # reference count; 0 = evictable
    }
"""

from collections import OrderedDict


class BufferPool:
    """
    LRU buffer pool for B+Tree pages.

    Parameters
    ----------
    capacity : int
        Maximum number of pages to keep in memory at once.
        Must be at least 1.
    """

    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("BufferPool capacity must be at least 1")
        self._capacity = capacity
        # OrderedDict preserves insertion/access order for LRU.
        # Key: page_id  Value: {"page": ..., "dirty": bool, "pins": int}
        self._frames: OrderedDict[int, dict] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API used by BPlusPager
    # ------------------------------------------------------------------

    def get(self, page_id: int):
        """
        Return the cached page for page_id, or None on a cache miss.

        Marks the frame as most-recently-used.  Does NOT change the pin
        count — use pin() / unpin() explicitly.
        """
        if page_id not in self._frames:
            return None
        self._frames.move_to_end(page_id)   # MRU position
        return self._frames[page_id]["page"]

    def put(self, page_id: int, page, *, dirty: bool = False) -> list:
        """
        Insert (or replace) a page in the buffer pool.

        New frames start with pin count 0 (evictable by default).  Use
        pin() afterwards if the caller needs the frame to be protected from
        eviction.

        If the pool is at capacity, the LRU unpinned frame is evicted first.
        If all frames are pinned the pool grows beyond capacity rather than
        blocking (the caller should avoid pinning everything indefinitely).

        Parameters
        ----------
        page_id : int
        page    : LeafPage | IndexPage
        dirty   : bool — True when the page was just modified.

        Returns
        -------
        list of (page_id, page) pairs that were evicted and need to be written
        to disk by the caller (dirty evictions only).
        """
        evicted = []

        if page_id in self._frames:
            # Update in place — preserve pin count, update dirty flag.
            frame = self._frames[page_id]
            frame["page"]  = page
            frame["dirty"] = frame["dirty"] or dirty
            self._frames.move_to_end(page_id)
            return evicted

        # Evict LRU unpinned pages until we have room.
        if len(self._frames) >= self._capacity:
            evicted = self._evict_one()

        self._frames[page_id] = {"page": page, "dirty": dirty, "pins": 0}
        return evicted

    def mark_dirty(self, page_id: int):
        """Mark a cached page as dirty (must be written before eviction)."""
        if page_id in self._frames:
            self._frames[page_id]["dirty"] = True

    def unpin(self, page_id: int):
        """Decrement the pin count for page_id (makes it eligible for eviction)."""
        if page_id in self._frames:
            frame = self._frames[page_id]
            if frame["pins"] > 0:
                frame["pins"] -= 1

    def pin(self, page_id: int):
        """Increment the pin count (prevent eviction)."""
        if page_id in self._frames:
            self._frames[page_id]["pins"] += 1

    def dirty_pages(self) -> list:
        """Return [(page_id, page), ...] for every dirty frame."""
        return [
            (pid, f["page"])
            for pid, f in self._frames.items()
            if f["dirty"]
        ]

    def clear_dirty(self, page_id: int):
        """Clear the dirty flag after the page has been written to disk."""
        if page_id in self._frames:
            self._frames[page_id]["dirty"] = False

    def contains(self, page_id: int) -> bool:
        return page_id in self._frames

    def size(self) -> int:
        """Number of frames currently in the pool."""
        return len(self._frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> list:
        """
        Evict the least-recently-used unpinned frame.

        Returns a list with 0 or 1 (page_id, page) pairs that are dirty
        and must be flushed to disk by the caller.
        """
        for pid, frame in self._frames.items():    # LRU → MRU order
            if frame["pins"] == 0:
                dirty_pair = [(pid, frame["page"])] if frame["dirty"] else []
                del self._frames[pid]
                return dirty_pair
        # All frames are pinned — pool grows beyond capacity.
        return []
