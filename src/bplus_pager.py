"""
BPlusPager — page allocator for the B+Tree.

Two modes
---------
In-memory  (default):  BPlusPager()
    Pages live in the buffer pool only.  Nothing is written to disk.
    Useful for unit tests and throwaway trees.

File-backed:           BPlusPager("mydb.db")
    Pages are persisted to a binary file.  The file begins with a 4096-byte
    *meta page* at offset 0, followed by data pages at offsets
    (page_id + 1) × PAGE_SIZE.

    Meta page layout (4096 bytes):
        [0:8]   magic       b"BPTREE\\x01\\x00"
        [8:12]  root_page_id  uint32LE  (0xFFFFFFFF = no root yet)
        [12:16] next_page_id  uint32LE  (next ID to hand out)
        [16:20] order         uint32LE  (max keys per node)
        [20:]   zeros

Cache / write policy
--------------------
Pages are managed through a BufferPool (LRU, configurable capacity).
New pages are written to disk immediately (write-through for allocation).
Modified pages are marked dirty and are NOT written automatically — call
flush() or close() to persist all dirty state to disk.  During normal
operation the buffer pool may also write dirty pages to disk eagerly when
it needs to evict a frame.
"""

import os
import struct
from typing import IO

from src.bplus_page  import LeafPage, IndexPage, PageType, PAGE_SIZE
from src.buffer_pool import BufferPool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_META_MAGIC   = b"BPTREE\x01\x00"   # 8 bytes
_NO_PAGE      = 0xFFFF_FFFF          # sentinel for "no page"
_META_OFFSET  = 0                    # meta page is always the first page


def _data_offset(page_id: int) -> int:
    """File offset for data page with the given ID."""
    return PAGE_SIZE + page_id * PAGE_SIZE   # skip the meta page at offset 0


# ---------------------------------------------------------------------------
# BPlusPager
# ---------------------------------------------------------------------------

class BPlusPager:
    """
    Allocates and retrieves B+Tree pages, optionally backed by a file.

    Parameters
    ----------
    filepath : str or None
        Path to the database file.  If the file does not exist it is
        created.  Pass None (default) for a pure in-memory pager.
    order : int
        Maximum number of keys per node before a split is triggered.
        Stored in the meta page so it is automatically restored when the
        file is reopened (you do not need to remember it).
    pool_capacity : int
        Maximum number of pages to keep in the buffer pool at once.
        Defaults to 256.  Ignored for in-memory pagers (the pool is still
        used but eviction only matters for file-backed operation).
    """

    def __init__(self, filepath: str | None = None, order: int = 4,
                 pool_capacity: int = 256):
        self._filepath  = filepath
        self._order     = order
        self._pool      = BufferPool(capacity=pool_capacity)
        self._next_id:  int = 0
        self.root_page_id: int | None = None
        self._file:     IO[bytes] | None = None

        if filepath is not None:
            if os.path.exists(filepath):
                self._open_existing()
            else:
                self._create_new()

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def _create_new(self):
        """Create a brand-new database file and write an empty meta page."""
        self._file = open(self._filepath, 'w+b')
        self._write_meta()

    def _open_existing(self):
        """Open an existing database file and restore state from the meta page."""
        self._file = open(self._filepath, 'r+b')
        self._read_meta()

    def _write_meta(self):
        """Serialise and write the meta page to offset 0."""
        buf = bytearray(PAGE_SIZE)
        buf[0:8] = _META_MAGIC
        root = _NO_PAGE if self.root_page_id is None else self.root_page_id
        struct.pack_into('<I', buf, 8,  root)
        struct.pack_into('<I', buf, 12, self._next_id)
        struct.pack_into('<I', buf, 16, self._order)
        self._file.seek(_META_OFFSET)
        self._file.write(buf)

    def _read_meta(self):
        """Read the meta page and restore root_page_id, next_page_id, and order."""
        self._file.seek(_META_OFFSET)
        buf = self._file.read(PAGE_SIZE)
        if len(buf) < 20 or buf[0:8] != _META_MAGIC:
            raise ValueError(
                f"'{self._filepath}' is not a valid B+Tree database file"
            )
        root           = struct.unpack_from('<I', buf, 8)[0]
        self.root_page_id = None if root == _NO_PAGE else root
        self._next_id  = struct.unpack_from('<I', buf, 12)[0]
        self._order    = struct.unpack_from('<I', buf, 16)[0]

    # ------------------------------------------------------------------
    # Low-level page I/O
    # ------------------------------------------------------------------

    def _write_page_to_disk(self, page: LeafPage | IndexPage):
        if self._file is None:
            return
        self._file.seek(_data_offset(page.page_id))
        self._file.write(page.to_bytes())

    def _read_page_from_disk(self, page_id: int) -> LeafPage | IndexPage:
        self._file.seek(_data_offset(page_id))
        data = self._file.read(PAGE_SIZE)
        if len(data) < PAGE_SIZE:
            raise IOError(f"Truncated read for page {page_id}")
        page_type = data[0]
        if page_type == PageType.LEAF.value:
            return LeafPage.from_bytes(page_id, data, max_keys=self._order)
        elif page_type == PageType.INDEX.value:
            return IndexPage.from_bytes(page_id, data, max_keys=self._order)
        else:
            raise ValueError(f"Unknown page type byte {page_type!r} for page {page_id}")

    # ------------------------------------------------------------------
    # Page allocation
    # ------------------------------------------------------------------

    def new_leaf_page(self) -> LeafPage:
        """Allocate a new leaf page, add it to the pool, and write it to disk."""
        page = LeafPage(self._next_id, max_keys=self._order)
        self._pool_put(page, dirty=False)   # new page written to disk below
        self._next_id += 1
        if self._file is not None:
            self._write_page_to_disk(page)
        return page

    def new_index_page(self) -> IndexPage:
        """Allocate a new index page, add it to the pool, and write it to disk."""
        page = IndexPage(self._next_id, max_keys=self._order)
        self._pool_put(page, dirty=False)
        self._next_id += 1
        if self._file is not None:
            self._write_page_to_disk(page)
        return page

    # ------------------------------------------------------------------
    # Page access
    # ------------------------------------------------------------------

    def get_page(self, page_id: int) -> LeafPage | IndexPage:
        """
        Return the page with the given ID.

        Checks the buffer pool first.  On a miss, reads from disk
        (file-backed mode only) and inserts the page into the pool.

        Raises
        ------
        KeyError  — in-memory mode, page_id was never allocated.
        IOError   — file-backed mode, read failed.
        """
        page = self._pool.get(page_id)
        if page is not None:
            return page
        if self._file is not None:
            page = self._read_page_from_disk(page_id)
            self._pool_put(page, dirty=False)
            return page
        raise KeyError(f"Page {page_id} does not exist")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self):
        """
        Write every dirty page and the meta page to disk.

        Call this after a batch of inserts to make the tree durable.  In
        in-memory mode this is a no-op.
        """
        if self._file is None:
            return
        for pid, page in self._pool.dirty_pages():
            self._write_page_to_disk(page)
            self._pool.clear_dirty(pid)
        self._write_meta()
        self._file.flush()

    def close(self):
        """Flush all data and close the file handle."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_pages(self) -> int:
        return self._next_id

    def mark_dirty(self, page):
        """
        Mark a page as dirty so it is written on the next flush.

        Accepts the page object (not just the ID) so that if the page was
        evicted from the pool while a Python reference was still live, it
        can be re-inserted before being marked dirty.
        """
        pid = page.page_id
        if not self._pool.contains(pid):
            # Page was evicted but the caller still holds a reference to it
            # and has just modified it — put it back so the dirty write
            # happens on the next flush.
            self._pool_put(page, dirty=True)
        else:
            self._pool.mark_dirty(pid)

    def _pool_put(self, page, dirty: bool):
        """
        Insert a page into the buffer pool, flushing any dirty evictions.
        """
        evicted = self._pool.put(page.page_id, page, dirty=dirty)
        for _pid, evicted_page in evicted:
            if self._file is not None:
                self._write_page_to_disk(evicted_page)
