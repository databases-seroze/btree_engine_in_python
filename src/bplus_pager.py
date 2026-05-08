"""
BPlusPager — page allocator for the B+Tree.

Two modes
---------
In-memory  (default):  BPlusPager()
    Pages live in the buffer pool only.  Nothing is written to disk.
    Useful for unit tests and throwaway trees.

File-backed:           BPlusPager("mydb.db")
    Pages are persisted to a binary file with a WAL for crash safety.
    The file begins with a 4096-byte *meta page* at offset 0, followed
    by data pages at offsets (page_id + 1) × PAGE_SIZE.

    Meta page layout (4096 bytes):
        [0:8]   magic       b"BPTREE\\x01\\x00"
        [8:12]  root_page_id  uint32LE  (0xFFFFFFFF = no root yet)
        [12:16] next_page_id  uint32LE  (next ID to hand out)
        [16:20] order         uint32LE  (max keys per node)
        [20:]   zeros

WAL integration
---------------
The WAL file lives alongside the data file as "<filepath>.wal".

Write path:
  mark_dirty(page):
    1. WAL.append_page_write(page_id, page.to_bytes()) → lsn
    2. page.page_lsn = lsn
    3. pool.mark_dirty(page_id)

  flush() / eviction:
    Before writing a dirty page to the data file:
      if page.page_lsn > wal.flushed_lsn:
          wal.fsync_up_to(page.page_lsn)   ← WAL first
    Write page to data file.
    After all dirty pages are written:
      wal.checkpoint()                      ← truncates WAL

Recovery (on open of existing file):
  If .wal exists and is non-empty:
    wal.recover(write_page_fn)  — replay PAGE_WRITE records
    Update next_page_id if WAL saw higher page ids than the meta page.
    Truncate WAL (done inside wal.recover → wal.checkpoint).
"""

import os
import struct
from typing import IO

from src.bplus_page  import LeafPage, IndexPage, PageType, PAGE_SIZE
from src.buffer_pool import BufferPool
from src.wal         import WAL


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
        Defaults to 256.
    """

    def __init__(self, filepath: str | None = None, order: int = 4,
                 pool_capacity: int = 256):
        self._filepath  = filepath
        self._order     = order
        self._pool      = BufferPool(capacity=pool_capacity)
        self._next_id:  int = 0
        self.root_page_id: int | None = None
        self._file:     IO[bytes] | None = None
        self._wal:      WAL | None = None
        # Transaction support: when True, buffer pool evictions are deferred
        # so that no partially-committed pages reach the data file before
        # TXN_COMMIT is fsynced to the WAL.
        self._txn_writing:        bool = False
        self._deferred_evictions: list = []
        self._next_txn_id:        int  = 1

        if filepath is not None:
            if os.path.exists(filepath):
                self._open_existing()
            else:
                self._create_new()

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def _create_new(self):
        """Create a brand-new database file, WAL, and write an empty meta page."""
        self._file = open(self._filepath, 'w+b')
        self._wal  = WAL(self._filepath + '.wal')
        self._write_meta()

    def _open_existing(self):
        """
        Open an existing database file.

        If a non-empty WAL file exists alongside it, run crash recovery
        before handing control back to the caller.
        """
        self._file = open(self._filepath, 'r+b')
        self._read_meta()

        wal_path = self._filepath + '.wal'
        self._wal = WAL(wal_path)

        if not self._wal.is_empty():
            self._recover()

    def _recover(self):
        """
        Replay the WAL into the data file.

        Called on startup when the WAL is non-empty, indicating a previous
        process crashed before completing a flush/checkpoint.
        """
        info = self._wal.recover(self._write_raw_page_to_disk)

        # Restore root_page_id from WAL if the meta page is stale.
        if info["had_meta"]:
            self.root_page_id = info["root_page_id"]

        # next_page_id: use whichever is higher — meta page or WAL high-water mark.
        max_page_id = info["max_page_id"]
        if max_page_id >= self._next_id:
            self._next_id = max_page_id + 1

        # Write the corrected meta page so root_page_id and next_page_id are
        # durable.  If we crash before this the WAL is still intact and replay
        # will run again on the next open (idempotent).
        self._write_meta()
        self._file.flush()
        os.fsync(self._file.fileno())

        # All page changes are now durably in the data file — truncate WAL.
        self._wal.checkpoint()

    # ------------------------------------------------------------------
    # Meta page I/O
    # ------------------------------------------------------------------

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
        root              = struct.unpack_from('<I', buf, 8)[0]
        self.root_page_id = None if root == _NO_PAGE else root
        self._next_id     = struct.unpack_from('<I', buf, 12)[0]
        self._order       = struct.unpack_from('<I', buf, 16)[0]

    # ------------------------------------------------------------------
    # Low-level page I/O
    # ------------------------------------------------------------------

    def _write_page_to_disk(self, page: LeafPage | IndexPage):
        """Write an in-memory page object to the data file."""
        if self._file is None:
            return
        self._file.seek(_data_offset(page.page_id))
        self._file.write(page.to_bytes())

    def _write_raw_page_to_disk(self, page_id: int, page_bytes: bytes):
        """Write raw page bytes to the data file (used during WAL recovery)."""
        self._file.seek(_data_offset(page_id))
        self._file.write(page_bytes)

    def _read_page_from_disk(self, page_id: int) -> LeafPage | IndexPage:
        self._file.seek(_data_offset(page_id))
        data = self._file.read(PAGE_SIZE)
        if len(data) < PAGE_SIZE:
            raise IOError(f"Truncated read for page {page_id}")
        # page_lsn is at [0:8], page_type is at [8]
        page_type = data[8]
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
        self._pool_put(page, dirty=False)
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
    # Durability
    # ------------------------------------------------------------------

    def mark_dirty(self, page):
        """
        Mark a page as dirty and append a WAL record for it.

        Must be called AFTER the page has been modified in memory.
        The WAL record captures the page's new state.  The record is
        buffered in memory — not yet on disk — until fsync_up_to() is
        called during flush or eviction.

        page.page_lsn is updated to the LSN of the new WAL record so the
        eviction path can enforce the WAL-before-page guarantee.
        """
        pid = page.page_id

        # Log the page's current (modified) state to the WAL buffer.
        if self._wal is not None:
            lsn           = self._wal.append_page_write(pid, page.to_bytes())
            page.page_lsn = lsn

        # Re-insert into pool if it was evicted while still referenced.
        if not self._pool.contains(pid):
            self._pool_put(page, dirty=True)
        else:
            self._pool.mark_dirty(pid)

    def flush(self):
        """
        Write every dirty page and the meta page to disk, then checkpoint WAL.

        Sequence:
          1. fsync all buffered WAL records (guarantees WAL is ahead of data).
          2. Write dirty data pages to the data file.
          3. fsync the data file.
          4. Write meta page to the data file.
          5. fsync the data file again (meta page is now durable).
          6. WAL checkpoint: append CHECKPOINT record, fsync, truncate WAL.

        In-memory mode: no-op.
        """
        if self._file is None:
            return

        # Step 1 — WAL must reach disk before any data page does.
        if self._wal is not None:
            self._wal.fsync()

        # Step 2 — write dirty data pages.
        for pid, page in self._pool.dirty_pages():
            self._write_page_to_disk(page)
            self._pool.clear_dirty(pid)

        # Step 3 — fsync data file (data pages durable).
        self._file.flush()
        os.fsync(self._file.fileno())

        # Step 4 & 5 — meta page durable.
        self._write_meta()
        self._file.flush()
        os.fsync(self._file.fileno())

        # Step 6 — checkpoint: truncate WAL (everything is now in the data file).
        if self._wal is not None:
            self._wal.checkpoint()

    def close(self):
        """Flush all data, close WAL, and close the data file."""
        self.flush()
        if self._wal is not None:
            self._wal.close()
            self._wal = None
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

    def alloc_txn_id(self) -> int:
        """Return a monotonically increasing transaction ID."""
        txn_id = self._next_txn_id
        self._next_txn_id += 1
        return txn_id

    def log_meta_update(self):
        """
        Append a META_UPDATE WAL record with the current root_page_id.

        Must be called by BPlusTree immediately after changing root_page_id
        (root split or root collapse).  This ensures recovery can restore
        the correct root even if we crash before flush() writes the meta page.
        """
        if self._wal is not None:
            self._wal.append_meta_update(self.root_page_id)

    def begin_txn_write(self):
        """
        Signal that an explicit transaction is about to apply its ops.

        While _txn_writing is True, buffer pool evictions are deferred —
        no modified page can reach the data file before TXN_COMMIT is fsynced.
        """
        self._txn_writing        = True
        self._deferred_evictions = []

    def end_txn_write(self):
        """
        Signal that TXN_COMMIT has been fsynced.

        Flushes all pages that were deferred during the transaction apply phase.
        These pages are safe to write because TXN_COMMIT is now on disk.
        """
        self._txn_writing = False
        for evicted_page in self._deferred_evictions:
            if self._file is not None:
                if self._wal is not None and evicted_page.page_lsn > self._wal.flushed_lsn:
                    self._wal.fsync_up_to(evicted_page.page_lsn)
                self._write_page_to_disk(evicted_page)
        self._deferred_evictions = []

    def _pool_put(self, page, dirty: bool):
        """
        Insert a page into the buffer pool, handling dirty evictions.

        For file-backed pagers, an evicted dirty page must have its WAL
        record on disk before the page bytes reach the data file.
        When _txn_writing is True, evictions are deferred so that no
        partially-committed page reaches disk before TXN_COMMIT is fsynced.
        """
        evicted = self._pool.put(page.page_id, page, dirty=dirty)
        for _pid, evicted_page in evicted:
            if self._file is not None:
                if self._txn_writing:
                    # Defer: the transaction hasn't committed yet.
                    self._deferred_evictions.append(evicted_page)
                else:
                    if self._wal is not None and evicted_page.page_lsn > self._wal.flushed_lsn:
                        self._wal.fsync_up_to(evicted_page.page_lsn)
                    self._write_page_to_disk(evicted_page)
