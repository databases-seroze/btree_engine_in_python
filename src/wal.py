"""
WAL — Write-Ahead Log for the B+Tree storage engine.

Design
------
Before any modified page reaches the data file, a PAGE_WRITE record
describing that page's new state must be fsynced to the WAL file.
This is the classic WAL rule: log first, data second.

On explicit flush() (checkpoint cycle):
  1. All buffered WAL records are fsynced to the WAL file.
  2. Dirty pages are written to the data file.
  3. A CHECKPOINT record is appended and fsynced.
  4. The WAL file is truncated to zero.

This keeps the WAL bounded: it only ever holds changes since the last
flush.  On startup, if the WAL is non-empty, recovery replays it.

Record formats
--------------
PAGE_WRITE  (4113 bytes total):
    [LSN:8][type:1=1][page_id:4][page_bytes:4096][CRC32:4]

CHECKPOINT  (13 bytes total):
    [LSN:8][type:1=2][CRC32:4]

LSN — Log Sequence Number, uint64LE, monotonically increasing.
CRC — zlib.crc32 of everything before the CRC field in the record.
      Detects torn writes at the tail of the log.

Recovery algorithm
------------------
1. Scan WAL records from the beginning.
2. For each PAGE_WRITE, verify CRC.  On mismatch, stop (torn write).
3. Apply the page bytes to the data file.
4. After scanning, truncate the WAL.

Crash-during-recovery is safe because:
* PAGE_WRITE replay is idempotent (writing the same page bytes twice
  produces the same result).
* A torn record at the tail is detected by CRC and replay stops before
  applying a partial record.

pageLSN / flushedLSN protocol
------------------------------
Every in-memory page tracks page.page_lsn — the LSN of the most recent
WAL record that described it.

The WAL tracks _flushed_lsn — the highest LSN safely on disk in the WAL
file after an fsync.

Rule: page.page_lsn must be <= _flushed_lsn before the page is written
to the data file.  Enforced by the pager's eviction path, which calls
wal.fsync_up_to(page.page_lsn) before each page write.
"""

import os
import struct
import zlib


# ---------------------------------------------------------------------------
# Record type constants
# ---------------------------------------------------------------------------

PAGE_WRITE  = 1   # full page image
CHECKPOINT  = 2   # checkpoint marker — WAL truncated after this
META_UPDATE = 3   # root_page_id changed (needed for correct recovery)

# Fixed-size portions of each record type
_PW_FIXED = 8 + 1 + 4          # LSN(8) + type(1) + page_id(4)
_PW_DATA  = 4096                # full page image
_PW_CRC   = 4
PAGE_WRITE_SIZE = _PW_FIXED + _PW_DATA + _PW_CRC   # 4113 bytes

_CP_FIXED = 8 + 1              # LSN(8) + type(1)
_CP_CRC   = 4
CHECKPOINT_SIZE = _CP_FIXED + _CP_CRC               # 13 bytes

_MU_FIXED = 8 + 1 + 4          # LSN(8) + type(1) + root_page_id(4)
_MU_CRC   = 4
META_UPDATE_SIZE = _MU_FIXED + _MU_CRC              # 17 bytes

_NO_ROOT = 0xFFFF_FFFF          # sentinel for root_page_id = None


def _crc(data: bytes | bytearray) -> int:
    return zlib.crc32(data) & 0xFFFF_FFFF


# ---------------------------------------------------------------------------
# WAL
# ---------------------------------------------------------------------------

class WAL:
    """
    Write-Ahead Log backed by a single append-only file.

    Parameters
    ----------
    filepath : str
        Path to the WAL file.  Created if it does not exist.
    """

    def __init__(self, filepath: str):
        self._filepath    = filepath
        self._next_lsn:   int = 1
        self._flushed_lsn: int = 0
        # In-memory buffer: list of (lsn, record_bytes) not yet written to disk
        self._buf: list[tuple[int, bytes]] = []

        if os.path.exists(filepath):
            self._file = open(filepath, 'r+b')
            # Scan existing records to find the highest LSN so we never reuse one.
            self._scan_for_max_lsn()
        else:
            self._file = open(filepath, 'w+b')
        # Always append — seek to end so new records go after existing ones.
        self._file.seek(0, 2)

    # ------------------------------------------------------------------
    # Internal startup helper
    # ------------------------------------------------------------------

    def _scan_for_max_lsn(self):
        """
        Scan the existing WAL file to find the highest valid LSN.

        Called once on open when the file already exists.  Ensures
        _next_lsn starts above any LSN already in the file so we never
        assign a duplicate.  Stops at the first corrupt/partial record
        (same logic as recover).
        """
        self._file.seek(0)
        while True:
            header = self._file.read(9)
            if len(header) < 9:
                break
            lsn, rec_type = struct.unpack('<QB', header)

            if rec_type == PAGE_WRITE:
                rest = self._file.read(4 + _PW_DATA + _PW_CRC)
                if len(rest) < 4 + _PW_DATA + _PW_CRC:
                    break
                payload    = header + rest[:-_PW_CRC]
                crc_stored = struct.unpack_from('<I', rest, -_PW_CRC)[0]
                if _crc(payload) != crc_stored:
                    break
            elif rec_type == META_UPDATE:
                rest = self._file.read(4 + _MU_CRC)
                if len(rest) < 4 + _MU_CRC:
                    break
                payload    = header + rest[:-_MU_CRC]
                crc_stored = struct.unpack_from('<I', rest, -_MU_CRC)[0]
                if _crc(payload) != crc_stored:
                    break
            elif rec_type == CHECKPOINT:
                rest = self._file.read(_CP_CRC)
                if len(rest) < _CP_CRC:
                    break
                payload    = header
                crc_stored = struct.unpack_from('<I', rest, 0)[0]
                if _crc(payload) != crc_stored:
                    break
            else:
                break

            self._next_lsn = max(self._next_lsn, lsn + 1)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_page_write(self, page_id: int, page_bytes: bytes) -> int:
        """
        Buffer a PAGE_WRITE record.

        Does NOT fsync — call fsync() or fsync_up_to() to make it durable.

        Returns the LSN assigned to this record.
        """
        assert len(page_bytes) == 4096, f"page must be 4096 bytes, got {len(page_bytes)}"
        lsn     = self._next_lsn
        self._next_lsn += 1

        payload = struct.pack('<QBI', lsn, PAGE_WRITE, page_id) + page_bytes
        record  = payload + struct.pack('<I', _crc(payload))
        self._buf.append((lsn, record))
        return lsn

    def append_meta_update(self, root_page_id: int | None) -> int:
        """
        Buffer a META_UPDATE record capturing the current root_page_id.

        Called whenever the tree changes the root (split or collapse).
        During recovery, the last META_UPDATE record wins, restoring the
        correct root so the tree can navigate from the right page.
        """
        lsn     = self._next_lsn
        self._next_lsn += 1

        root    = _NO_ROOT if root_page_id is None else root_page_id
        payload = struct.pack('<QBI', lsn, META_UPDATE, root)
        record  = payload + struct.pack('<I', _crc(payload))
        self._buf.append((lsn, record))
        return lsn

    def append_checkpoint(self) -> int:
        """
        Buffer a CHECKPOINT record.

        Typically called by checkpoint() after all dirty pages are on disk.
        """
        lsn     = self._next_lsn
        self._next_lsn += 1

        payload = struct.pack('<QB', lsn, CHECKPOINT)
        record  = payload + struct.pack('<I', _crc(payload))
        self._buf.append((lsn, record))
        return lsn

    # ------------------------------------------------------------------
    # Fsync
    # ------------------------------------------------------------------

    def fsync_up_to(self, lsn: int):
        """
        Write all buffered records with LSN <= lsn to disk and fsync.

        After this returns, every record up to lsn is guaranteed durable.
        _flushed_lsn is updated to reflect the new high-water mark.
        """
        to_write  = []
        remaining = []
        for rec_lsn, rec_bytes in self._buf:
            if rec_lsn <= lsn:
                to_write.append(rec_bytes)
            else:
                remaining.append((rec_lsn, rec_bytes))

        if to_write:
            self._file.write(b''.join(to_write))
            self._file.flush()
            os.fsync(self._file.fileno())
            self._flushed_lsn = max(self._flushed_lsn, lsn)

        self._buf = remaining

    def fsync(self):
        """Write and fsync ALL buffered records to the WAL file."""
        if self._buf:
            self.fsync_up_to(self._buf[-1][0])

    @property
    def flushed_lsn(self) -> int:
        """Highest LSN confirmed durable in the WAL file."""
        return self._flushed_lsn

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def checkpoint(self):
        """
        Write a CHECKPOINT record, fsync, then truncate the WAL to zero.

        The caller (BPlusPager.flush) is responsible for writing all dirty
        pages to the data file BEFORE calling this.

        After this returns:
        * The WAL file is empty (zero bytes).
        * All previously logged page changes are in the data file.
        * On next startup no recovery is needed.
        """
        self.append_checkpoint()
        self.fsync()
        # Truncate: all changes are now in the data file
        self._file.seek(0)
        self._file.truncate(0)
        self._file.flush()
        os.fsync(self._file.fileno())

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover(self, write_page_fn) -> dict:
        """
        Replay WAL records from the beginning of the file.

        For each valid PAGE_WRITE record, write_page_fn(page_id, page_bytes)
        is called.  For each META_UPDATE record, the root_page_id is tracked.
        Replay stops at the first corrupt CRC (torn write) or EOF.

        Returns a dict:
            {
              "max_page_id":  int,        # highest page_id seen (-1 if none)
              "root_page_id": int | None, # last root seen (None if no META_UPDATE)
              "had_meta":     bool,       # True if any META_UPDATE was replayed
            }
        The caller uses this to fix up stale meta page state after recovery.
        """
        self._file.seek(0)
        max_page_id  = -1
        root_page_id = None
        had_meta     = False

        while True:
            # Read the fixed LSN + type header (9 bytes)
            header = self._file.read(9)
            if len(header) < 9:
                break   # EOF

            lsn, rec_type = struct.unpack('<QB', header)

            if rec_type == PAGE_WRITE:
                rest = self._file.read(4 + _PW_DATA + _PW_CRC)
                if len(rest) < 4 + _PW_DATA + _PW_CRC:
                    break   # partial record — torn write, stop here

                payload    = header + rest[:-_PW_CRC]
                crc_stored = struct.unpack_from('<I', rest, -_PW_CRC)[0]
                if _crc(payload) != crc_stored:
                    break

                page_id    = struct.unpack_from('<I', rest, 0)[0]
                page_bytes = rest[4: 4 + _PW_DATA]
                write_page_fn(page_id, page_bytes)
                max_page_id = max(max_page_id, page_id)

            elif rec_type == META_UPDATE:
                rest = self._file.read(4 + _MU_CRC)   # root_page_id(4) + crc(4)
                if len(rest) < 4 + _MU_CRC:
                    break

                payload    = header + rest[:-_MU_CRC]
                crc_stored = struct.unpack_from('<I', rest, -_MU_CRC)[0]
                if _crc(payload) != crc_stored:
                    break

                root       = struct.unpack_from('<I', rest, 0)[0]
                root_page_id = None if root == _NO_ROOT else root
                had_meta   = True

            elif rec_type == CHECKPOINT:
                rest = self._file.read(_CP_CRC)
                if len(rest) < _CP_CRC:
                    break
                payload    = header
                crc_stored = struct.unpack_from('<I', rest, 0)[0]
                if _crc(payload) != crc_stored:
                    break
                # CHECKPOINT is a no-op during replay

            else:
                break   # Unknown record type — treat as corruption

            self._next_lsn = max(self._next_lsn, lsn + 1)

        return {"max_page_id": max_page_id, "root_page_id": root_page_id,
                "had_meta": had_meta}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """True when the WAL file has no records (e.g., right after a checkpoint)."""
        current = self._file.seek(0, 1)   # save position
        self._file.seek(0, 2)
        size = self._file.tell()
        self._file.seek(current)
        return size == 0 and len(self._buf) == 0

    def close(self):
        """Flush any buffered records and close the WAL file."""
        self.fsync()
        self._file.close()
