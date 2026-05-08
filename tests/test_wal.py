"""
Tests for the WAL (Write-Ahead Log).

Covers:
  - Record appending and LSN sequencing
  - fsync_up_to / fsync
  - Checkpoint (write + truncate)
  - Recovery: full replay, partial torn record, empty WAL
  - pageLSN / flushedLSN enforcement via BPlusPager
  - Crash simulation: process dies before flush, data survives via replay
"""

import os
import struct
import zlib
import pytest

from src.wal         import WAL, PAGE_WRITE, CHECKPOINT, PAGE_WRITE_SIZE, CHECKPOINT_SIZE
from src.bplus_page  import LeafPage, IndexPage
from src.bplus_pager import BPlusPager
from src.bplus_tree  import BPlusTree
from src.record      import encode_record, decode_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wal(tmp_path, name="test.wal") -> WAL:
    return WAL(str(tmp_path / name))


def fake_page_bytes(page_id: int, marker: int = 0xAB) -> bytes:
    """Return 4096 bytes where the first 4 bytes encode page_id."""
    buf = bytearray(4096)
    struct.pack_into('<I', buf, 0, page_id)
    buf[4] = marker
    return bytes(buf)


# ---------------------------------------------------------------------------
# LSN sequencing
# ---------------------------------------------------------------------------

def test_lsn_starts_at_one(tmp_path):
    wal = make_wal(tmp_path)
    lsn = wal.append_page_write(0, fake_page_bytes(0))
    assert lsn == 1
    wal.close()


def test_lsn_increments(tmp_path):
    wal = make_wal(tmp_path)
    lsns = [wal.append_page_write(i, fake_page_bytes(i)) for i in range(5)]
    assert lsns == [1, 2, 3, 4, 5]
    wal.close()


def test_checkpoint_lsn_increments(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    lsn = wal.append_checkpoint()
    assert lsn == 2
    wal.close()


# ---------------------------------------------------------------------------
# flushedLSN starts at 0 and advances after fsync
# ---------------------------------------------------------------------------

def test_flushed_lsn_zero_before_fsync(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    assert wal.flushed_lsn == 0
    wal.close()


def test_flushed_lsn_advances_after_fsync_up_to(tmp_path):
    wal = make_wal(tmp_path)
    lsn = wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync_up_to(lsn)
    assert wal.flushed_lsn == lsn
    wal.close()


def test_flushed_lsn_advances_after_fsync(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_page_write(1, fake_page_bytes(1))
    wal.fsync()
    assert wal.flushed_lsn == 2
    wal.close()


def test_fsync_up_to_partial(tmp_path):
    """fsync_up_to(1) flushes only record 1; record 2 stays buffered."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))   # LSN 1
    wal.append_page_write(1, fake_page_bytes(1))   # LSN 2
    wal.fsync_up_to(1)
    assert wal.flushed_lsn == 1
    # Record 2 is still in the buffer (not yet on disk)
    # Confirm by checking file size: only one PAGE_WRITE record on disk
    path = str(tmp_path / "test.wal")
    assert os.path.getsize(path) == PAGE_WRITE_SIZE
    wal.close()


# ---------------------------------------------------------------------------
# Checkpoint — truncates WAL to zero
# ---------------------------------------------------------------------------

def test_checkpoint_truncates_wal(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.checkpoint()
    path = str(tmp_path / "test.wal")
    assert os.path.getsize(path) == 0
    wal.close()


def test_is_empty_after_checkpoint(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.checkpoint()
    assert wal.is_empty()
    wal.close()


def test_is_empty_on_fresh_wal(tmp_path):
    wal = make_wal(tmp_path)
    assert wal.is_empty()
    wal.close()


# ---------------------------------------------------------------------------
# Recovery — full replay
# ---------------------------------------------------------------------------

def test_recover_replays_page_writes(tmp_path):
    wal = make_wal(tmp_path)
    pages_written = {i: fake_page_bytes(i, marker=i) for i in range(3)}
    for pid, pb in pages_written.items():
        wal.append_page_write(pid, pb)
    wal.fsync()

    replayed = {}
    wal.recover(lambda pid, pb: replayed.update({pid: pb}))

    assert replayed == pages_written
    wal.close()


def test_recover_returns_max_page_id(tmp_path):
    wal = make_wal(tmp_path)
    for i in [0, 5, 3, 9, 2]:
        wal.append_page_write(i, fake_page_bytes(i))
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["max_page_id"] == 9
    wal.close()


def test_recover_empty_wal_returns_minus_one(tmp_path):
    wal = make_wal(tmp_path)
    info = wal.recover(lambda *_: None)
    assert info["max_page_id"] == -1
    wal.close()


def test_recover_skips_checkpoint_records(tmp_path):
    """CHECKPOINT records are valid but produce no replay actions."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_checkpoint()
    wal.append_page_write(1, fake_page_bytes(1))
    wal.fsync()

    replayed = []
    wal.recover(lambda pid, pb: replayed.append(pid))
    assert replayed == [0, 1]
    wal.close()


def test_recover_meta_update(tmp_path):
    """META_UPDATE records are replayed and root_page_id is restored."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_meta_update(root_page_id=5)
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is True
    assert info["root_page_id"] == 5
    wal.close()


def test_recover_meta_update_none_root(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_meta_update(root_page_id=None)
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is True
    assert info["root_page_id"] is None
    wal.close()


def test_recover_no_meta_update(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is False
    wal.close()


# ---------------------------------------------------------------------------
# Recovery — torn write detection
# ---------------------------------------------------------------------------

def test_recover_stops_at_corrupt_crc(tmp_path):
    """Corrupting the CRC of a record makes recovery stop before it."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))   # LSN 1 — good
    wal.append_page_write(1, fake_page_bytes(1))   # LSN 2 — will be corrupted
    wal.fsync()
    wal.close()

    # Corrupt the CRC of the second record (last 4 bytes of it)
    with open(path, 'r+b') as f:
        f.seek(PAGE_WRITE_SIZE - 4)    # end of record 1
        f.seek(PAGE_WRITE_SIZE + PAGE_WRITE_SIZE - 4)  # CRC of record 2
        f.write(b'\xFF\xFF\xFF\xFF')

    wal2 = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]   # record 2 was discarded
    wal2.close()


def test_recover_stops_at_partial_record(tmp_path):
    """A partial record at the tail (torn write) is silently discarded."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.close()

    # Append a partial (truncated) PAGE_WRITE record
    with open(path, 'ab') as f:
        f.write(b'\x00' * 100)   # far too short for a full record

    wal2 = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]   # partial tail record silently discarded
    wal2.close()


# ---------------------------------------------------------------------------
# pageLSN / flushedLSN enforcement
# ---------------------------------------------------------------------------

def test_page_lsn_set_after_mark_dirty(tmp_path):
    path = str(tmp_path / "db.db")
    with BPlusPager(path, order=4) as pager:
        leaf = pager.new_leaf_page()
        assert leaf.page_lsn == 0   # before any dirty mark

        leaf.insert(1, b"hello")
        pager.mark_dirty(leaf)
        assert leaf.page_lsn > 0   # WAL assigned an LSN


def test_flushed_lsn_advances_on_flush(tmp_path):
    path = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    leaf.insert(1, b"hello")
    pager.mark_dirty(leaf)

    # WAL buffer has records but flushed_lsn is still 0
    assert pager._wal.flushed_lsn == 0

    pager.flush()

    # After flush, WAL is checkpointed (truncated) and flushed_lsn advanced
    assert pager._wal.flushed_lsn > 0
    pager.close()


# ---------------------------------------------------------------------------
# End-to-end crash simulation
# ---------------------------------------------------------------------------

def test_wal_recovery_after_simulated_crash(tmp_path):
    """
    Simulate a crash: insert data, fsync WAL but do NOT flush data pages,
    then close without calling flush().  On reopen, WAL replay must restore
    all inserts.
    """
    path = str(tmp_path / "crash.db")
    n    = 30

    # Open pager + tree, insert data, force WAL to disk, then 'crash'
    # (close without flush — data pages never hit the data file).
    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k, f"v{k}"]))

    # Force WAL to disk so the records survive the simulated crash.
    pager._wal.fsync()

    # Simulate crash: close files without calling flush()
    pager._file.close()
    pager._wal._file.close()

    # Reopen — recovery should replay the WAL
    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(n):
            raw = tree2.search(k)
            assert raw is not None, f"key {k} missing after crash recovery"
            assert decode_record(raw) == [k, f"v{k}"]


def test_wal_is_empty_after_clean_flush(tmp_path):
    """After a normal flush, the WAL file should be empty (checkpointed)."""
    path = str(tmp_path / "clean.db")
    with BPlusPager(path, order=4) as pager:
        tree = BPlusTree(pager)
        for k in range(20):
            tree.insert(k, encode_record([k]))
        # flush() is called by close() via __exit__

    wal_path = path + '.wal'
    assert os.path.exists(wal_path)
    assert os.path.getsize(wal_path) == 0


def test_wal_recovery_large_dataset(tmp_path):
    """WAL recovery works correctly for a large tree spanning many pages."""
    path = str(tmp_path / "large_crash.db")
    n    = 200

    pager = BPlusPager(path, order=10)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k, f"name_{k}"]))

    pager._wal.fsync()
    pager._file.close()
    pager._wal._file.close()

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(n):
            assert decode_record(tree2.search(k)) == [k, f"name_{k}"]


def test_no_wal_file_created_for_in_memory_pager():
    """In-memory pager must not create any WAL file."""
    pager = BPlusPager()   # no filepath
    assert pager._wal is None
    pager.close()
