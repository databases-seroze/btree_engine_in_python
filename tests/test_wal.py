"""
Tests for the WAL (Write-Ahead Log).

Sections
--------
1.  Record format on disk        — exact byte layout matches the spec
2.  LSN sequencing               — monotonic, no reuse across reopen
3.  flushedLSN / fsync_up_to     — buffer drain, partial flush, idempotency
4.  Checkpoint                   — truncation, is_empty, post-checkpoint writes
5.  Recovery — happy path        — full replay, order, last-write-wins
6.  Recovery — META_UPDATE       — root_page_id restored, last value wins
7.  Recovery — corruption        — bad CRC on body, bad CRC on CRC field,
                                   unknown type byte, partial record
8.  Recovery — idempotency       — calling recover twice replays the same data
9.  pageLSN / flushedLSN         — stamped by mark_dirty, enforced by flush
10. End-to-end crash simulations — insert, delete, root-split, root-collapse
11. Sequential flush cycles      — WAL stays bounded across multiple flush()s
"""

import os
import struct
import zlib
import pytest

from src.wal         import (WAL, PAGE_WRITE, CHECKPOINT, META_UPDATE,
                              PAGE_WRITE_SIZE, CHECKPOINT_SIZE, META_UPDATE_SIZE)
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
    """4096 bytes: first 4 encode page_id, byte 4 is a marker."""
    buf = bytearray(4096)
    struct.pack_into('<I', buf, 0, page_id)
    buf[4] = marker & 0xFF
    return bytes(buf)


def _crc(data: bytes | bytearray) -> int:
    return zlib.crc32(data) & 0xFFFF_FFFF


def crash_pager(pager):
    """Close file handles without calling flush() — simulates a process crash."""
    pager._file.close()
    pager._wal._file.close()


# ===========================================================================
# 1. Record format on disk
# ===========================================================================

def test_page_write_record_size(tmp_path):
    """A PAGE_WRITE record is exactly PAGE_WRITE_SIZE bytes on disk."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.close()
    assert os.path.getsize(path) == PAGE_WRITE_SIZE


def test_checkpoint_record_size(tmp_path):
    """A CHECKPOINT record appended after a PAGE_WRITE grows the file correctly."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_checkpoint()
    wal.fsync()
    wal.close()
    assert os.path.getsize(path) == PAGE_WRITE_SIZE + CHECKPOINT_SIZE


def test_meta_update_record_size(tmp_path):
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_meta_update(root_page_id=3)
    wal.fsync()
    wal.close()
    assert os.path.getsize(path) == META_UPDATE_SIZE


def test_page_write_byte_layout(tmp_path):
    """Verify LSN, type, page_id, and CRC are at the documented offsets."""
    path    = str(tmp_path / "test.wal")
    pb      = fake_page_bytes(42, marker=0xDE)
    wal     = make_wal(tmp_path)
    lsn     = wal.append_page_write(42, pb)
    wal.fsync()
    wal.close()

    with open(path, 'rb') as f:
        raw = f.read()

    assert len(raw) == PAGE_WRITE_SIZE

    lsn_on_disk  = struct.unpack_from('<Q', raw, 0)[0]
    type_on_disk = raw[8]
    pid_on_disk  = struct.unpack_from('<I', raw, 9)[0]
    body_on_disk = raw[13 : 13 + 4096]
    crc_on_disk  = struct.unpack_from('<I', raw, PAGE_WRITE_SIZE - 4)[0]

    assert lsn_on_disk  == lsn
    assert type_on_disk == PAGE_WRITE
    assert pid_on_disk  == 42
    assert body_on_disk == pb
    assert crc_on_disk  == _crc(raw[: PAGE_WRITE_SIZE - 4])


def test_meta_update_byte_layout(tmp_path):
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    lsn  = wal.append_meta_update(root_page_id=7)
    wal.fsync()
    wal.close()

    with open(path, 'rb') as f:
        raw = f.read()

    lsn_on_disk  = struct.unpack_from('<Q', raw, 0)[0]
    type_on_disk = raw[8]
    root_on_disk = struct.unpack_from('<I', raw, 9)[0]
    crc_on_disk  = struct.unpack_from('<I', raw, META_UPDATE_SIZE - 4)[0]

    assert lsn_on_disk  == lsn
    assert type_on_disk == META_UPDATE
    assert root_on_disk == 7
    assert crc_on_disk  == _crc(raw[: META_UPDATE_SIZE - 4])


def test_crc_covers_lsn_type_pageid_body(tmp_path):
    """CRC must be computed over the full payload (LSN + type + page_id + body)."""
    path = str(tmp_path / "test.wal")
    pb   = fake_page_bytes(1, marker=0xAA)
    wal  = make_wal(tmp_path)
    wal.append_page_write(1, pb)
    wal.fsync()
    wal.close()

    with open(path, 'rb') as f:
        raw = f.read()

    # Flip a single bit in the LSN field (offset 0) — CRC must change
    tampered = bytearray(raw)
    tampered[0] ^= 0x01
    stored_crc   = struct.unpack_from('<I', raw, PAGE_WRITE_SIZE - 4)[0]
    tampered_crc = _crc(bytes(tampered)[: PAGE_WRITE_SIZE - 4])
    assert stored_crc != tampered_crc


# ===========================================================================
# 2. LSN sequencing
# ===========================================================================

def test_lsn_starts_at_one(tmp_path):
    wal = make_wal(tmp_path)
    lsn = wal.append_page_write(0, fake_page_bytes(0))
    assert lsn == 1
    wal.close()


def test_lsn_increments_across_record_types(tmp_path):
    wal  = make_wal(tmp_path)
    lsn1 = wal.append_page_write(0, fake_page_bytes(0))
    lsn2 = wal.append_meta_update(root_page_id=0)
    lsn3 = wal.append_checkpoint()
    lsn4 = wal.append_page_write(1, fake_page_bytes(1))
    assert [lsn1, lsn2, lsn3, lsn4] == [1, 2, 3, 4]
    wal.close()


def test_lsn_not_reused_after_reopen(tmp_path):
    """Reopening the WAL must resume from where it left off, not restart at 1."""
    path = str(tmp_path / "test.wal")
    wal1 = WAL(path)
    lsn1 = wal1.append_page_write(0, fake_page_bytes(0))
    wal1.fsync()
    wal1.close()

    wal2 = WAL(path)
    lsn2 = wal2.append_page_write(1, fake_page_bytes(1))
    assert lsn2 > lsn1, "reopened WAL must not reuse LSNs"
    wal2.close()


def test_lsn_not_reused_after_checkpoint(tmp_path):
    """Even after checkpoint (WAL truncated), LSN must keep incrementing."""
    path = str(tmp_path / "test.wal")
    wal  = WAL(path)
    wal.append_page_write(0, fake_page_bytes(0))  # LSN 1
    wal.fsync()
    wal.checkpoint()   # truncates — but _next_lsn must not reset

    lsn = wal.append_page_write(1, fake_page_bytes(1))
    assert lsn > 1
    wal.close()


# ===========================================================================
# 3. flushedLSN / fsync_up_to
# ===========================================================================

def test_flushed_lsn_zero_before_any_fsync(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    assert wal.flushed_lsn == 0
    wal.close()


def test_flushed_lsn_advances_to_target(tmp_path):
    wal = make_wal(tmp_path)
    lsn = wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync_up_to(lsn)
    assert wal.flushed_lsn == lsn
    wal.close()


def test_fsync_up_to_does_not_flush_higher_lsns(tmp_path):
    """Records with LSN > target stay in the buffer after fsync_up_to."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))   # LSN 1
    wal.append_page_write(1, fake_page_bytes(1))   # LSN 2
    wal.fsync_up_to(1)
    # Only LSN 1 is on disk
    assert os.path.getsize(path) == PAGE_WRITE_SIZE
    assert wal.flushed_lsn == 1
    wal.close()


def test_fsync_up_to_is_noop_when_already_flushed(tmp_path):
    """Calling fsync_up_to with lsn <= flushed_lsn must not re-write."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    lsn  = wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync_up_to(lsn)
    size_before = os.path.getsize(path)
    wal.fsync_up_to(lsn)   # second call — nothing buffered
    assert os.path.getsize(path) == size_before
    wal.close()


def test_fsync_flushes_all_buffered(tmp_path):
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    for i in range(4):
        wal.append_page_write(i, fake_page_bytes(i))
    wal.fsync()
    assert os.path.getsize(path) == PAGE_WRITE_SIZE * 4
    assert wal.flushed_lsn == 4
    wal.close()


def test_fsync_on_empty_buffer_is_noop(tmp_path):
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.fsync()   # nothing buffered
    assert os.path.getsize(path) == 0
    wal.close()


# ===========================================================================
# 4. Checkpoint
# ===========================================================================

def test_checkpoint_truncates_to_zero(tmp_path):
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    for i in range(5):
        wal.append_page_write(i, fake_page_bytes(i))
    wal.fsync()
    wal.checkpoint()
    assert os.path.getsize(path) == 0
    wal.close()


def test_is_empty_after_checkpoint(tmp_path):
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.checkpoint()
    assert wal.is_empty()
    wal.close()


def test_is_empty_fresh_wal(tmp_path):
    wal = make_wal(tmp_path)
    assert wal.is_empty()
    wal.close()


def test_is_not_empty_with_buffered_records(tmp_path):
    """WAL with only buffered (unfsynced) records is not empty."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    assert not wal.is_empty()
    wal.close()


def test_writes_after_checkpoint_work(tmp_path):
    """After a checkpoint the WAL must accept new records normally."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.checkpoint()

    lsn = wal.append_page_write(1, fake_page_bytes(1))
    wal.fsync()
    assert os.path.getsize(path) == PAGE_WRITE_SIZE
    assert lsn > 1   # LSN kept incrementing
    wal.close()


def test_multiple_checkpoints(tmp_path):
    """Calling checkpoint twice in a row is safe — second is a no-op write."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.checkpoint()
    wal.checkpoint()   # nothing buffered, should not raise
    assert os.path.getsize(path) == 0
    wal.close()


# ===========================================================================
# 5. Recovery — happy path
# ===========================================================================

def test_recover_all_page_writes(tmp_path):
    wal          = make_wal(tmp_path)
    pages        = {i: fake_page_bytes(i, marker=i) for i in range(5)}
    for pid, pb  in pages.items():
        wal.append_page_write(pid, pb)
    wal.fsync()

    replayed = {}
    wal.recover(lambda pid, pb: replayed.update({pid: pb}))
    assert replayed == pages
    wal.close()


def test_recover_replay_order_is_sequential(tmp_path):
    """Pages must be replayed in LSN order (insertion order)."""
    wal   = make_wal(tmp_path)
    order = [3, 1, 4, 1, 5]   # deliberately out of page-id order
    for pid in order:
        wal.append_page_write(pid, fake_page_bytes(pid))
    wal.fsync()

    replayed_order = []
    wal.recover(lambda pid, _: replayed_order.append(pid))
    assert replayed_order == order
    wal.close()


def test_recover_last_write_wins_for_same_page(tmp_path):
    """Multiple PAGE_WRITE records for the same page_id: last one wins."""
    pb_v1 = fake_page_bytes(0, marker=0x11)
    pb_v2 = fake_page_bytes(0, marker=0x22)
    wal   = make_wal(tmp_path)
    wal.append_page_write(0, pb_v1)
    wal.append_page_write(0, pb_v2)
    wal.fsync()

    seen = []
    wal.recover(lambda pid, pb: seen.append(pb))
    assert len(seen) == 2
    assert seen[-1] == pb_v2   # last written version is the final state
    wal.close()


def test_recover_returns_correct_max_page_id(tmp_path):
    wal = make_wal(tmp_path)
    for pid in [0, 7, 3, 15, 2]:
        wal.append_page_write(pid, fake_page_bytes(pid))
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["max_page_id"] == 15
    wal.close()


def test_recover_empty_wal(tmp_path):
    wal  = make_wal(tmp_path)
    info = wal.recover(lambda *_: None)
    assert info["max_page_id"] == -1
    assert info["had_meta"] is False
    wal.close()


def test_recover_checkpoint_in_middle_does_not_stop_replay(tmp_path):
    """A CHECKPOINT record mid-WAL is a no-op; records after it are still replayed."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_checkpoint()
    wal.append_page_write(1, fake_page_bytes(1))
    wal.fsync()

    replayed = []
    wal.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0, 1]
    wal.close()


# ===========================================================================
# 6. Recovery — META_UPDATE
# ===========================================================================

def test_recover_meta_update_restores_root(tmp_path):
    wal  = make_wal(tmp_path)
    wal.append_meta_update(root_page_id=5)
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is True
    assert info["root_page_id"] == 5
    wal.close()


def test_recover_meta_update_last_value_wins(tmp_path):
    """If META_UPDATE appears multiple times, the last one determines root."""
    wal = make_wal(tmp_path)
    wal.append_meta_update(root_page_id=1)
    wal.append_meta_update(root_page_id=3)
    wal.append_meta_update(root_page_id=7)
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["root_page_id"] == 7
    wal.close()


def test_recover_meta_update_none_root(tmp_path):
    wal  = make_wal(tmp_path)
    wal.append_meta_update(root_page_id=None)
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is True
    assert info["root_page_id"] is None
    wal.close()


def test_recover_no_meta_update_had_meta_false(tmp_path):
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()

    info = wal.recover(lambda *_: None)
    assert info["had_meta"] is False
    wal.close()


def test_recover_meta_update_mixed_with_page_writes(tmp_path):
    """META_UPDATE interleaved with PAGE_WRITEs — both are handled correctly."""
    wal = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.append_meta_update(root_page_id=0)
    wal.append_page_write(1, fake_page_bytes(1))
    wal.append_meta_update(root_page_id=2)
    wal.fsync()

    replayed = []
    info = wal.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0, 1]
    assert info["root_page_id"] == 2
    wal.close()


# ===========================================================================
# 7. Recovery — corruption / torn writes
# ===========================================================================

def test_recover_stops_at_corrupt_crc_field(tmp_path):
    """Flipping the CRC bytes of a record stops replay before that record."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))   # good
    wal.append_page_write(1, fake_page_bytes(1))   # CRC will be corrupted
    wal.fsync()
    wal.close()

    with open(path, 'r+b') as f:
        f.seek(PAGE_WRITE_SIZE * 2 - 4)   # CRC of record 2
        f.write(b'\xFF\xFF\xFF\xFF')

    wal2     = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]
    wal2.close()


def test_recover_stops_at_corrupt_page_body(tmp_path):
    """Flipping a bit in the page body also invalidates the CRC."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))   # good
    wal.append_page_write(1, fake_page_bytes(1))   # body will be corrupted
    wal.fsync()
    wal.close()

    with open(path, 'r+b') as f:
        # Flip a byte in the page body of record 2 (offset 13 inside record)
        f.seek(PAGE_WRITE_SIZE + 13)
        f.write(b'\xFF')

    wal2     = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]
    wal2.close()


def test_recover_stops_at_partial_record(tmp_path):
    """A truncated record at the tail is silently discarded."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.close()

    with open(path, 'ab') as f:
        f.write(b'\x01' * 50)   # partial PAGE_WRITE (not full 4113 bytes)

    wal2     = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]
    wal2.close()


def test_recover_stops_at_unknown_type_byte(tmp_path):
    """An unknown record type byte stops replay cleanly."""
    path = str(tmp_path / "test.wal")
    wal  = make_wal(tmp_path)
    wal.append_page_write(0, fake_page_bytes(0))
    wal.fsync()
    wal.close()

    with open(path, 'ab') as f:
        # Write a record with an unknown type byte (0x42)
        fake_header = struct.pack('<QB', 999, 0x42) + b'\x00' * 100
        f.write(fake_header)

    wal2     = WAL(path)
    replayed = []
    wal2.recover(lambda pid, _: replayed.append(pid))
    assert replayed == [0]   # stops before the unknown type
    wal2.close()


def test_recover_empty_file_after_corruption(tmp_path):
    """WAL that only contains garbage bytes replays nothing safely."""
    path = str(tmp_path / "test.wal")
    with open(path, 'wb') as f:
        f.write(b'\xDE\xAD\xBE\xEF' * 1024)

    wal  = WAL(path)
    info = wal.recover(lambda *_: None)
    assert info["max_page_id"] == -1
    wal.close()


# ===========================================================================
# 8. Recovery — idempotency
# ===========================================================================

def test_recover_is_idempotent(tmp_path):
    """Calling recover twice on the same WAL replays the same pages both times."""
    wal   = make_wal(tmp_path)
    pages = {i: fake_page_bytes(i) for i in range(3)}
    for pid, pb in pages.items():
        wal.append_page_write(pid, pb)
    wal.fsync()

    seen1 = {}
    wal.recover(lambda pid, pb: seen1.update({pid: pb}))

    # Recover again from the same file (seek back to start internally)
    seen2 = {}
    wal.recover(lambda pid, pb: seen2.update({pid: pb}))

    assert seen1 == seen2 == pages
    wal.close()


def test_recover_after_crash_during_recovery(tmp_path):
    """
    Simulate a crash mid-recovery: the data file gets partial writes.
    On the next startup, recovery replays again from scratch and produces
    the correct final state (idempotency makes this safe).
    """
    path = str(tmp_path / "db.db")
    n    = 20

    # Phase 1: write data, crash without flush
    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k, f"v{k}"]))
    pager._wal.fsync()
    crash_pager(pager)

    # Phase 2: simulate crash during first recovery attempt
    #   — open, run recovery, then crash again immediately
    pager2 = BPlusPager(path, order=4)   # recovery runs on open
    crash_pager(pager2)

    # Phase 3: open again — should still be correct
    with BPlusPager(path) as pager3:
        tree3 = BPlusTree(pager3)
        for k in range(n):
            assert decode_record(tree3.search(k)) == [k, f"v{k}"]


# ===========================================================================
# 9. pageLSN / flushedLSN via BPlusPager
# ===========================================================================

def test_page_lsn_is_zero_before_mark_dirty(tmp_path):
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    assert leaf.page_lsn == 0
    pager.close()


def test_page_lsn_set_after_mark_dirty(tmp_path):
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    leaf.insert(1, b"v")
    pager.mark_dirty(leaf)
    assert leaf.page_lsn > 0
    pager.close()


def test_page_lsn_increases_monotonically(tmp_path):
    """Each successive mark_dirty must assign a strictly higher LSN."""
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()

    prev_lsn = 0
    for i in range(5):
        leaf.insert(i, f"v{i}".encode())
        pager.mark_dirty(leaf)
        assert leaf.page_lsn > prev_lsn
        prev_lsn = leaf.page_lsn
    pager.close()


def test_flushed_lsn_zero_before_flush(tmp_path):
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    leaf.insert(1, b"v")
    pager.mark_dirty(leaf)
    assert pager._wal.flushed_lsn == 0
    pager.close()


def test_flushed_lsn_advances_after_flush(tmp_path):
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    leaf.insert(1, b"v")
    pager.mark_dirty(leaf)
    pager.flush()
    assert pager._wal.flushed_lsn > 0
    pager.close()


def test_page_lsn_serialised_and_restored(tmp_path):
    """page_lsn must survive a round-trip through to_bytes / from_bytes."""
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    leaf  = pager.new_leaf_page()
    leaf.insert(99, b"hello")
    pager.mark_dirty(leaf)
    lsn_before = leaf.page_lsn

    pager.flush()   # writes page to disk
    pager.close()

    pager2 = BPlusPager(path, order=4)
    leaf2  = pager2.get_page(leaf.page_id)
    assert leaf2.page_lsn == lsn_before
    pager2.close()


def test_no_wal_for_in_memory_pager():
    pager = BPlusPager()
    assert pager._wal is None
    pager.close()


# ===========================================================================
# 10. End-to-end crash simulations
# ===========================================================================

def test_crash_after_inserts(tmp_path):
    """Crash after inserting N keys; recovery restores all of them."""
    path = str(tmp_path / "db.db")
    n    = 40

    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k, f"v{k}"]))
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(n):
            raw = tree2.search(k)
            assert raw is not None, f"key {k} missing"
            assert decode_record(raw) == [k, f"v{k}"]


def test_crash_after_deletes(tmp_path):
    """Crash after inserting and deleting some keys; recovery reflects the deletes."""
    path = str(tmp_path / "db.db")
    n    = 30
    to_delete = set(range(0, n, 3))   # delete every 3rd key

    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k]))
    for k in to_delete:
        tree.delete(k)
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(n):
            raw = tree2.search(k)
            if k in to_delete:
                assert raw is None, f"deleted key {k} should be absent"
            else:
                assert raw is not None, f"key {k} should be present"


def test_crash_during_root_split(tmp_path):
    """
    Crash just after a root split.  META_UPDATE records the new root so
    recovery can navigate from the correct root page.
    """
    path = str(tmp_path / "db.db")

    pager = BPlusPager(path, order=2)   # order=2 forces early splits
    tree  = BPlusTree(pager)
    for k in range(10):
        tree.insert(k, encode_record([k]))
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(10):
            assert decode_record(tree2.search(k)) == [k]


def test_crash_during_root_collapse(tmp_path):
    """
    Crash just after deleting enough keys to collapse the root back to a
    leaf.  META_UPDATE must record the new (leaf) root.
    """
    path = str(tmp_path / "db.db")

    # order=2: insert 3 keys to get a 2-level tree, then delete until collapse
    pager = BPlusPager(path, order=2)
    tree  = BPlusTree(pager)
    for k in [1, 2, 3]:
        tree.insert(k, encode_record([k]))
    # After collapse the root should be a LeafPage
    tree.delete(3)
    tree.delete(2)
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2  = BPlusTree(pager2)
        root   = pager2.get_page(pager2.root_page_id)
        assert isinstance(root, LeafPage), "root should have collapsed to a leaf"
        assert decode_record(tree2.search(1)) == [1]
        assert tree2.search(2) is None
        assert tree2.search(3) is None


def test_wal_empty_after_clean_close(tmp_path):
    """After a normal close(), the WAL file is empty (checkpointed)."""
    path = str(tmp_path / "db.db")
    with BPlusPager(path, order=4) as pager:
        tree = BPlusTree(pager)
        for k in range(20):
            tree.insert(k, encode_record([k]))

    assert os.path.getsize(path + '.wal') == 0


def test_crash_with_large_dataset(tmp_path):
    """Recovery works correctly for a large tree spanning many index levels."""
    path = str(tmp_path / "db.db")
    n    = 300

    pager = BPlusPager(path, order=10)
    tree  = BPlusTree(pager)
    for k in range(n):
        tree.insert(k, encode_record([k, f"name_{k}"]))
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(n):
            assert decode_record(tree2.search(k)) == [k, f"name_{k}"]


# ===========================================================================
# 11. Sequential flush cycles
# ===========================================================================

def test_wal_empty_after_each_flush(tmp_path):
    """WAL must be empty after every flush, not just the first one."""
    path  = str(tmp_path / "db.db")
    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)

    for batch in range(3):
        for k in range(batch * 10, (batch + 1) * 10):
            tree.insert(k, encode_record([k]))
        pager.flush()
        assert os.path.getsize(path + '.wal') == 0, \
            f"WAL not empty after flush #{batch + 1}"

    pager.close()


def test_data_survives_multiple_flush_reopen_cycles(tmp_path):
    """Data written across multiple flush cycles is all readable on reopen."""
    path  = str(tmp_path / "db.db")
    total = 0

    with BPlusPager(path, order=4) as pager:
        tree = BPlusTree(pager)
        for k in range(20):
            tree.insert(k, encode_record([k, "first"]))
        pager.flush()
        total += 20

    with BPlusPager(path) as pager:
        tree = BPlusTree(pager)
        for k in range(20, 40):
            tree.insert(k, encode_record([k, "second"]))
        pager.flush()
        total += 20

    with BPlusPager(path) as pager:
        tree = BPlusTree(pager)
        for k in range(total):
            raw = tree.search(k)
            assert raw is not None, f"key {k} missing after multi-cycle flush"


def test_crash_between_flush_cycles(tmp_path):
    """
    Flush once (clean), then insert more and crash without flushing.
    Recovery must restore the second batch while keeping the first.
    """
    path = str(tmp_path / "db.db")

    # First batch — clean flush
    with BPlusPager(path, order=4) as pager:
        tree = BPlusTree(pager)
        for k in range(15):
            tree.insert(k, encode_record([k, "batch1"]))

    # Second batch — crash before flush
    pager = BPlusPager(path, order=4)
    tree  = BPlusTree(pager)
    for k in range(15, 30):
        tree.insert(k, encode_record([k, "batch2"]))
    pager._wal.fsync()
    crash_pager(pager)

    with BPlusPager(path) as pager2:
        tree2 = BPlusTree(pager2)
        for k in range(30):
            raw = tree2.search(k)
            assert raw is not None, f"key {k} missing after crash"
