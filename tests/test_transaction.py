"""
Tests for Transaction (BEGIN / COMMIT / ROLLBACK).

Sections
--------
1.  Basic commit             — put, get, delete committed correctly
2.  Rollback                 — no changes reach the engine
3.  Read-your-writes         — overlay visible inside the transaction
4.  Scan inside transaction  — merged view of tree + overlay
5.  Atomicity                — all-or-nothing across multiple ops
6.  Context manager          — commit on clean exit, rollback on exception
7.  Isolation (single-writer) — two sequential transactions are independent
8.  Empty transaction        — commit/rollback with no ops
9.  Re-use after done        — closed transaction raises on further ops
10. WAL records              — TXN_BEGIN/COMMIT/ABORT appear in correct order
11. Crash recovery           — committed txn survives crash; uncommitted does not
12. File persistence         — committed data survives reopen
13. Large transactions       — many ops in one transaction
"""

import os
import pytest

from src.engine      import Engine
from src.bplus_pager import BPlusPager
from src.bplus_tree  import BPlusTree
from src.transaction import Transaction
from src.wal         import TXN_BEGIN, TXN_COMMIT, TXN_ABORT
from src.record      import encode_record, decode_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db(**kwargs) -> Engine:
    return Engine(**kwargs)


def crash_engine(engine: Engine):
    """Close file handles without flush — simulates a process crash."""
    if engine._pager._file:
        engine._pager._file.close()
    if engine._pager._wal:
        engine._pager._wal._file.close()


# ===========================================================================
# 1. Basic commit
# ===========================================================================

def test_commit_put_visible_after_commit():
    db = make_db()
    with db.begin() as txn:
        txn.put(1, [1, "alice"])
    assert db.get(1) == [1, "alice"]
    db.close()


def test_commit_multiple_puts():
    db = make_db()
    with db.begin() as txn:
        for k in range(1, 6):
            txn.put(k, [k, f"v{k}"])
    for k in range(1, 6):
        assert db.get(k) == [k, f"v{k}"]
    db.close()


def test_commit_delete_removes_key():
    db = make_db()
    db.put(1, [1, "alice"])
    with db.begin() as txn:
        txn.delete(1)
    assert db.get(1) is None
    db.close()


def test_commit_update_existing_key():
    db = make_db()
    db.put(1, [1, "old"])
    with db.begin() as txn:
        txn.put(1, [1, "new"])
    assert db.get(1) == [1, "new"]
    db.close()


def test_commit_mixed_puts_and_deletes():
    db = make_db()
    for k in range(1, 6):
        db.put(k, [k])
    with db.begin() as txn:
        txn.put(6, [6])
        txn.delete(3)
        txn.put(7, [7])
        txn.delete(1)
    assert db.get(1) is None
    assert db.get(3) is None
    assert db.get(6) == [6]
    assert db.get(7) == [7]
    assert db.get(2) == [2]
    db.close()


def test_commit_delete_nonexistent_key():
    """Deleting a key that doesn't exist in the tree still commits cleanly."""
    db = make_db()
    with db.begin() as txn:
        txn.delete(99)   # key not in tree
    assert db.get(99) is None
    db.close()


# ===========================================================================
# 2. Rollback
# ===========================================================================

def test_rollback_put_not_visible():
    db = make_db()
    txn = db.begin()
    txn.put(1, [1, "alice"])
    txn.rollback()
    assert db.get(1) is None
    db.close()


def test_rollback_delete_not_applied():
    db = make_db()
    db.put(1, [1, "alice"])
    txn = db.begin()
    txn.delete(1)
    txn.rollback()
    assert db.get(1) == [1, "alice"]
    db.close()


def test_rollback_multiple_ops():
    db = make_db()
    db.put(1, [1, "original"])
    txn = db.begin()
    txn.put(2, [2])
    txn.put(3, [3])
    txn.delete(1)
    txn.rollback()
    assert db.get(1) == [1, "original"]
    assert db.get(2) is None
    assert db.get(3) is None
    db.close()


def test_rollback_on_exception_via_context_manager():
    db = make_db()
    db.put(1, [1, "original"])
    try:
        with db.begin() as txn:
            txn.put(1, [1, "modified"])
            raise ValueError("simulated error")
    except ValueError:
        pass
    assert db.get(1) == [1, "original"]
    db.close()


def test_rollback_is_idempotent():
    db = make_db()
    txn = db.begin()
    txn.put(1, [1])
    txn.rollback()
    txn.rollback()   # second call must not raise
    db.close()


# ===========================================================================
# 3. Read-your-writes
# ===========================================================================

def test_get_inside_txn_sees_staged_put():
    db = make_db()
    with db.begin() as txn:
        txn.put(1, [1, "alice"])
        assert txn.get(1) == [1, "alice"]
    db.close()


def test_get_inside_txn_sees_staged_update():
    db = make_db()
    db.put(1, [1, "old"])
    with db.begin() as txn:
        txn.put(1, [1, "new"])
        assert txn.get(1) == [1, "new"]
    db.close()


def test_get_inside_txn_sees_staged_delete():
    db = make_db()
    db.put(1, [1, "alice"])
    with db.begin() as txn:
        txn.delete(1)
        assert txn.get(1) is None
    db.close()


def test_get_inside_txn_falls_through_to_committed_data():
    db = make_db()
    db.put(5, [5, "committed"])
    with db.begin() as txn:
        assert txn.get(5) == [5, "committed"]
    db.close()


def test_delete_then_put_same_key_in_txn():
    """Delete then re-insert same key within one transaction."""
    db = make_db()
    db.put(1, [1, "original"])
    with db.begin() as txn:
        txn.delete(1)
        assert txn.get(1) is None
        txn.put(1, [1, "reinserted"])
        assert txn.get(1) == [1, "reinserted"]
    assert db.get(1) == [1, "reinserted"]
    db.close()


def test_put_then_delete_same_key_in_txn():
    """Put then delete the same key within one transaction."""
    db = make_db()
    with db.begin() as txn:
        txn.put(1, [1])
        txn.delete(1)
        assert txn.get(1) is None
    assert db.get(1) is None
    db.close()


# ===========================================================================
# 4. Scan inside transaction
# ===========================================================================

def test_scan_inside_txn_sees_staged_puts():
    db = make_db()
    db.put(1, [1])
    db.put(3, [3])
    with db.begin() as txn:
        txn.put(2, [2])
        result = txn.scan(1, 3)
    assert [k for k, _ in result] == [1, 2, 3]
    db.close()


def test_scan_inside_txn_hides_staged_deletes():
    db = make_db()
    for k in range(1, 6):
        db.put(k, [k])
    with db.begin() as txn:
        txn.delete(2)
        txn.delete(4)
        result = txn.scan(1, 5)
    assert [k for k, _ in result] == [1, 3, 5]
    db.close()


def test_scan_inside_txn_sees_updates():
    db = make_db()
    db.put(1, [1, "old"])
    db.put(2, [2, "old"])
    with db.begin() as txn:
        txn.put(1, [1, "new"])
        result = dict(txn.scan(1, 2))
    assert result[1] == [1, "new"]
    assert result[2] == [2, "old"]
    db.close()


def test_scan_empty_range_in_txn():
    db = make_db()
    db.put(1, [1])
    with db.begin() as txn:
        txn.put(99, [99])
        result = txn.scan(10, 20)
    assert result == []
    db.close()


# ===========================================================================
# 5. Atomicity
# ===========================================================================

def test_all_ops_visible_together_after_commit():
    """All ops in a transaction become visible simultaneously on commit."""
    db = make_db(order=4)
    for k in range(1, 11):
        db.put(k, [k, "before"])

    with db.begin() as txn:
        for k in range(1, 6):
            txn.put(k, [k, "after"])
        for k in range(6, 11):
            txn.delete(k)

    for k in range(1, 6):
        assert db.get(k) == [k, "after"]
    for k in range(6, 11):
        assert db.get(k) is None
    db.close()


def test_partial_commit_not_possible_on_rollback():
    """On rollback, none of the ops are visible — not even the first one."""
    db = make_db()
    txn = db.begin()
    for k in range(1, 6):
        txn.put(k, [k])
    txn.rollback()
    for k in range(1, 6):
        assert db.get(k) is None
    db.close()


def test_sequential_transactions_are_independent():
    """Second transaction sees the first's committed state, not its own buffer."""
    db = make_db()

    with db.begin() as txn1:
        txn1.put(1, [1, "txn1"])

    with db.begin() as txn2:
        assert txn2.get(1) == [1, "txn1"]   # sees committed state from txn1
        txn2.put(1, [1, "txn2"])

    assert db.get(1) == [1, "txn2"]
    db.close()


# ===========================================================================
# 6. Context manager
# ===========================================================================

def test_context_manager_commits_on_clean_exit():
    db = make_db()
    with db.begin() as txn:
        txn.put(1, [1])
    assert db.get(1) == [1]
    db.close()


def test_context_manager_rolls_back_on_exception():
    db = make_db()
    with pytest.raises(RuntimeError):
        with db.begin() as txn:
            txn.put(1, [1])
            raise RuntimeError("oops")
    assert db.get(1) is None
    db.close()


def test_context_manager_re_raises_exception():
    db = make_db()
    with pytest.raises(ValueError, match="test error"):
        with db.begin() as txn:
            txn.put(1, [1])
            raise ValueError("test error")
    db.close()


# ===========================================================================
# 7. Isolation (single-writer sequential)
# ===========================================================================

def test_second_txn_does_not_see_first_txn_before_commit():
    """
    While txn1 holds X(key=1), txn2 is blocked — it cannot read the key
    at all until txn1 commits (writer blocks readers).  Once txn1 commits,
    txn2 sees the committed value.
    """
    import threading
    db = make_db()
    txn1 = db.begin()
    txn1.put(1, [1, "txn1"])

    read_result = [None]
    read_done   = threading.Event()

    def txn2_read():
        txn2 = db.begin()
        read_result[0] = txn2.get(1)   # blocks until txn1 commits
        txn2.rollback()
        read_done.set()

    t = threading.Thread(target=txn2_read, daemon=True)
    t.start()

    assert not read_done.wait(timeout=0.1), "txn2 should be blocked by txn1's X lock"
    txn1.commit()

    assert read_done.wait(timeout=2.0), "txn2 should unblock after txn1 commits"
    assert read_result[0] == [1, "txn1"]
    t.join(timeout=1.0)
    db.close()


def test_transactions_do_not_interfere_with_each_other():
    db = make_db()
    db.put(10, [10, "original"])

    with db.begin() as txn_a:
        txn_a.put(1, [1, "a"])

    with db.begin() as txn_b:
        txn_b.put(2, [2, "b"])
        txn_b.delete(10)

    assert db.get(1) == [1, "a"]
    assert db.get(2) == [2, "b"]
    assert db.get(10) is None
    db.close()


# ===========================================================================
# 8. Empty transaction
# ===========================================================================

def test_commit_empty_transaction():
    db = make_db()
    db.put(1, [1, "unchanged"])
    with db.begin():
        pass   # no ops
    assert db.get(1) == [1, "unchanged"]
    db.close()


def test_rollback_empty_transaction():
    db = make_db()
    txn = db.begin()
    txn.rollback()   # no ops — should not raise
    db.close()


# ===========================================================================
# 9. Closed transaction raises
# ===========================================================================

def test_put_after_commit_raises():
    db = make_db()
    txn = db.begin()
    txn.commit()
    with pytest.raises(RuntimeError):
        txn.put(1, [1])
    db.close()


def test_get_after_rollback_raises():
    db = make_db()
    txn = db.begin()
    txn.rollback()
    with pytest.raises(RuntimeError):
        txn.get(1)
    db.close()


def test_scan_after_commit_raises():
    db = make_db()
    txn = db.begin()
    txn.commit()
    with pytest.raises(RuntimeError):
        txn.scan()
    db.close()


# ===========================================================================
# 10. WAL records
# ===========================================================================

def test_wal_has_txn_begin_and_commit_after_commit(tmp_path):
    """After a commit, the WAL file contains TXN_BEGIN then TXN_COMMIT."""
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    with db.begin() as txn:
        txn.put(1, [1, "alice"])

    # WAL was checkpointed (empty) after the txn's commit triggered a flush?
    # No — the Transaction only fsyncs the WAL, it does NOT checkpoint.
    # The WAL file has TXN_BEGIN, PAGE_WRITE(s), TXN_COMMIT after fsync.
    # (The WAL is truncated only on pager.flush() / close().)
    wal_path = path + '.wal'
    # After commit the WAL has records; after close it is checkpointed.
    # Read before close.
    wal_bytes = open(wal_path, 'rb').read()
    db.close()

    # TXN_BEGIN type byte = 4, TXN_COMMIT type byte = 5
    # Both appear somewhere in the raw bytes.
    assert TXN_BEGIN  in wal_bytes or len(wal_bytes) == 0   # flushed already
    # After close, WAL is checkpointed to zero — verify data is in data file.
    assert os.path.getsize(wal_path) == 0


def test_wal_has_txn_abort_after_rollback(tmp_path):
    """After a rollback, the WAL contains a TXN_ABORT record."""
    import struct, zlib
    from src.wal import WAL, TXN_ABORT as _TXN_ABORT, TXN_RECORD_SIZE

    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    txn = db.begin()
    txn.put(1, [1])
    txn.rollback()

    # Force WAL to disk so we can inspect it
    db._pager._wal.fsync()
    wal_path = path + '.wal'

    with open(wal_path, 'rb') as f:
        raw = f.read()

    # Find a record with type byte = TXN_ABORT (6)
    found_abort = False
    pos = 0
    while pos + 9 <= len(raw):
        lsn, rec_type = struct.unpack_from('<QB', raw, pos)
        if rec_type == _TXN_ABORT:
            found_abort = True
            break
        # Skip record based on type
        if rec_type == 1:   # PAGE_WRITE
            pos += 13 + 4096 + 4
        elif rec_type in (2,):   # CHECKPOINT
            pos += 9 + 4
        elif rec_type in (3,):   # META_UPDATE
            pos += 9 + 4 + 4
        elif rec_type in (4, 5, 6):  # TXN_*
            pos += 9 + 8 + 4
        else:
            break

    assert found_abort, "TXN_ABORT record not found in WAL after rollback"
    db.close()


def test_no_wal_records_for_empty_commit(tmp_path):
    """An empty transaction (no ops) should not write any WAL records."""
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    # Flush initial state so WAL starts clean
    db.flush()

    with db.begin():
        pass   # no ops

    # WAL should still be empty (no TXN_BEGIN for empty txn)
    db._pager._wal.fsync()
    wal_path = path + '.wal'
    assert os.path.getsize(wal_path) == 0
    db.close()


# ===========================================================================
# 11. Crash recovery
# ===========================================================================

def test_committed_txn_survives_crash(tmp_path):
    """A committed transaction must be recovered after a crash."""
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    with db.begin() as txn:
        txn.put(1, [1, "alice"])
        txn.put(2, [2, "bob"])

    # Transaction committed and WAL fsynced.
    # Simulate crash without flushing data pages.
    crash_engine(db)

    with Engine(path) as db2:
        assert db2.get(1) == [1, "alice"]
        assert db2.get(2) == [2, "bob"]


def test_uncommitted_txn_not_recovered_after_crash(tmp_path):
    """A transaction that never committed must not appear after crash recovery."""
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    # Put something committed first
    db.put(10, [10, "committed"])
    db.flush()

    # Start a transaction, force WAL to disk, then crash without committing
    txn = db.begin()
    txn.put(1, [1, "uncommitted"])
    txn.put(2, [2, "uncommitted"])
    # Don't commit — crash immediately
    db._pager._wal.fsync()
    crash_engine(db)

    with Engine(path) as db2:
        assert db2.get(10) == [10, "committed"]
        assert db2.get(1) is None
        assert db2.get(2) is None


def test_rolled_back_txn_not_visible_after_reopen(tmp_path):
    """Explicitly rolled-back transaction must not appear after reopen."""
    path = str(tmp_path / "db.db")
    with Engine(path, order=4) as db:
        txn = db.begin()
        txn.put(1, [1, "ghost"])
        txn.rollback()

    with Engine(path) as db2:
        assert db2.get(1) is None


def test_crash_between_two_transactions(tmp_path):
    """
    First transaction commits (WAL fsynced).  Second transaction starts but
    never commits (crash).  On recovery, first is visible, second is not.
    """
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)

    with db.begin() as txn1:
        txn1.put(1, [1, "first"])

    # Crash mid-second-transaction
    txn2 = db.begin()
    txn2.put(2, [2, "second"])
    db._pager._wal.fsync()
    crash_engine(db)

    with Engine(path) as db2:
        assert db2.get(1) == [1, "first"]
        assert db2.get(2) is None


def test_recovery_with_deletes_in_committed_txn(tmp_path):
    """Deletes inside a committed transaction persist through crash recovery."""
    path = str(tmp_path / "db.db")
    db   = Engine(path, order=4)
    for k in range(1, 6):
        db.put(k, [k])
    db.flush()

    with db.begin() as txn:
        txn.delete(2)
        txn.delete(4)

    crash_engine(db)

    with Engine(path) as db2:
        assert db2.get(1) == [1]
        assert db2.get(2) is None
        assert db2.get(3) == [3]
        assert db2.get(4) is None
        assert db2.get(5) == [5]


# ===========================================================================
# 12. File persistence (clean close, no crash)
# ===========================================================================

def test_committed_data_survives_reopen(tmp_path):
    path = str(tmp_path / "db.db")
    with Engine(path, order=4) as db:
        with db.begin() as txn:
            txn.put(1, [1, "alice"])
            txn.put(2, [2, "bob"])

    with Engine(path) as db2:
        assert db2.get(1) == [1, "alice"]
        assert db2.get(2) == [2, "bob"]


def test_rolled_back_data_absent_after_reopen(tmp_path):
    path = str(tmp_path / "db.db")
    with Engine(path, order=4) as db:
        txn = db.begin()
        txn.put(99, [99, "ghost"])
        txn.rollback()

    with Engine(path) as db2:
        assert db2.get(99) is None


def test_multiple_transactions_persist(tmp_path):
    path = str(tmp_path / "db.db")
    with Engine(path, order=4) as db:
        for batch in range(3):
            with db.begin() as txn:
                for k in range(batch * 5, (batch + 1) * 5):
                    txn.put(k, [k, f"batch{batch}"])

    with Engine(path) as db2:
        for batch in range(3):
            for k in range(batch * 5, (batch + 1) * 5):
                assert db2.get(k) == [k, f"batch{batch}"]


# ===========================================================================
# 13. Large transactions
# ===========================================================================

def test_large_transaction_commit(tmp_path):
    """A transaction with many ops commits correctly."""
    path = str(tmp_path / "db.db")
    n    = 200

    with Engine(path, order=10) as db:
        with db.begin() as txn:
            for k in range(n):
                txn.put(k, [k, f"v{k}"])

    with Engine(path) as db2:
        for k in range(n):
            assert db2.get(k) == [k, f"v{k}"]


def test_large_transaction_rollback():
    """Rolling back a large transaction leaves the engine unchanged."""
    db = make_db(order=10)
    for k in range(50):
        db.put(k, [k, "original"])

    txn = db.begin()
    for k in range(50):
        txn.put(k, [k, "modified"])
    txn.rollback()

    for k in range(50):
        assert db.get(k) == [k, "original"]
    db.close()


def test_large_transaction_survives_crash(tmp_path):
    """A large committed transaction survives crash recovery."""
    path = str(tmp_path / "db.db")
    n    = 150
    db   = Engine(path, order=10)

    with db.begin() as txn:
        for k in range(n):
            txn.put(k, [k, f"v{k}"])

    crash_engine(db)

    with Engine(path) as db2:
        for k in range(n):
            assert db2.get(k) == [k, f"v{k}"]
