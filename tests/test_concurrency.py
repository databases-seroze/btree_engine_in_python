"""
Concurrency integration tests — Transaction + LockManager + Engine.

These tests use real threads and verify that the locking layer prevents
lost updates, enforces isolation, and surfaces deadlocks correctly.
"""

import threading
import time
import tempfile
import os
import pytest

from src.engine       import Engine
from src.lock_manager import DeadlockError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db():
    return Engine()


# ===========================================================================
# 1. Lock acquisition through Transaction API
# ===========================================================================

def test_txn_put_acquires_x_lock():
    db = make_db()
    txn = db.begin()
    txn.put(1, [1, "alice"])
    # Key 1 should have an X lock held by this txn
    entry = db._lock_manager._table.get(1)
    assert entry is not None
    assert entry.holders.get(txn._txn_id) == 'X'
    txn.rollback()


def test_txn_delete_acquires_x_lock():
    db = make_db()
    db.put(1, [1, "alice"])
    txn = db.begin()
    txn.delete(1)
    entry = db._lock_manager._table.get(1)
    assert entry is not None
    assert entry.holders.get(txn._txn_id) == 'X'
    txn.rollback()


def test_txn_get_acquires_s_lock():
    db = make_db()
    db.put(1, [1, "alice"])
    txn = db.begin()
    txn.get(1)
    entry = db._lock_manager._table.get(1)
    assert entry is not None
    assert entry.holders.get(txn._txn_id) == 'S'
    txn.rollback()


def test_txn_scan_acquires_s_locks_on_results():
    db = make_db()
    for i in range(1, 6):
        db.put(i, [i, f"row{i}"])
    txn = db.begin()
    rows = txn.scan(1, 5)
    assert len(rows) == 5
    for key, _ in rows:
        entry = db._lock_manager._table.get(key)
        assert entry is not None, f"no lock entry for key {key}"
        assert entry.holders.get(txn._txn_id) == 'S'
    txn.rollback()


def test_commit_releases_all_locks():
    db = make_db()
    with db.begin() as txn:
        txn.put(1, [1, "alice"])
        txn.put(2, [2, "bob"])
    # After commit, lock table should be empty for these keys
    assert db._lock_manager._table.get(1) is None
    assert db._lock_manager._table.get(2) is None


def test_rollback_releases_all_locks():
    db = make_db()
    txn = db.begin()
    txn.put(1, [1, "alice"])
    txn.put(2, [2, "bob"])
    txn.rollback()
    assert db._lock_manager._table.get(1) is None
    assert db._lock_manager._table.get(2) is None


# ===========================================================================
# 2. Isolation — concurrent transactions
# ===========================================================================

def test_second_writer_blocked_until_first_commits():
    """
    T1 holds X on key 1.  T2 tries to write key 1 in a background thread.
    T2 must block until T1 commits.
    """
    db = make_db()
    db.put(1, [1, "original"])

    t1_txn  = db.begin()
    t1_txn.put(1, [1, "t1"])

    t2_done  = threading.Event()
    t2_value = [None]

    def t2_run():
        with db.begin() as txn:
            txn.put(1, [1, "t2"])
        t2_value[0] = db.get(1)
        t2_done.set()

    t2 = threading.Thread(target=t2_run, daemon=True)
    t2.start()

    assert not t2_done.wait(timeout=0.15), "T2 should be blocked"
    t1_txn.commit()

    assert t2_done.wait(timeout=2.0), "T2 should unblock after T1 commits"
    assert t2_value[0] == [1, "t2"]
    t2.join(timeout=1.0)


def test_no_lost_update_under_concurrent_increments():
    """
    N threads each read a counter, increment it, and write it back inside a
    transaction.  Because the read acquires S and the write upgrades to X,
    each thread serialises — the final value must equal N.
    """
    db = make_db()
    db.put(0, [0, 0])   # counter row: key=0, value=[id, count]

    N       = 10
    errors  = []
    barrier = threading.Barrier(N)

    def increment(thread_id):
        barrier.wait()
        try:
            with db.begin() as txn:
                row   = txn.get(0)          # S lock
                count = row[1]
                txn.put(0, [0, count + 1])  # upgrades to X
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=increment, args=(i,), daemon=True)
               for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Thread errors: {errors}"
    assert db.get(0) == [0, N], f"Expected count={N}, got {db.get(0)}"


def test_reader_not_blocked_by_reader():
    """
    Multiple concurrent readers on the same key all proceed without blocking.
    """
    db = make_db()
    db.put(1, [1, "data"])

    N       = 8
    done    = []
    barrier = threading.Barrier(N)

    def reader():
        barrier.wait()
        with db.begin() as txn:
            val = txn.get(1)
        done.append(val)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    assert len(done) == N
    assert all(v == [1, "data"] for v in done)


def test_sequential_transactions_see_each_others_commits():
    db = make_db()

    with db.begin() as t1:
        t1.put(1, [1, "first"])

    with db.begin() as t2:
        assert t2.get(1) == [1, "first"]
        t2.put(1, [1, "second"])

    assert db.get(1) == [1, "second"]


# ===========================================================================
# 3. Deadlock through Transaction API
# ===========================================================================

def test_deadlock_two_transactions():
    """
    T1: put(key=1) then tries put(key=2).
    T2: put(key=2) then tries put(key=1).
    → Deadlock: whichever resolves second raises DeadlockError.
    """
    db = make_db()
    db.put(1, [1, "a"])
    db.put(2, [2, "b"])

    t1_held  = threading.Event()
    t2_held  = threading.Event()
    deadlock = []

    def t1_run():
        try:
            txn = db.begin()
            txn.put(1, [1, "t1-key1"])   # grab X(1)
            t1_held.set()
            t2_held.wait()               # wait until T2 grabs X(2)
            txn.put(2, [2, "t1-key2"])   # will block or deadlock
            txn.commit()
        except DeadlockError:
            deadlock.append('t1')
            try:
                txn.rollback()
            except Exception:
                pass

    def t2_run():
        try:
            txn = db.begin()
            txn.put(2, [2, "t2-key2"])   # grab X(2)
            t2_held.set()
            t1_held.wait()               # wait until T1 grabs X(1)
            time.sleep(0.05)             # let T1 queue up on key 2
            txn.put(1, [1, "t2-key1"])   # will detect cycle → DeadlockError
            txn.commit()
        except DeadlockError:
            deadlock.append('t2')
            try:
                txn.rollback()
            except Exception:
                pass

    t1 = threading.Thread(target=t1_run, daemon=True)
    t2 = threading.Thread(target=t2_run, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert len(deadlock) >= 1, "At least one transaction should have detected the deadlock"


# ===========================================================================
# 4. File-backed persistence with concurrent transactions
# ===========================================================================

def test_concurrent_commits_all_persist(tmp_path):
    db_path = str(tmp_path / "conc.db")
    N = 5

    with Engine(db_path) as db:
        barrier = threading.Barrier(N)
        errors  = []

        def writer(key):
            barrier.wait()
            try:
                with db.begin() as txn:
                    txn.put(key, [key, f"row{key}"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(k,), daemon=True)
                   for k in range(1, N + 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors

    # Reopen and verify all rows are present
    with Engine(db_path) as db:
        for key in range(1, N + 1):
            assert db.get(key) == [key, f"row{key}"], f"key {key} missing after reopen"
