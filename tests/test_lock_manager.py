"""
Tests for LockManager — key-level two-phase locking.

Sections
--------
1.  Immediate grant — single transaction
2.  Shared-lock compatibility
3.  Exclusive-lock exclusivity
4.  Lock upgrade (S → X)
5.  FIFO waiter ordering
6.  Deadlock detection
7.  release_all cleanup
8.  Concurrent-thread integration
"""

import threading
import time
import pytest

from src.lock_manager import LockManager, DeadlockError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lm():
    return LockManager()


def bg_acquire(lm, txn_id, key, mode, *, started=None, done=None, exc_box=None):
    """
    Acquire a lock in a background thread.

    started : threading.Event — set just before acquire() is called
    done    : threading.Event — set after acquire() returns
    exc_box : list            — if acquire raises, exception is appended here
    """
    def _run():
        if started:
            started.set()
        try:
            lm.acquire(txn_id, key, mode)
        except Exception as e:
            if exc_box is not None:
                exc_box.append(e)
        finally:
            if done:
                done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ===========================================================================
# 1. Immediate grant — single transaction
# ===========================================================================

def test_s_lock_granted_immediately():
    lm = make_lm()
    lm.acquire(1, 42, 'S')  # must not block


def test_x_lock_granted_immediately():
    lm = make_lm()
    lm.acquire(1, 42, 'X')


def test_acquire_is_idempotent_ss():
    lm = make_lm()
    lm.acquire(1, 10, 'S')
    lm.acquire(1, 10, 'S')  # second call is a no-op


def test_acquire_is_idempotent_xx():
    lm = make_lm()
    lm.acquire(1, 10, 'X')
    lm.acquire(1, 10, 'X')


def test_x_subsumes_s_for_same_txn():
    # T1 holds X; acquiring S on same key is a no-op (X is stronger)
    lm = make_lm()
    lm.acquire(1, 10, 'X')
    lm.acquire(1, 10, 'S')  # must not block or change state


def test_locks_on_different_keys_are_independent():
    lm = make_lm()
    lm.acquire(1, 1, 'X')
    lm.acquire(1, 2, 'X')  # different key — no conflict


# ===========================================================================
# 2. Shared-lock compatibility
# ===========================================================================

def test_two_txns_can_hold_s_on_same_key():
    lm = make_lm()
    lm.acquire(1, 99, 'S')
    lm.acquire(2, 99, 'S')  # must not block — S is compatible with S


def test_many_txns_can_hold_s_simultaneously():
    lm = make_lm()
    for txn in range(1, 11):
        lm.acquire(txn, 7, 'S')
    # Release all — should not raise
    for txn in range(1, 11):
        lm.release_all(txn)


# ===========================================================================
# 3. Exclusive-lock exclusivity
# ===========================================================================

def test_x_blocks_s_from_other_txn():
    lm = make_lm()
    lm.acquire(1, 5, 'X')

    done  = threading.Event()
    exc   = []
    t     = bg_acquire(lm, 2, 5, 'S', done=done, exc_box=exc)

    # T2 must be blocked — done should NOT be set yet
    assert not done.wait(timeout=0.1), "T2 should be blocked waiting for X holder to release"

    lm.release_all(1)
    assert done.wait(timeout=1.0), "T2 should unblock after T1 releases"
    assert not exc, f"unexpected exception in T2: {exc}"
    t.join(timeout=1.0)


def test_x_blocks_x_from_other_txn():
    lm = make_lm()
    lm.acquire(1, 5, 'X')

    done = threading.Event()
    exc  = []
    t    = bg_acquire(lm, 2, 5, 'X', done=done, exc_box=exc)

    assert not done.wait(timeout=0.1), "T2 should be blocked"
    lm.release_all(1)
    assert done.wait(timeout=1.0), "T2 should unblock"
    assert not exc
    t.join(timeout=1.0)


def test_s_blocks_x_from_other_txn():
    lm = make_lm()
    lm.acquire(1, 5, 'S')

    done = threading.Event()
    exc  = []
    t    = bg_acquire(lm, 2, 5, 'X', done=done, exc_box=exc)

    assert not done.wait(timeout=0.1), "T2 (X) should be blocked by T1 (S)"
    lm.release_all(1)
    assert done.wait(timeout=1.0)
    assert not exc
    t.join(timeout=1.0)


def test_multiple_s_holders_all_block_x():
    lm = make_lm()
    lm.acquire(1, 3, 'S')
    lm.acquire(2, 3, 'S')

    done = threading.Event()
    exc  = []
    t    = bg_acquire(lm, 3, 3, 'X', done=done, exc_box=exc)

    assert not done.wait(timeout=0.1)

    lm.release_all(1)
    assert not done.wait(timeout=0.1), "Still blocked — T2 still holds S"

    lm.release_all(2)
    assert done.wait(timeout=1.0), "Now unblocked"
    assert not exc
    t.join(timeout=1.0)


# ===========================================================================
# 4. Lock upgrade (S → X)
# ===========================================================================

def test_upgrade_s_to_x_when_sole_holder():
    # T1 is the only S holder — upgrade to X must be immediate
    lm = make_lm()
    lm.acquire(1, 10, 'S')
    lm.acquire(1, 10, 'X')  # must not block


def test_upgrade_s_to_x_waits_for_other_s_holder():
    lm = make_lm()
    lm.acquire(1, 10, 'S')   # T1 holds S
    lm.acquire(2, 10, 'S')   # T2 holds S

    done = threading.Event()
    exc  = []

    # T1 tries to upgrade S → X in background (will block because T2 holds S)
    t = bg_acquire(lm, 1, 10, 'X', done=done, exc_box=exc)

    assert not done.wait(timeout=0.1), "T1 upgrade should block while T2 holds S"

    lm.release_all(2)
    assert done.wait(timeout=1.0), "T1 upgrade should complete after T2 releases"
    assert not exc
    t.join(timeout=1.0)


def test_upgrade_deadlock_two_upgraders():
    # T1 and T2 both hold S on key 10 and both try to upgrade to X.
    # Each waits for the other → deadlock.
    lm = make_lm()
    lm.acquire(1, 10, 'S')
    lm.acquire(2, 10, 'S')

    upgrade_started = threading.Event()
    exc1 = []

    def t1_upgrade():
        upgrade_started.set()
        try:
            lm.acquire(1, 10, 'X')
        except DeadlockError as e:
            exc1.append(e)

    t1 = threading.Thread(target=t1_upgrade, daemon=True)
    t1.start()
    upgrade_started.wait()
    time.sleep(0.05)   # give T1 time to block

    # T2 now also tries to upgrade — this should detect the deadlock
    with pytest.raises(DeadlockError):
        lm.acquire(2, 10, 'X')

    # Release T2 so T1 can proceed
    lm.release_all(2)
    t1.join(timeout=2.0)
    # Either T1 got the deadlock error or it successfully upgraded — both are valid
    # (depends on which side the detector fires first)


# ===========================================================================
# 5. FIFO waiter ordering
# ===========================================================================

def test_fifo_x_waiter_served_before_later_s_waiter():
    """
    T1 holds X.  T2 waits for X.  T3 arrives after T2 and waits for S.
    When T1 releases: T2 (X) should be granted first, blocking T3 (S).
    """
    lm = make_lm()
    lm.acquire(1, 55, 'X')

    done2 = threading.Event()
    done3 = threading.Event()

    # Start T2 (X waiter) first
    t2 = bg_acquire(lm, 2, 55, 'X', done=done2)
    time.sleep(0.05)   # ensure T2 is enqueued before T3
    # Start T3 (S waiter) after T2
    t3 = bg_acquire(lm, 3, 55, 'S', done=done3)
    time.sleep(0.05)

    # Release T1
    lm.release_all(1)

    # T2 (X) should be granted
    assert done2.wait(timeout=1.0), "T2 should be granted"
    # T3 (S) should still be blocked behind T2 (X)
    assert not done3.wait(timeout=0.1), "T3 should be blocked by T2's X"

    lm.release_all(2)
    assert done3.wait(timeout=1.0), "T3 should be granted after T2 releases"

    t2.join(timeout=1.0)
    t3.join(timeout=1.0)


def test_multiple_s_waiters_granted_together():
    """
    T1 holds X.  T2 and T3 both wait for S.
    When T1 releases, both S waiters should be granted simultaneously.
    """
    lm = make_lm()
    lm.acquire(1, 77, 'X')

    done2 = threading.Event()
    done3 = threading.Event()

    t2 = bg_acquire(lm, 2, 77, 'S', done=done2)
    t3 = bg_acquire(lm, 3, 77, 'S', done=done3)
    time.sleep(0.05)

    lm.release_all(1)

    assert done2.wait(timeout=1.0)
    assert done3.wait(timeout=1.0)

    lm.release_all(2)
    lm.release_all(3)
    t2.join(timeout=1.0)
    t3.join(timeout=1.0)


# ===========================================================================
# 6. Deadlock detection
# ===========================================================================

def test_deadlock_two_txns_cross_waiting():
    """
    T1 holds X(key=1) and waits for X(key=2).
    T2 holds X(key=2) and tries X(key=1) → cycle → DeadlockError.
    """
    lm = make_lm()
    lm.acquire(1, 1, 'X')   # T1 holds key 1
    lm.acquire(2, 2, 'X')   # T2 holds key 2

    t1_blocked = threading.Event()
    exc1       = []

    def t1_wait():
        t1_blocked.set()
        try:
            lm.acquire(1, 2, 'X')   # T1 waits for key 2
        except DeadlockError as e:
            exc1.append(e)

    t1 = threading.Thread(target=t1_wait, daemon=True)
    t1.start()
    t1_blocked.wait()
    time.sleep(0.05)   # let T1 register in wait-for graph

    # T2 now tries key 1 → forms cycle T2→T1→T2
    with pytest.raises(DeadlockError):
        lm.acquire(2, 1, 'X')

    # Clean up: release T2's locks so T1 can finish
    lm.release_all(2)
    t1.join(timeout=2.0)


def test_no_deadlock_with_single_chain():
    """
    T1 waits for T2 waits for T3 — a chain, not a cycle.
    None of them should get a DeadlockError.
    """
    lm = make_lm()
    lm.acquire(3, 10, 'X')  # T3 holds key 10
    lm.acquire(2, 20, 'X')  # T2 holds key 20

    # T2 waits for T3 (T2 wants key 10, held by T3)
    t2_started = threading.Event()
    t2_done    = threading.Event()
    exc2       = []

    def t2_wait():
        t2_started.set()
        try:
            lm.acquire(2, 10, 'X')
        except Exception as e:
            exc2.append(e)
        finally:
            t2_done.set()

    t2 = threading.Thread(target=t2_wait, daemon=True)
    t2.start()
    t2_started.wait()
    time.sleep(0.05)

    # T1 waits for T2 (T1 wants key 20, held by T2) — chain T1→T2→T3, no cycle
    t1_done = threading.Event()
    exc1    = []

    def t1_wait():
        try:
            lm.acquire(1, 20, 'X')
        except Exception as e:
            exc1.append(e)
        finally:
            t1_done.set()

    t1 = threading.Thread(target=t1_wait, daemon=True)
    t1.start()
    time.sleep(0.05)

    # Release in order: T3 → T2 can proceed → T1 can proceed
    lm.release_all(3)
    assert t2_done.wait(timeout=1.0)
    assert not exc2, f"T2 unexpected error: {exc2}"

    lm.release_all(2)
    assert t1_done.wait(timeout=1.0)
    assert not exc1, f"T1 unexpected error: {exc1}"

    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


# ===========================================================================
# 7. release_all cleanup
# ===========================================================================

def test_release_all_removes_entry_when_no_waiters():
    lm = make_lm()
    lm.acquire(1, 42, 'X')
    lm.release_all(1)
    # Internal table should be empty
    assert 42 not in lm._table


def test_release_all_for_multiple_keys():
    lm = make_lm()
    for key in range(10):
        lm.acquire(1, key, 'S')
    lm.release_all(1)
    assert len(lm._table) == 0


def test_release_all_idempotent():
    lm = make_lm()
    lm.acquire(1, 1, 'X')
    lm.release_all(1)
    lm.release_all(1)   # second call on a txn with no locks — must not raise


def test_release_all_unblocks_multiple_waiters():
    lm = make_lm()
    lm.acquire(1, 100, 'X')

    events = []
    threads = []
    for txn in range(2, 6):
        done = threading.Event()
        events.append(done)
        t = bg_acquire(lm, txn, 100, 'S', done=done)
        threads.append(t)

    time.sleep(0.05)   # all 4 waiters enqueued

    lm.release_all(1)

    for ev in events:
        assert ev.wait(timeout=1.0), "All S waiters should unblock"

    for txn in range(2, 6):
        lm.release_all(txn)
    for t in threads:
        t.join(timeout=1.0)


# ===========================================================================
# 8. Concurrent-thread integration
# ===========================================================================

def test_concurrent_readers_do_not_block_each_other():
    lm = make_lm()
    N = 10
    acquired = []
    barrier  = threading.Barrier(N)

    def reader(txn_id):
        barrier.wait()
        lm.acquire(txn_id, 1, 'S')
        acquired.append(txn_id)
        lm.release_all(txn_id)

    threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    assert len(acquired) == N


def test_writer_serialises_against_concurrent_writers():
    """
    N threads each increment a shared counter under an X lock on key 0.
    The final value should equal N.
    """
    lm      = make_lm()
    counter = [0]
    N       = 20
    barrier = threading.Barrier(N)

    def writer(txn_id):
        barrier.wait()
        lm.acquire(txn_id, 0, 'X')
        val = counter[0]
        time.sleep(0.001)        # tiny delay to expose races
        counter[0] = val + 1
        lm.release_all(txn_id)

    threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert counter[0] == N, f"Expected {N}, got {counter[0]} (lost updates!)"
