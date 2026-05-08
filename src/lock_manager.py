"""
LockManager — key-level two-phase locking for the B+Tree engine.

Protocol
--------
Shared (S) lock   : multiple transactions may hold simultaneously (readers).
Exclusive (X) lock: only one transaction may hold; incompatible with S and X.

A transaction acquires an S lock on every key it reads and an X lock on every
key it writes.  Locks are released all-at-once on commit or rollback (strict
2PL), which guarantees serializability.

Lock compatibility
------------------
         holder→  S    X
waiter↓ S        ✓    ✗
        X        ✗    ✗

Upgrade (S → X)
---------------
If a transaction already holds S on a key and later wants X on the same key,
the request is treated as an upgrade.  If the transaction is the only S holder
the upgrade is granted immediately.  Otherwise it waits (keeping its S lock)
until all other S holders release.

If two transactions simultaneously try to upgrade S → X on the same key, a
deadlock results and is detected by the wait-for graph check.

Deadlock detection
------------------
A wait-for graph records, for each blocked transaction, the set of transactions
it is waiting for.  Before any transaction blocks it performs a DFS on the
graph.  If a cycle is found, DeadlockError is raised immediately for the
transaction that would complete the cycle (i.e., the one currently trying to
acquire).  The caller is responsible for rolling back and releasing all locks.

FIFO waiter ordering
--------------------
Waiters are served in arrival order.  When a holder releases, we scan from the
head of the queue and grant as many consecutive compatible requests as possible,
stopping at the first incompatible one.  This prevents exclusive-lock starvation.
"""

import threading
from collections import defaultdict


class DeadlockError(Exception):
    """Raised when the lock manager detects a cycle in the wait-for graph."""


class _LockEntry:
    """Per-key lock state (not thread-safe on its own; protected by LockManager._mu)."""

    __slots__ = ('holders', 'waiters')

    def __init__(self):
        # txn_id → 'S' | 'X'
        self.holders: dict[int, str] = {}
        # FIFO queue: list of (mode, txn_id, threading.Event)
        self.waiters: list = []


class LockManager:
    """
    Key-level two-phase lock manager.

    Thread-safe.  All state is protected by a single internal mutex (_mu).
    Blocking is done on per-request threading.Events so that the mutex is
    not held while a transaction waits.
    """

    def __init__(self):
        self._mu        = threading.Lock()
        # key (int) → _LockEntry
        self._table:    dict[int, _LockEntry] = {}
        # txn_id → set of keys it currently holds (for bulk release)
        self._txn_keys: dict[int, set[int]]   = defaultdict(set)
        # txn_id → set of txn_ids it is currently waiting for (wait-for graph)
        self._wait_for: dict[int, set[int]]   = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, txn_id: int, key: int, mode: str) -> None:
        """
        Acquire a lock on *key* for *txn_id*.

        Parameters
        ----------
        txn_id : int   Transaction identifier.
        key    : int   The key being locked.
        mode   : str   'S' (shared/read) or 'X' (exclusive/write).

        Blocks until the lock is granted.  Raises DeadlockError immediately
        if granting would create a cycle in the wait-for graph.  Raises
        TimeoutError after 30 s as a safety net against missed wakeups.
        """
        event = None
        entry = None

        with self._mu:
            entry = self._table.setdefault(key, _LockEntry())
            self._txn_keys[txn_id].add(key)

            held = entry.holders.get(txn_id)

            # Already holds X (strongest), or holds S and only needs S → done.
            if held == 'X' or (held == 'S' and mode == 'S'):
                return

            # Check if we can grant right now.
            if self._can_grant(entry, txn_id, mode):
                entry.holders[txn_id] = mode
                return

            # Must wait: register in the waiter queue.
            event = threading.Event()
            entry.waiters.append((mode, txn_id, event))

            # Build wait-for edges: txn_id waits for every current holder
            # except itself (relevant for upgrade S → X when still in holders).
            self._wait_for[txn_id] = set(entry.holders) - {txn_id}

            # Deadlock check before sleeping.
            if self._has_cycle(txn_id):
                entry.waiters.remove((mode, txn_id, event))
                del self._wait_for[txn_id]
                raise DeadlockError(
                    f"Deadlock detected: txn {txn_id} waiting for {mode!r} lock"
                    f" on key {key}"
                )

        # --- Block outside the global mutex so holders can release. ---
        granted = event.wait(timeout=30.0)
        if not granted:
            # Safety net: clean up and surface the timeout.
            with self._mu:
                try:
                    entry.waiters.remove((mode, txn_id, event))
                except ValueError:
                    pass  # already removed by a releaser (race)
                self._wait_for.pop(txn_id, None)
            raise TimeoutError(
                f"Lock timeout: txn {txn_id} waiting for {mode!r} lock on key {key}"
            )
        # The releaser already moved us into entry.holders — nothing left to do.

    def release_all(self, txn_id: int) -> None:
        """
        Release every lock held by *txn_id*.

        Called on transaction commit or rollback.  After this returns the
        transaction holds no locks and blocked waiters have been woken.
        """
        to_signal = []

        with self._mu:
            for key in list(self._txn_keys.pop(txn_id, ())):
                entry = self._table.get(key)
                if entry is None:
                    continue
                entry.holders.pop(txn_id, None)
                to_signal.extend(self._advance_waiters(entry))
                # Prune empty entries to keep the table small.
                if not entry.holders and not entry.waiters:
                    del self._table[key]

            # Remove this txn from the wait-for graph (it's no longer waiting).
            self._wait_for.pop(txn_id, None)

        # Signal outside the mutex so awakened threads don't contend on _mu.
        for ev in to_signal:
            ev.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_grant(self, entry: _LockEntry, txn_id: int, mode: str) -> bool:
        """
        Return True if *txn_id* can be granted *mode* on *entry* right now.

        Treats *txn_id*'s own existing hold as transparent (enabling upgrades).
        """
        others = {t: m for t, m in entry.holders.items() if t != txn_id}
        if not others:
            return True
        if mode == 'X':
            # X is incompatible with any other holder.
            return False
        # S is compatible only when all other holders also have S.
        return all(m == 'S' for m in others.values())

    def _advance_waiters(self, entry: _LockEntry) -> list:
        """
        Grant locks to as many head-of-queue waiters as possible.

        Scans in FIFO order; stops at the first request that cannot be
        granted given the current (and newly granted) holder set.  This
        prevents exclusive-lock starvation.

        Returns the list of Events to signal (after releasing _mu).
        """
        to_signal  = []
        new_waiters = []
        stopped    = False

        for mode, txn_id, event in entry.waiters:
            if not stopped and self._can_grant(entry, txn_id, mode):
                entry.holders[txn_id] = mode
                self._wait_for.pop(txn_id, None)
                to_signal.append(event)
            else:
                stopped = True
                new_waiters.append((mode, txn_id, event))

        entry.waiters = new_waiters
        return to_signal

    def _has_cycle(self, start: int) -> bool:
        """
        Return True if there is a cycle in the wait-for graph reachable from
        *start* that passes back through *start*.

        Uses iterative DFS.  Called while _mu is held.
        """
        visited = set()
        stack   = list(self._wait_for.get(start, ()))

        while stack:
            node = stack.pop()
            if node == start:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._wait_for.get(node, ()))

        return False
