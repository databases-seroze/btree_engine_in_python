"""
Transaction — explicit BEGIN / COMMIT / ROLLBACK for the B+Tree engine.

Design
------
A Transaction buffers all write operations (put / delete) in memory and
only applies them to the real B+Tree on COMMIT.  This keeps the tree
completely unchanged during the transaction — reads see the committed
state of the tree plus the transaction's own writes via an in-memory
overlay.

Commit sequence
---------------
1. WAL.append_txn_begin(txn_id)        — log the start
2. pager.begin_txn_write()             — defer buffer-pool evictions to disk
3. Apply all buffered ops to the tree  — tree modified, PAGE_WRITE records
                                         buffered in WAL
4. WAL.append_txn_commit(txn_id)
5. WAL.fsync()                         — TXN_COMMIT is now durable
6. pager.end_txn_write()               — flush any deferred evictions

After step 5 the transaction is permanently committed even if we crash
before step 6; the deferred evictions are covered by the PAGE_WRITE
records already in the WAL.

Rollback
--------
Discard the buffered ops — the tree was never touched, so there is
nothing to undo.  A TXN_ABORT record is appended to the WAL (buffered,
not necessarily fsynced) so recovery can confirm the abort.

Crash safety
------------
If we crash between steps 3 and 5, the WAL has PAGE_WRITE records for
this transaction but no TXN_COMMIT.  On recovery, those PAGE_WRITEs are
buffered internally and discarded because no matching TXN_COMMIT is found.
The tree is restored to its pre-transaction state.

Read-your-writes
----------------
get() and scan() check the transaction's overlay first so that a put()
followed by a get() within the same transaction returns the new value,
even though it hasn't been committed to the tree yet.

Usage
-----
    with db.begin() as txn:
        txn.put(1, [1, "alice"])
        txn.put(2, [2, "bob"])
        txn.delete(3)
        val = txn.get(1)     # returns [1, "alice"] (read-your-write)
    # committed on clean exit; rolled back on exception

    # Manual control:
    txn = db.begin()
    try:
        txn.put(1, [1, "alice"])
        txn.commit()
    except:
        txn.rollback()
        raise
"""

from src.record import encode_record, decode_record


class Transaction:
    """
    Buffered transaction over an Engine.

    Parameters
    ----------
    engine : Engine
        The engine this transaction operates on.  The Engine must not be
        closed while the transaction is open.
    """

    def __init__(self, engine):
        self._engine   = engine
        self._lm       = engine._lock_manager
        self._txn_id   = engine._pager.alloc_txn_id()
        # Buffered ops: list of ('put', key, record_bytes) or ('delete', key)
        self._ops:     list  = []
        # Read-your-writes overlay: key → raw record bytes
        self._overlay: dict  = {}
        # Keys deleted in this transaction (absent from overlay to allow re-insert)
        self._deleted: set   = set()
        self._done:    bool  = False   # True after commit or rollback

    # ------------------------------------------------------------------
    # Write operations (buffered)
    # ------------------------------------------------------------------

    def put(self, key: int, row: list) -> None:
        """
        Stage an insert / update.

        The change is visible immediately within this transaction via get()
        and scan(), but is not applied to the tree until commit().
        """
        self._check_open()
        self._lm.acquire(self._txn_id, key, 'X')
        record = encode_record(row)
        self._ops.append(('put', key, record))
        self._overlay[key] = record
        self._deleted.discard(key)

    def delete(self, key: int) -> None:
        """
        Stage a delete.

        Subsequent get() calls within this transaction return None for key.
        """
        self._check_open()
        self._lm.acquire(self._txn_id, key, 'X')
        self._ops.append(('delete', key))
        self._deleted.add(key)
        self._overlay.pop(key, None)

    # ------------------------------------------------------------------
    # Read operations (overlay-aware)
    # ------------------------------------------------------------------

    def get(self, key: int) -> list | None:
        """
        Return the row for key, or None if absent.

        Acquires a shared lock on *key* so that no concurrent transaction can
        modify it until this transaction commits or rolls back.  Checks the
        transaction's own overlay first so that staged puts are visible.
        """
        self._check_open()
        self._lm.acquire(self._txn_id, key, 'S')
        if key in self._deleted:
            return None
        if key in self._overlay:
            return decode_record(self._overlay[key])
        return self._engine.get(key)

    def scan(self, start: int = 0, end: int = 0xFFFF_FFFF) -> list:
        """
        Return [(key, row), ...] for all keys in [start, end], reflecting
        this transaction's staged writes.

        Acquires a shared lock on every key in the result set so that no
        concurrent transaction can modify those rows until this transaction
        commits or rolls back.

        Note: new keys inserted into the range by concurrent transactions
        after this scan completes are not locked (phantom reads).  Full
        range/predicate locking is deferred to a later phase.
        """
        self._check_open()
        # Start from the committed tree's range
        base = {k: v for k, v in self._engine._tree.range_scan(start, end)}
        # Remove keys deleted in this transaction
        for k in self._deleted:
            base.pop(k, None)
        # Apply staged puts (including updates)
        for k, v in self._overlay.items():
            if start <= k <= end:
                base[k] = v
        result = [(k, decode_record(v)) for k, v in sorted(base.items())]
        # Acquire S locks on every key we are about to return.
        for k, _ in result:
            self._lm.acquire(self._txn_id, k, 'S')
        return result

    # ------------------------------------------------------------------
    # Commit / Rollback
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Apply all staged operations atomically and make them durable.

        After this returns, the changes are visible in the engine and
        survive crashes.
        """
        self._check_open()
        self._done = True

        if not self._ops:
            return   # nothing to do — no WAL records needed

        pager = self._engine._pager
        tree  = self._engine._tree
        wal   = pager._wal

        # Step 1: log the start of the transaction
        if wal is not None:
            wal.append_txn_begin(self._txn_id)

        # Step 2: prevent evictions from reaching disk before TXN_COMMIT
        pager.begin_txn_write()

        try:
            # Step 3: apply all buffered ops to the real tree
            for op, *args in self._ops:
                if op == 'put':
                    key, record = args
                    tree.insert(key, record)
                else:
                    key, = args
                    tree.delete(key)

            # Step 4 & 5: log commit and fsync — transaction is durable
            if wal is not None:
                wal.append_txn_commit(self._txn_id)
                wal.fsync()

        finally:
            # Step 6: safe to flush deferred evictions now
            pager.end_txn_write()

        # Step 7: release all locks — other transactions may now proceed
        self._lm.release_all(self._txn_id)

    def rollback(self) -> None:
        """
        Discard all staged operations.

        The engine is unchanged.  A TXN_ABORT record is buffered in the
        WAL so that recovery can confirm this transaction was aborted.
        """
        if self._done:
            return
        self._done = True

        if self._ops:
            wal = self._engine._pager._wal
            if wal is not None:
                wal.append_txn_abort(self._txn_id)

        self._lm.release_all(self._txn_id)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'Transaction':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self):
        if self._done:
            raise RuntimeError("Transaction is already closed (committed or rolled back)")
