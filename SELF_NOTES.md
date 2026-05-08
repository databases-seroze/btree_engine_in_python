# py_btree_engine — Self Notes

## Milestones

### Phase 1 — Core ✅
- [x] SlottedPage — fixed-size page with slot directory, insert/read/delete/compact
- [x] Record encoding — variable-length rows with INT, STRING, NULL types
- [x] B+Tree — LeafPage (SlottedPage-backed), IndexPage, search/insert/range_scan
- [x] File-backed pager — 4096-byte pages, meta page, flush/close, reopen
- [x] Delete — borrow-right, borrow-left, merge, root collapse; full underflow propagation
- [x] Engine API (`engine.py`) — put/get/delete/scan; context manager; in-memory + file-backed
- [x] Cursor — next() O(1) via leaf links; prev() O(k) re-walk; reset(); for-loop iterator

### Phase 2 — Durability ✅
- [x] Buffer pool — LRU eviction, dirty-page tracking, pin/unpin
- [x] WAL (Write-Ahead Log) — physiological: full-page images, pageLSN/flushedLSN,
      META_UPDATE for root changes, CRC32 torn-write detection, checkpoint+truncation

### Phase 3 — Transactions ✅
- [x] Transactions — BEGIN/COMMIT/ROLLBACK built on top of WAL
- [x] Read-your-writes overlay, deferred evictions, crash-safe recovery

### Phase 4 — Concurrency ✅
- [x] Key-level S/X locks via LockManager (multiple readers, serialized writers)
- [x] Lock upgrade: S → X, immediate if sole holder, waits otherwise
- [x] FIFO waiter queue — prevents X-lock starvation
- [x] Wait-for graph + DFS cycle detection → DeadlockError before blocking
- [x] Strict 2PL: locks held until commit/rollback (serializability)

### Phase 5 — SQL / Query Layer ✅
- [x] Lexer — tokenises SQL; keywords, literals, symbols, identifiers
- [x] Parser — recursive descent; produces typed AST nodes
- [x] Catalog — JSON-backed schema registry; survives close/reopen
- [x] Executor — access path selection: PK equality → get(), PK BETWEEN → scan(), else full scan + filter
- [x] Database — top-level interface; one Engine file per table; context manager

### Phase 6 — Secondary Indexes ✅
- [x] One Engine file per index (`<index_name>.db`)
- [x] IndexManager: on_insert / on_delete / on_update keep indexes in sync
- [x] Back-fill: CREATE INDEX on non-empty table scans and populates the index
- [x] DROP INDEX: closes engine, deletes files, falls back to full scan
- [x] Executor access path: index equality → lookup(); index BETWEEN → range_lookup()
- [x] Catalog persists index schemas to catalog.json (backward-compatible)
- [x] Indexes persist across close/reopen; multiple indexes per table supported
- [x] Limitation: only INT columns can be indexed (B+Tree keys are uint32)

### Phase 7 — Multi-table / Joins
- [ ] Nested-loop join, hash join
- [ ] Foreign keys

---

## Data Flow

When you call `db.get(1)`:

1. Engine asks B+Tree to find key `1`
2. B+Tree asks Pager for the root page
3. If root is an IndexPage: decode keys, pick the right child pointer, repeat
4. Once a LeafPage is reached: binary-search the slot array for key `1`
5. Record decodes the raw bytes and returns the row to the Engine

---

## Delete — design notes

Three cases when removing a key from a leaf:

1. **Leaf stays at or above half-full** — just remove the key, done.
2. **Leaf underflows, right sibling has a spare key** — *borrow right*:
   move the sibling's first key into this leaf, update the parent separator.
3. **Leaf underflows, left sibling has a spare key** — *borrow left*:
   move the sibling's last key into this leaf, update the parent separator.
4. **Both siblings are at minimum** — *merge*: combine this leaf and one
   sibling into one page, remove the separator key from the parent.
   The parent may now underflow too — propagate upward recursively.

Minimum occupancy = ceil(max_keys / 2). Root is exempt (can have 1 key).

---

## File Format

```
Offset 0                  : Meta page (4096 bytes)
                              [0:8]   magic b"BPTREE\x01\x00"
                              [8:12]  root_page_id  (uint32LE)
                              [12:16] next_page_id  (uint32LE)
                              [16:20] order         (uint32LE)

Offset PAGE_SIZE + id*PAGE_SIZE : Data page for page_id=id (4096 bytes)

LeafPage on disk:
  [0:8]    page_lsn      (uint64LE — LSN of last WAL record for this page)
  [8]      page_type = 1
  [9:13]   next_page_id  (uint32LE, 0xFFFFFFFF = no sibling)
  [13:4096] SlottedPage body (4083 bytes)

IndexPage on disk:
  [0:8]    page_lsn      (uint64LE)
  [8]      page_type = 0
  [9:13]   num_keys  (uint32LE)
  [13…]    keys[]      (num_keys × uint32LE)
  […]      children[]  ((num_keys+1) × uint32LE)

WAL file (<dbpath>.wal):
  PAGE_WRITE  record: [LSN:8][type=1:1][page_id:4][page_bytes:4096][CRC32:4]  = 4113 B
  META_UPDATE record: [LSN:8][type=3:1][root_page_id:4][CRC32:4]              = 17 B
  CHECKPOINT  record: [LSN:8][type=2:1][CRC32:4]                              = 13 B
  TXN_BEGIN   record: [LSN:8][type=4:1][txn_id:8][CRC32:4]                      = 21 B
  TXN_COMMIT  record: [LSN:8][type=5:1][txn_id:8][CRC32:4]                      = 21 B
  TXN_ABORT   record: [LSN:8][type=6:1][txn_id:8][CRC32:4]                      = 21 B
  WAL is truncated to zero after each successful flush (checkpoint cycle).
  Recovery: PAGE_WRITEs outside a TXN_BEGIN…TXN_COMMIT pair are discarded.
```

---

## Concurrency — design notes

Two-phase locking (2PL) at the key level:

- **S lock** (shared/read): multiple txns may hold simultaneously.
- **X lock** (exclusive/write): only one txn; incompatible with S and X.
- `get/scan` → acquire S; `put/delete` → acquire X.
- **Upgrade** (S → X): if no other S holders, granted immediately.  Otherwise
  waits — keeping the S hold — until all other S holders release.  Two txns
  trying to upgrade simultaneously cause a deadlock (detected by cycle check).
- **Strict 2PL**: all locks released at once on commit/rollback, not earlier.
  This is what gives serializability.
- **FIFO queue**: waiters served in arrival order; we stop at the first
  incompatible waiter so an X request cannot be starved by a stream of S requests.
- **Deadlock detection**: before blocking, DFS the wait-for graph.  If the new
  edges complete a cycle, raise `DeadlockError` for the txn that would close it.
  Caller must rollback (which calls `release_all`, unblocking other waiters).
- **Phantom reads**: scan() locks individual keys in its result, not the range.
  A new key inserted into the scanned range by a concurrent txn is not locked.
  Full predicate/range locking is deferred to a later phase.

---

## Key Design Decisions

- **Copy-up vs push-up on split**: leaf splits *copy* the middle key to the
  parent (it stays in the right leaf for routing). Index splits *push* the
  middle key up (it leaves both children entirely).
- **max_keys = order**: stored in the meta page so reopening the file always
  uses the same split threshold.
- **SlottedPage size param**: LeafPage uses `SlottedPage(size=4091)` leaving
  the first 5 bytes of the 4096-byte page for the B+Tree header.
