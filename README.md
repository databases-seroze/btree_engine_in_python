# py_btree_engine

A B+Tree database storage engine built from scratch in Python, implementing the core components found in production databases — slotted pages, record serialization, buffer pool, write-ahead log, transactions, and key-level locking — with a clean separation of concerns between each layer.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Engine API                        │  put / get / delete / scan / begin
├──────────────────────────────────────────────────────┤
│          Transaction  │  LockManager                  │  BEGIN/COMMIT/ROLLBACK, 2PL S/X locks
├──────────────────────────────────────────────────────┤
│                     B+Tree                            │  search, insert, delete, range_scan
├──────────────────────────────────────────────────────┤
│                     Pager                             │  page allocation, buffer pool, WAL
├──────────────────┬───────────────────────────────────┤
│   Slotted Page   │   Record   │   WAL   │ Buffer Pool │
└──────────────────┴───────────────────────────────────┘
```

### Modules

| Module | Description |
|---|---|
| `src/engine.py` | Top-level `put` / `get` / `delete` / `scan` / `begin` API |
| `src/bplus_tree.py` | B+Tree: search, insert, delete (borrow/merge), range scan |
| `src/bplus_pager.py` | Page allocator: buffer pool integration, WAL writes, crash recovery |
| `src/bplus_page.py` | LeafPage (SlottedPage-backed) and IndexPage with `page_lsn` header |
| `src/slotted_page.py` | Fixed 4 KB pages with slot array and variable-length records |
| `src/record.py` | Row serialization: `INT`, `STRING`, `NULL` |
| `src/buffer_pool.py` | LRU buffer pool with dirty-page tracking and pin/unpin |
| `src/wal.py` | Write-ahead log: full-page images, pageLSN/flushedLSN, crash recovery |
| `src/transaction.py` | Buffered BEGIN/COMMIT/ROLLBACK with read-your-writes overlay |
| `src/lock_manager.py` | Key-level S/X two-phase locking with deadlock detection |
| `src/cursor.py` | Range scan iterator via leaf linked list |

## Features

### B+Tree Storage
- Full B+Tree with internal (index) and leaf pages
- Leaf pages linked in sorted order for efficient range scans
- Splits: copy-up for leaves, push-up for index nodes
- Deletes with borrow-left, borrow-right, merge, and root collapse

### Durability (WAL)
- Write-ahead log with full-page images (physiological logging)
- ARIES-style `pageLSN` / `flushedLSN` protocol: page only written to disk after its WAL record is fsynced
- `META_UPDATE` records track root changes for crash recovery
- CRC32 on every WAL record for torn-write detection
- Checkpoint + WAL truncation after every full flush cycle

### Transactions
- Explicit `BEGIN` / `COMMIT` / `ROLLBACK` with context manager support
- Buffered writes: all ops staged in memory, applied atomically on commit
- Read-your-writes: staged puts/deletes visible within the same transaction
- Crash safety: uncommitted transactions are discarded on recovery

### Concurrency (2PL)
- Key-level **shared (S)** and **exclusive (X)** locks
- Multiple concurrent readers, serialized writers
- Lock upgrade: S → X when a read key is later written
- FIFO waiter queue prevents exclusive-lock starvation
- Wait-for graph with DFS cycle detection raises `DeadlockError` before blocking
- Strict 2PL: locks held until commit/rollback (serializability)

## Usage

```python
from src.engine import Engine

# In-memory (no persistence)
db = Engine()
db.put(1, [1, "alice", 30])
db.put(2, [2, "bob",   25])

print(db.get(1))           # [1, "alice", 30]
db.delete(1)

for key, row in db.scan(1, 100):
    print(key, row)

# File-backed (survives process restart)
with Engine("users.db", order=100) as db:
    db.put(1, [1, "alice", 30])
    db.flush()

# Transactions
with Engine("users.db") as db:
    with db.begin() as txn:
        txn.put(3, [3, "carol", 28])
        txn.delete(2)
        print(txn.get(3))  # [3, "carol", 28] — read your own write

# Concurrent transactions (thread-safe)
from src.lock_manager import DeadlockError

with Engine("users.db") as db:
    try:
        with db.begin() as txn:
            row = txn.get(1)         # acquires S lock
            txn.put(1, [1, "updated", 31])  # upgrades to X
    except DeadlockError:
        pass  # retry logic here
```

## File Format

```
Offset 0                       : Meta page (4096 bytes)
                                   [0:8]   magic b"BPTREE\x01\x00"
                                   [8:12]  root_page_id  (uint32LE)
                                   [12:16] next_page_id  (uint32LE)
                                   [16:20] order         (uint32LE)

Offset PAGE_SIZE + id*PAGE_SIZE : Data page (4096 bytes)

LeafPage:   [0:8] page_lsn | [8] type=1 | [9:13] next_page_id | [13:] SlottedPage
IndexPage:  [0:8] page_lsn | [8] type=0 | [9:13] num_keys | keys[] | children[]

WAL file (<dbpath>.wal):
  PAGE_WRITE  [LSN:8][type=1:1][page_id:4][page_bytes:4096][CRC32:4]  = 4113 B
  META_UPDATE [LSN:8][type=3:1][root_page_id:4][CRC32:4]              = 17 B
  CHECKPOINT  [LSN:8][type=2:1][CRC32:4]                              = 13 B
  TXN_BEGIN   [LSN:8][type=4:1][txn_id:8][CRC32:4]                    = 21 B
  TXN_COMMIT  [LSN:8][type=5:1][txn_id:8][CRC32:4]                    = 21 B
  TXN_ABORT   [LSN:8][type=6:1][txn_id:8][CRC32:4]                    = 21 B
```

## Roadmap

| Phase | Status |
|---|---|
| Phase 1 — Core B+Tree (slotted pages, records, insert/delete/scan, file-backed pager) | Done |
| Phase 2 — Durability (buffer pool, write-ahead log, crash recovery) | Done |
| Phase 3 — Transactions (BEGIN/COMMIT/ROLLBACK, WAL integration) | Done |
| Phase 4 — Concurrency (key-level 2PL, deadlock detection) | Done |
| Phase 5 — SQL / Query Layer (parser, schema, SELECT/INSERT/DELETE/UPDATE) | Planned |
| Phase 6 — Secondary Indexes (per-column B+Trees) | Planned |
| Phase 7 — Multi-table / Joins (nested-loop, hash join, foreign keys) | Planned |

## Getting Started

No external dependencies — only the Python standard library.

```bash
git clone git@github.com:databases-seroze/btree_engine_in_python.git
cd btree_engine_in_python
pytest tests/ -v
```

## Project Structure

```
py_btree_engine/
├── src/
│   ├── engine.py
│   ├── bplus_tree.py
│   ├── bplus_pager.py
│   ├── bplus_page.py
│   ├── slotted_page.py
│   ├── record.py
│   ├── buffer_pool.py
│   ├── wal.py
│   ├── transaction.py
│   ├── lock_manager.py
│   └── cursor.py
├── tests/
│   ├── test_btree.py
│   ├── test_slotted_page.py
│   ├── test_record.py
│   ├── test_file_pager.py
│   ├── test_buffer_pool.py
│   ├── test_wal.py
│   ├── test_transaction.py
│   ├── test_lock_manager.py
│   └── test_concurrency.py
└── README.md
```
