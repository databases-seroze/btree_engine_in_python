# py_btree_engine

A relational database engine built from scratch in Python — B+Tree storage, write-ahead log, ACID transactions, key-level locking, a SQL query layer, secondary indexes, and multi-table joins with foreign keys. Every layer is implemented without external dependencies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Database (SQL API)                        │
│         execute("SELECT * FROM users WHERE age > 25")            │
├──────────────┬──────────────┬───────────────────────────────────┤
│    Lexer     │    Parser    │           Executor                 │
│  (tokens)    │    (AST)     │  (access path + FK + join)        │
├──────────────┴──────────────┴───────────────────────────────────┤
│            Catalog (JSON)  │  IndexManager (secondary indexes)   │
├────────────────────────────┴─────────────────────────────────────┤
│                        Engine API                                 │
│              put / get / delete / scan / begin                    │
├──────────────────────────┬───────────────────────────────────────┤
│       Transaction        │          LockManager                  │
│  BEGIN/COMMIT/ROLLBACK   │    key-level S/X locks, deadlock      │
├──────────────────────────┴───────────────────────────────────────┤
│                          B+Tree                                   │
│           search, insert, delete, range_scan                      │
├──────────────────────────────────────────────────────────────────┤
│                          Pager                                    │
│          page allocation, buffer pool, WAL, recovery              │
├──────────────────┬──────────────────┬───────────┬───────────────┤
│   Slotted Page   │     Record       │    WAL    │  Buffer Pool  │
└──────────────────┴──────────────────┴───────────┴───────────────┘
```

### Modules

| Module | Description |
|---|---|
| `src/repl.py` | Interactive SQL shell — input loop, table printer, meta-commands |
| `src/database.py` | Top-level SQL interface; one Engine file per table; context manager |
| `src/lexer.py` | Tokenises SQL strings into keyword, identifier, literal, and symbol tokens |
| `src/parser.py` | Recursive descent parser; produces typed AST nodes |
| `src/ast_nodes.py` | AST dataclasses: all statements and expressions |
| `src/catalog.py` | JSON-backed schema registry: tables, indexes, foreign keys |
| `src/executor.py` | Evaluates AST; access path selection; FK enforcement; join execution |
| `src/index_manager.py` | Opens/syncs secondary index Engine files on every write |
| `src/engine.py` | Low-level `put` / `get` / `delete` / `scan` / `begin` API |
| `src/transaction.py` | Buffered BEGIN/COMMIT/ROLLBACK with read-your-writes overlay |
| `src/lock_manager.py` | Key-level S/X two-phase locking with deadlock detection |
| `src/bplus_tree.py` | B+Tree: search, insert, delete (borrow/merge), range scan |
| `src/bplus_pager.py` | Page allocator: buffer pool, WAL integration, crash recovery |
| `src/bplus_page.py` | LeafPage (SlottedPage-backed) and IndexPage with `page_lsn` header |
| `src/slotted_page.py` | Fixed 4 KB pages with slot array and variable-length records |
| `src/record.py` | Row serialization: `INT`, `STRING`, `NULL` |
| `src/buffer_pool.py` | LRU buffer pool with dirty-page tracking and pin/unpin |
| `src/wal.py` | Write-ahead log: full-page images, pageLSN/flushedLSN, crash recovery |
| `src/cursor.py` | Range scan iterator via leaf linked list |

## Features

### Phase 1 — B+Tree Storage
- Full B+Tree with internal (index) and leaf pages
- Leaf pages linked in sorted order for O(log n + k) range scans
- Splits: copy-up for leaves, push-up for index nodes
- Deletes with borrow-left, borrow-right, merge, and root collapse

### Phase 2 — Durability (WAL)
- Write-ahead log with full-page images (physiological logging)
- ARIES-style `pageLSN` / `flushedLSN`: page reaches disk only after its WAL record is fsynced
- `META_UPDATE` records track root page changes for crash recovery
- CRC32 on every WAL record for torn-write detection
- Checkpoint + WAL truncation after every successful flush

### Phase 3 — Transactions
- Explicit `BEGIN` / `COMMIT` / `ROLLBACK` with context manager support
- Buffered writes: all ops staged in memory, applied atomically on commit
- Read-your-writes: staged puts/deletes visible within the same transaction
- Crash safety: uncommitted transactions discarded on recovery

### Phase 4 — Concurrency (2PL)
- Key-level **shared (S)** and **exclusive (X)** locks
- Multiple concurrent readers, serialised writers
- Lock upgrade: S → X when a read key is later written
- FIFO waiter queue prevents exclusive-lock starvation
- Wait-for graph with DFS cycle detection raises `DeadlockError` before blocking
- Strict 2PL: locks held until commit/rollback (serializability)

### Phase 5 — SQL Query Layer
- `CREATE TABLE`, `INSERT`, `SELECT`, `UPDATE`, `DELETE`
- `WHERE` with `=`, `!=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `AND`
- Schema persisted to `catalog.json`; survives close/reopen
- Access path selection: PK equality → `get()`, PK range → `scan()`, else full scan

### Phase 6 — Secondary Indexes
- `CREATE INDEX idx ON table (col)` / `DROP INDEX idx`
- One Engine file per index; non-unique by default (list of PKs per value)
- Back-fill: CREATE INDEX on non-empty table scans and populates
- Executor promotes index lookup over full scan automatically
- Indexes kept in sync on every INSERT / UPDATE / DELETE

### Phase 7 — Joins and Foreign Keys
- `REFERENCES table (col)` in `CREATE TABLE` — FK constraints stored in catalog
- INSERT: referenced parent row must exist; NULL FK skipped
- DELETE: raises if any child row references the deleted PK (no CASCADE)
- `SELECT ... FROM a JOIN b ON a.col = b.col [WHERE ...]`
- Nested-loop join with **index NLJ** when right join column is indexed
- Post-join `WHERE` with `table.col` dotted predicates
- FK constraints and join data persist across close/reopen

## Usage

```python
from src.database import Database

with Database("mydb/") as db:
    # Create tables with foreign key
    db.execute("CREATE TABLE users  (id INT PRIMARY KEY, name STRING, age INT)")
    db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")

    # Insert
    db.execute("INSERT INTO users  VALUES (1, 'alice', 30)")
    db.execute("INSERT INTO users  VALUES (2, 'bob',   25)")
    db.execute("INSERT INTO orders VALUES (1, 1, 500)")
    db.execute("INSERT INTO orders VALUES (2, 1, 200)")

    # Select with WHERE
    rows = db.execute("SELECT * FROM users WHERE age > 25")
    # [{'id': 1, 'name': 'alice', 'age': 30}]

    # Secondary index
    db.execute("CREATE INDEX idx_age ON users (age)")
    rows = db.execute("SELECT * FROM users WHERE age BETWEEN 25 AND 32")

    # JOIN
    rows = db.execute("""
        SELECT users.name, orders.amount
        FROM users JOIN orders ON users.id = orders.user_id
        WHERE orders.amount > 100
    """)
    # [{'users.name': 'alice', 'orders.amount': 500},
    #  {'users.name': 'alice', 'orders.amount': 200}]

    # Update and delete
    db.execute("UPDATE users SET age = 31 WHERE id = 1")
    db.execute("DELETE FROM orders WHERE id = 1")
    db.execute("DELETE FROM users  WHERE id = 1")   # ok — no more child rows

# Re-open: schema, indexes, FK constraints all restored
with Database("mydb/") as db:
    rows = db.execute("SELECT * FROM users")
```

### Low-level Engine (with transactions)

```python
from src.engine import Engine
from src.lock_manager import DeadlockError

with Engine("users.db") as db:
    with db.begin() as txn:
        row = txn.get(1)                    # acquires S lock
        txn.put(1, [1, "updated", 31])      # upgrades to X lock
    # committed — locks released

    try:
        with db.begin() as txn:
            txn.put(1, [1, "a"])
            txn.put(2, [2, "b"])
    except DeadlockError:
        pass  # retry
```

## File Format

```
<dbdir>/
  catalog.json          — table schemas, index definitions, foreign keys
  <table>.db            — one B+Tree Engine file per table
  <table>.db.wal        — WAL alongside each Engine file
  <index_name>.db       — one Engine file per secondary index

B+Tree data file:
  Offset 0                       : Meta page (4096 bytes)
                                     [0:8]   magic b"BPTREE\x01\x00"
                                     [8:12]  root_page_id  (uint32LE)
                                     [12:16] next_page_id  (uint32LE)
                                     [16:20] order         (uint32LE)
  Offset PAGE_SIZE + id*PAGE_SIZE : Data page (4096 bytes)

  LeafPage:   [0:8] page_lsn | [8] type=1 | [9:13] next_page_id | [13:] SlottedPage
  IndexPage:  [0:8] page_lsn | [8] type=0 | [9:13] num_keys | keys[] | children[]

WAL records:
  PAGE_WRITE  [LSN:8][type=1:1][page_id:4][page_bytes:4096][CRC32:4]  = 4113 B
  META_UPDATE [LSN:8][type=3:1][root_page_id:4][CRC32:4]              = 17 B
  CHECKPOINT  [LSN:8][type=2:1][CRC32:4]                              = 13 B
  TXN_BEGIN   [LSN:8][type=4:1][txn_id:8][CRC32:4]                    = 21 B
  TXN_COMMIT  [LSN:8][type=5:1][txn_id:8][CRC32:4]                    = 21 B
  TXN_ABORT   [LSN:8][type=6:1][txn_id:8][CRC32:4]                    = 21 B
```

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | Core B+Tree — slotted pages, records, insert/delete/scan, file-backed pager | Done |
| 2 | Durability — LRU buffer pool, write-ahead log, crash recovery | Done |
| 3 | Transactions — BEGIN/COMMIT/ROLLBACK, WAL integration, read-your-writes | Done |
| 4 | Concurrency — key-level 2PL, lock upgrade, FIFO waiters, deadlock detection | Done |
| 5 | SQL layer — lexer, parser, catalog, SELECT/INSERT/UPDATE/DELETE | Done |
| 6 | Secondary indexes — per-column B+Trees, auto-sync, index scan | Done |
| 7 | Joins and foreign keys — nested-loop join, index NLJ, FK enforcement | Done |
| 8 | Interactive REPL — SQL shell with history, aligned output, meta-commands | Done |

## Getting Started

No external dependencies — only the Python standard library.

```bash
git clone git@github.com:databases-seroze/btree_engine_in_python.git
cd btree_engine_in_python
pytest tests/ -v   # 469 tests
```

### Running the REPL

```bash
python __main__.py mydb/
```

```
py_btree_engine  —  database: /path/to/mydb
Type SQL statements ending with ';', or \help for commands.

db> CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT);
OK
db> INSERT INTO users VALUES (1, 'alice', 30);
OK
db> SELECT * FROM users;
+----+-------+-----+
| id | name  | age |
+----+-------+-----+
| 1  | alice | 30  |
+----+-------+-----+
(1 row)
db> \schema users
+--------+--------+-----+----+
| column | type   | pk  | fk |
+--------+--------+-----+----+
| id     | INT    | YES |    |
| name   | STRING |     |    |
| age    | INT    |     |    |
+--------+--------+-----+----+
db> \tables
  users
db> \quit
Bye.
```

**REPL features:**
- Multi-line input — statement executes when `;` is reached
- Up/down arrow key history (via `readline`)
- Aligned ASCII table output with row count
- Meta-commands: `\tables`, `\schema <table>`, `\indexes`, `\help`, `\quit`
- Ctrl-C clears current input buffer; Ctrl-D exits
- All errors printed without crashing the shell

## Project Structure

```
py_btree_engine/
├── src/
│   ├── database.py        # SQL entry point
│   ├── lexer.py
│   ├── parser.py
│   ├── ast_nodes.py
│   ├── catalog.py
│   ├── executor.py
│   ├── index_manager.py
│   ├── engine.py
│   ├── transaction.py
│   ├── lock_manager.py
│   ├── bplus_tree.py
│   ├── bplus_pager.py
│   ├── bplus_page.py
│   ├── slotted_page.py
│   ├── record.py
│   ├── buffer_pool.py
│   ├── wal.py
│   └── cursor.py
├── __main__.py            # REPL entry point: python __main__.py <dbdir>
└── tests/
    ├── test_btree.py
    ├── test_slotted_page.py
    ├── test_record.py
    ├── test_file_pager.py
    ├── test_buffer_pool.py
    ├── test_wal.py
    ├── test_transaction.py
    ├── test_lock_manager.py
    ├── test_concurrency.py
    ├── test_sql.py
    ├── test_indexes.py
    ├── test_joins_fkeys.py
    └── test_repl.py
```
