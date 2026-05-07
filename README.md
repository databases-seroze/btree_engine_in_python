# py_btree_engine

A B-Tree database storage engine built from scratch in Python. This project implements the core components found in production databases like SQLite and PostgreSQL — slotted pages, record serialization, and B-Tree leaf node operations — with a clean separation of concerns between each layer.

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Engine API                  │  put / get / delete
├─────────────────────────────────────────────┤
│                   B-Tree                    │  search, insert, split
├─────────────────────────────────────────────┤
│                   Pager                     │  page allocation & access
├──────────────────┬──────────────────────────┤
│  Slotted Page    │        Record            │  storage layout & encoding
└──────────────────┴──────────────────────────┘
```

### Components

| Module | Description |
|---|---|
| `src/slotted_page.py` | Fixed 4KB pages with a slot array and variable-length records |
| `src/record.py` | Row serialization supporting `INT`, `STRING`, and `NULL` types |
| `src/btree.py` | B-Tree leaf node: search, insert, and split |
| `src/pager.py` | Page manager (in-memory; designed for file-backed I/O) |
| `src/engine.py` | Top-level `put` / `get` / `delete` API (in progress) |
| `src/cursor.py` | Sequential scan iterator (in progress) |
| `src/constants.py` | Shared constants (`PAGE_SIZE = 4096`) |

## Slotted Page Layout

Each page is 4096 bytes. The header tracks free space, slots grow left-to-right, and records are packed right-to-left:

```
┌──────────────────────────────────────────────┐
│ Header (12 bytes)                            │
│   free_start | free_end | num_slots          │
├──────────────────────────────────────────────┤
│ Slot array  →→→                              │
├──────────────────────────────────────────────┤
│              free space                      │
├──────────────────────────────────────────────┤
│                        ←←← Records          │
└──────────────────────────────────────────────┘
```

Deletion is lazy (slot marked invalid); `compact()` reclaims space and defragments.

## Record Format

Records carry their own type information, enabling schema-less storage:

```
[header_size: 2B][num_cols: 2B][type_0: 1B][type_1: 1B]...[payload...]
```

Supported types:

| Type | Tag | Size |
|---|---|---|
| `NULL` | `0` | 0 bytes |
| `INT` | `1` | 4 bytes (little-endian) |
| `STRING` | `2` | 2-byte length prefix + UTF-8 data |

## Data Flow

A `db.get("user_1")` call traverses the tree like this:

1. Engine asks B-Tree to locate `"user_1"`
2. B-Tree asks Pager for the root page
3. B-Tree decodes the page's keys via Record
4. B-Tree follows the child page pointer and repeats until a leaf is reached
5. Record extracts the value and returns it to the caller

## Getting Started

No external dependencies — only the Python standard library.

```bash
git clone git@github.com:databases-seroze/btree_engine_in_python.git
cd btree_engine_in_python
```

Run the tests:

```bash
pytest tests/ -v
```

Quick example:

```python
from src.slotted_page import SlottedPage
from src.record import encode_record, decode_record
from src.btree import BTreeLeaf

# Slotted page
page = SlottedPage()
slot = page.insert(b"hello")
assert page.read(slot) == b"hello"

# Record encoding
data = encode_record([42, "alice", None])
assert decode_record(data) == [42, "alice", None]

# B-Tree leaf
leaf = BTreeLeaf()
leaf.insert("user_1", encode_record([1, "alice", None]))
raw = leaf.search("user_1")
print(decode_record(raw))  # [1, 'alice', None]
```

## Status

| Component | Status |
|---|---|
| Slotted page (insert, read, delete, compact) | Done |
| Record serialization (INT, STRING, NULL) | Done |
| B-Tree leaf (search, insert, split) | Done |
| Pager (in-memory) | Done |
| B-Tree internal nodes & rebalancing | In progress |
| File-backed pager | In progress |
| Engine API (put / get / delete) | In progress |
| Cursor / iterator | In progress |

## Project Structure

```
py_btree_engine/
├── src/
│   ├── btree.py
│   ├── constants.py
│   ├── cursor.py
│   ├── engine.py
│   ├── pager.py
│   ├── record.py
│   └── slotted_page.py
├── tests/
│   ├── test_btree.py
│   ├── test_record.py
│   └── test_slotted_page.py
├── scratch_codes/
│   └── lsm_tree.py        # exploratory LSM Tree with WAL + compaction
├── main.py
└── README.md
```
