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

### Phase 3 — Transactions (next)
- [ ] Transactions — BEGIN/COMMIT/ROLLBACK built on top of WAL

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
  WAL is truncated to zero after each successful flush (checkpoint cycle).
```

---

## Key Design Decisions

- **Copy-up vs push-up on split**: leaf splits *copy* the middle key to the
  parent (it stays in the right leaf for routing). Index splits *push* the
  middle key up (it leaves both children entirely).
- **max_keys = order**: stored in the meta page so reopening the file always
  uses the same split threshold.
- **SlottedPage size param**: LeafPage uses `SlottedPage(size=4091)` leaving
  the first 5 bytes of the 4096-byte page for the B+Tree header.
