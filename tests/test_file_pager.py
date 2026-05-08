"""
Persistence tests for BPlusPager and BPlusTree.

All tests use pytest's tmp_path fixture so no test files are left behind.

Coverage:
    - File creation and magic-number validation
    - Meta page: root_page_id, next_page_id, order round-trip
    - LeafPage to_bytes / from_bytes round-trip
    - IndexPage to_bytes / from_bytes round-trip
    - BPlusPager opens an existing file correctly (cache miss → disk read)
    - BPlusTree: insert → flush → reopen → search still works
    - BPlusTree: insert → flush → reopen → range_scan still works
    - Large tree: 200 keys survive a close/reopen cycle
    - Reopening preserves the tree's order from the meta page
    - Opening a corrupt file raises ValueError
"""

import os
import struct
import pytest

from src.bplus_page  import LeafPage, IndexPage, PageType, PAGE_SIZE, NO_NEXT
from src.bplus_pager import BPlusPager, _META_MAGIC
from src.bplus_tree  import BPlusTree
from src.record      import encode_record, decode_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_pager(tmp_path, order=4, name="test.db") -> BPlusPager:
    return BPlusPager(str(tmp_path / name), order=order)


def reopen_pager(tmp_path, name="test.db") -> BPlusPager:
    """Open an already-existing database file."""
    return BPlusPager(str(tmp_path / name))


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_file_is_created(tmp_path):
    pager = fresh_pager(tmp_path)
    pager.close()
    assert os.path.exists(tmp_path / "test.db")


def test_new_file_starts_with_magic(tmp_path):
    pager = fresh_pager(tmp_path)
    pager.close()
    with open(tmp_path / "test.db", 'rb') as f:
        assert f.read(8) == _META_MAGIC


def test_corrupt_magic_raises(tmp_path):
    path = str(tmp_path / "bad.db")
    with open(path, 'wb') as f:
        f.write(b"\x00" * PAGE_SIZE)
    with pytest.raises(ValueError, match="not a valid"):
        BPlusPager(path)


# ---------------------------------------------------------------------------
# Meta page round-trip
# ---------------------------------------------------------------------------

def test_meta_stores_order(tmp_path):
    pager = fresh_pager(tmp_path, order=7)
    pager.close()
    pager2 = reopen_pager(tmp_path)
    assert pager2._order == 7
    pager2.close()


def test_meta_stores_next_page_id(tmp_path):
    pager = fresh_pager(tmp_path, order=4)
    pager.new_leaf_page()
    pager.new_leaf_page()
    pager.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    assert pager2._next_id == 2
    pager2.close()


def test_meta_stores_root_page_id(tmp_path):
    pager = fresh_pager(tmp_path)
    pager.root_page_id = 0
    pager.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    assert pager2.root_page_id == 0
    pager2.close()


def test_meta_root_none_when_unset(tmp_path):
    pager = fresh_pager(tmp_path)
    pager.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    assert pager2.root_page_id is None
    pager2.close()


# ---------------------------------------------------------------------------
# LeafPage serialization round-trip
# ---------------------------------------------------------------------------

def test_leaf_page_to_from_bytes_empty():
    leaf  = LeafPage(page_id=3, max_keys=10)
    leaf2 = LeafPage.from_bytes(3, leaf.to_bytes(), max_keys=10)
    assert leaf2.page_id      == 3
    assert leaf2.next_page_id is None
    assert leaf2._entries()   == []


def test_leaf_page_to_from_bytes_with_entries():
    leaf = LeafPage(page_id=5, max_keys=10)
    leaf.insert(1, b"alice")
    leaf.insert(2, b"bob")
    leaf.insert(3, b"carol")

    leaf2 = LeafPage.from_bytes(5, leaf.to_bytes(), max_keys=10)
    assert leaf2._entries() == [(1, b"alice"), (2, b"bob"), (3, b"carol")]


def test_leaf_page_preserves_next_page_id():
    leaf              = LeafPage(page_id=0, max_keys=10)
    leaf.next_page_id = 42
    leaf2             = LeafPage.from_bytes(0, leaf.to_bytes(), max_keys=10)
    assert leaf2.next_page_id == 42


def test_leaf_page_no_next_serialises_as_sentinel():
    leaf = LeafPage(page_id=0, max_keys=10)
    raw  = leaf.to_bytes()
    nxt  = struct.unpack_from('<I', raw, 9)[0]   # offset 9 = PAGE_LSN_SIZE(8) + type(1)
    assert nxt == NO_NEXT


def test_leaf_page_to_bytes_is_page_size():
    leaf = LeafPage(page_id=0, max_keys=10)
    assert len(leaf.to_bytes()) == PAGE_SIZE


# ---------------------------------------------------------------------------
# IndexPage serialization round-trip
# ---------------------------------------------------------------------------

def test_index_page_to_from_bytes_empty():
    page  = IndexPage(page_id=1, max_keys=4)
    page2 = IndexPage.from_bytes(1, page.to_bytes(), max_keys=4)
    assert page2.keys     == []
    assert page2.children == []


def test_index_page_to_from_bytes_with_data():
    page          = IndexPage(page_id=2, max_keys=4)
    page.keys     = [10, 20, 30]
    page.children = [0, 1, 2, 3]

    page2 = IndexPage.from_bytes(2, page.to_bytes(), max_keys=4)
    assert page2.keys     == [10, 20, 30]
    assert page2.children == [0, 1, 2, 3]


def test_index_page_to_bytes_is_page_size():
    page = IndexPage(page_id=0, max_keys=4)
    assert len(page.to_bytes()) == PAGE_SIZE


def test_index_page_type_byte_is_zero():
    page = IndexPage(page_id=0, max_keys=4)
    assert page.to_bytes()[0] == PageType.INDEX.value


def test_leaf_page_type_byte_is_one():
    leaf = LeafPage(page_id=0, max_keys=4)
    assert leaf.to_bytes()[8] == PageType.LEAF.value   # offset 8 = after PAGE_LSN_SIZE(8)


# ---------------------------------------------------------------------------
# BPlusPager: cache miss → read from disk
# ---------------------------------------------------------------------------

def test_pager_reads_leaf_page_from_disk(tmp_path):
    pager = fresh_pager(tmp_path)
    leaf  = pager.new_leaf_page()
    leaf.insert(99, b"hello")
    pager.mark_dirty(leaf)
    pager.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    # Cache is empty on the new pager instance — it must read from disk.
    loaded = pager2.get_page(0)
    assert isinstance(loaded, LeafPage)
    assert loaded.search(99) == b"hello"
    pager2.close()


def test_pager_reads_index_page_from_disk(tmp_path):
    pager = fresh_pager(tmp_path)
    idx   = pager.new_index_page()
    idx.keys     = [5, 10]
    idx.children = [0, 1, 2]
    pager.mark_dirty(idx)
    pager.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    loaded = pager2.get_page(0)
    assert isinstance(loaded, IndexPage)
    assert loaded.keys     == [5, 10]
    assert loaded.children == [0, 1, 2]
    pager2.close()


# ---------------------------------------------------------------------------
# BPlusTree end-to-end persistence
# ---------------------------------------------------------------------------

def test_tree_insert_flush_reopen_search(tmp_path):
    pager = fresh_pager(tmp_path, order=4)
    tree  = BPlusTree(pager)
    tree.insert(1, b"alice")
    tree.insert(2, b"bob")
    tree.insert(3, b"carol")
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    assert tree2.search(1) == b"alice"
    assert tree2.search(2) == b"bob"
    assert tree2.search(3) == b"carol"
    assert tree2.search(9) is None
    pager2.close()


def test_tree_search_after_reopen_with_splits(tmp_path):
    """Order=2 forces many splits; all keys must survive a close/reopen."""
    pager = fresh_pager(tmp_path, order=2)
    tree  = BPlusTree(pager)
    for k in range(1, 16):
        tree.insert(k, f"v{k}".encode())
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    for k in range(1, 16):
        assert tree2.search(k) == f"v{k}".encode(), f"key {k} missing after reopen"
    pager2.close()


def test_tree_range_scan_after_reopen(tmp_path):
    pager = fresh_pager(tmp_path, order=2)
    tree  = BPlusTree(pager)
    for k in range(1, 11):
        tree.insert(k, f"v{k}".encode())
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    result = tree2.range_scan(3, 7)
    assert [k for k, _ in result] == [3, 4, 5, 6, 7]
    pager2.close()


def test_tree_200_keys_survive_reopen(tmp_path):
    pager = fresh_pager(tmp_path, order=10)
    tree  = BPlusTree(pager)
    for k in range(200):
        tree.insert(k, f"record_{k}".encode())
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    for k in range(200):
        assert tree2.search(k) == f"record_{k}".encode()
    pager2.close()


def test_tree_order_restored_from_file(tmp_path):
    """The order stored in the meta page must be used when reopening."""
    pager = fresh_pager(tmp_path, order=7)
    tree  = BPlusTree(pager)
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    assert tree2._order == 7
    pager2.close()


def test_tree_with_encoded_records_survives_reopen(tmp_path):
    rows = {
        10: [10, "alice",  None],
        20: [20, "bob",    42  ],
        30: [30, "carol",  None],
    }
    pager = fresh_pager(tmp_path, order=4)
    tree  = BPlusTree(pager)
    for k, row in rows.items():
        tree.insert(k, encode_record(row))
    tree.flush()
    pager.close()

    pager2 = reopen_pager(tmp_path)
    tree2  = BPlusTree(pager2)
    for k, expected in rows.items():
        raw = tree2.search(k)
        assert raw is not None
        assert decode_record(raw) == expected
    pager2.close()


# ---------------------------------------------------------------------------
# File size sanity
# ---------------------------------------------------------------------------

def test_file_size_is_multiple_of_page_size(tmp_path):
    pager = fresh_pager(tmp_path, order=4)
    tree  = BPlusTree(pager)
    for k in range(20):
        tree.insert(k, b"x")
    tree.flush()
    pager.close()

    size = os.path.getsize(tmp_path / "test.db")
    assert size % PAGE_SIZE == 0


def test_file_size_grows_with_page_count(tmp_path):
    pager = fresh_pager(tmp_path, order=2)
    tree  = BPlusTree(pager)
    # 1 page before inserts (the root leaf)
    tree.flush()
    size_before = os.path.getsize(tmp_path / "test.db")

    for k in range(20):
        tree.insert(k, b"x")
    tree.flush()
    pager.close()

    size_after = os.path.getsize(tmp_path / "test.db")
    assert size_after > size_before
