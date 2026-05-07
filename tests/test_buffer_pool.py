"""
Tests for BufferPool (LRU eviction, dirty-page tracking, pin/unpin).
"""

import pytest
from src.buffer_pool import BufferPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_page(page_id: int):
    """Return a simple object that acts as a stand-in for a real page."""
    class _FakePage:
        def __init__(self, pid):
            self.page_id = pid
        def __repr__(self):
            return f"Page({self.page_id})"
    return _FakePage(page_id)


# ---------------------------------------------------------------------------
# Basic get / put
# ---------------------------------------------------------------------------

def test_get_miss_returns_none():
    pool = BufferPool(capacity=4)
    assert pool.get(0) is None


def test_put_then_get():
    pool = BufferPool(capacity=4)
    p = make_page(1)
    pool.put(1, p)
    assert pool.get(1) is p


def test_put_replaces_existing():
    pool = BufferPool(capacity=4)
    p1 = make_page(1)
    p2 = make_page(1)   # same page_id, different object
    pool.put(1, p1)
    pool.put(1, p2)
    assert pool.get(1) is p2


def test_size_increments():
    pool = BufferPool(capacity=4)
    assert pool.size() == 0
    pool.put(0, make_page(0))
    assert pool.size() == 1
    pool.put(1, make_page(1))
    assert pool.size() == 2


def test_contains():
    pool = BufferPool(capacity=4)
    pool.put(5, make_page(5))
    assert pool.contains(5)
    assert not pool.contains(6)


# ---------------------------------------------------------------------------
# LRU eviction — basic
# ---------------------------------------------------------------------------

def test_lru_evicts_oldest_when_full():
    """With capacity=3 inserting a 4th page evicts the LRU (first inserted)."""
    pool = BufferPool(capacity=3)
    for i in range(3):
        pool.put(i, make_page(i))

    # page 0 is LRU — it should be evicted when page 3 is inserted
    pool.put(3, make_page(3))

    assert not pool.contains(0)
    assert pool.contains(1)
    assert pool.contains(2)
    assert pool.contains(3)


def test_access_refreshes_lru_order():
    """Accessing a page makes it MRU so it is not the next eviction victim."""
    pool = BufferPool(capacity=3)
    for i in range(3):
        pool.put(i, make_page(i))   # order: 0(LRU) 1 2(MRU)

    pool.get(0)   # 0 is now MRU: order is 1(LRU) 2 0(MRU)

    pool.put(3, make_page(3))   # should evict 1, not 0

    assert pool.contains(0)
    assert not pool.contains(1)
    assert pool.contains(2)
    assert pool.contains(3)


def test_eviction_returns_dirty_page():
    """Evicting a dirty page must be reported so the caller can write it."""
    pool = BufferPool(capacity=1)
    p0 = make_page(0)
    pool.put(0, p0, dirty=True)

    evicted = pool.put(1, make_page(1))

    assert len(evicted) == 1
    assert evicted[0][0] == 0    # evicted page_id
    assert evicted[0][1] is p0


def test_eviction_of_clean_page_returns_empty():
    pool = BufferPool(capacity=1)
    pool.put(0, make_page(0), dirty=False)

    evicted = pool.put(1, make_page(1))
    assert evicted == []


# ---------------------------------------------------------------------------
# Dirty-page tracking
# ---------------------------------------------------------------------------

def test_new_page_not_dirty_by_default():
    pool = BufferPool(capacity=4)
    pool.put(0, make_page(0))
    assert pool.dirty_pages() == []


def test_put_with_dirty_flag():
    pool = BufferPool(capacity=4)
    p = make_page(0)
    pool.put(0, p, dirty=True)
    dirty = pool.dirty_pages()
    assert len(dirty) == 1
    assert dirty[0] == (0, p)


def test_mark_dirty():
    pool = BufferPool(capacity=4)
    p = make_page(0)
    pool.put(0, p)
    pool.mark_dirty(0)
    dirty = pool.dirty_pages()
    assert len(dirty) == 1
    assert dirty[0][0] == 0


def test_clear_dirty():
    pool = BufferPool(capacity=4)
    pool.put(0, make_page(0), dirty=True)
    pool.clear_dirty(0)
    assert pool.dirty_pages() == []


def test_put_update_preserves_dirty():
    """Replacing a dirty page with a clean one should keep the dirty flag."""
    pool = BufferPool(capacity=4)
    pool.put(0, make_page(0), dirty=True)
    pool.put(0, make_page(0), dirty=False)   # update — dirty should stay
    assert len(pool.dirty_pages()) == 1


def test_multiple_dirty_pages():
    pool = BufferPool(capacity=8)
    for i in range(4):
        pool.put(i, make_page(i), dirty=(i % 2 == 0))   # 0, 2 are dirty
    dirty_ids = {pid for pid, _ in pool.dirty_pages()}
    assert dirty_ids == {0, 2}


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------

def test_pinned_page_not_evicted():
    pool = BufferPool(capacity=2)
    pool.put(0, make_page(0))
    pool.pin(0)            # pin page 0 so it cannot be evicted
    pool.put(1, make_page(1))   # fills capacity

    # page 0 is pinned so page 1 (LRU unpinned) would be evicted first
    # but pool is only at capacity=2 so no eviction yet — let's go over:
    pool.put(2, make_page(2))   # triggers eviction of page 1, not 0

    assert pool.contains(0)   # pinned — must survive
    assert not pool.contains(1)
    assert pool.contains(2)


def test_unpin_allows_eviction():
    pool = BufferPool(capacity=2)
    pool.put(0, make_page(0))
    pool.pin(0)
    pool.put(1, make_page(1))

    # Unpin 0 — now 0 is the LRU unpinned page
    pool.unpin(0)

    pool.put(2, make_page(2))   # 0 is LRU unpinned, should be evicted

    assert not pool.contains(0)
    assert pool.contains(1)
    assert pool.contains(2)


def test_pin_count_multiple():
    """pin() is a reference count — need two unpins to allow eviction."""
    pool = BufferPool(capacity=2)
    pool.put(0, make_page(0))
    pool.pin(0)    # pins = 1
    pool.pin(0)    # pins = 2
    pool.unpin(0)  # pins = 1 — still pinned
    pool.put(1, make_page(1))
    pool.put(2, make_page(2))  # capacity exceeded — page 1 evicted, not 0

    assert pool.contains(0)


def test_all_pinned_pool_grows():
    """If all pages are pinned, pool grows beyond capacity rather than blocking."""
    pool = BufferPool(capacity=2)
    pool.put(0, make_page(0))
    pool.pin(0)
    pool.put(1, make_page(1))
    pool.pin(1)
    pool.put(2, make_page(2))   # no eviction possible — pool grows

    assert pool.contains(0)
    assert pool.contains(1)
    assert pool.contains(2)
    assert pool.size() == 3


# ---------------------------------------------------------------------------
# Capacity edge cases
# ---------------------------------------------------------------------------

def test_capacity_one():
    pool = BufferPool(capacity=1)
    pool.put(0, make_page(0))
    evicted = pool.put(1, make_page(1))
    assert not pool.contains(0)
    assert pool.contains(1)
    assert len(evicted) == 0   # page 0 was clean


def test_invalid_capacity():
    with pytest.raises(ValueError):
        BufferPool(capacity=0)


# ---------------------------------------------------------------------------
# Integration: dirty evictions flow through BPlusPager correctly
# ---------------------------------------------------------------------------

def test_pager_evicts_dirty_pages_to_disk(tmp_path):
    """
    Use a tiny buffer pool (capacity=4) with many insertions to force
    eviction, then verify all pages are readable after reopening.
    """
    from src.bplus_pager import BPlusPager
    from src.bplus_tree  import BPlusTree
    from src.record      import encode_record, decode_record

    path = str(tmp_path / "small_pool.db")
    n    = 60   # enough inserts to overflow a pool of 4

    with BPlusPager(path, order=4, pool_capacity=4) as pager:
        tree = BPlusTree(pager)
        for k in range(n):
            tree.insert(k, encode_record([k, f"v{k}"]))

    # Reopen with a normal pool and verify every key is intact
    with BPlusPager(path, pool_capacity=256) as pager:
        tree = BPlusTree(pager)
        for k in range(n):
            raw = tree.search(k)
            assert raw is not None, f"key {k} missing after reopen"
            assert decode_record(raw) == [k, f"v{k}"]
