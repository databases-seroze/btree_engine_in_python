"""
Integration tests for BPlusTree.

Covers:
  - search on empty tree
  - insert + search
  - update (re-insert same key)
  - leaf split (order=2 forces splits early)
  - root split (new IndexPage root is created)
  - multi-level split propagation
  - range_scan (same leaf)
  - range_scan across multiple leaves (uses linked list)
  - large sequential insert + full scan
  - large random insert + point lookups
"""

import random
import pytest
from src.bplus_tree  import BPlusTree
from src.bplus_page  import LeafPage, IndexPage
from src.bplus_pager import BPlusPager


def make_tree(order=4):
    return BPlusTree(order=order)


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------

def test_search_empty_tree():
    tree = make_tree()
    assert tree.search(42) is None


def test_insert_and_search_single():
    tree = make_tree()
    tree.insert(1, b"alice")
    assert tree.search(1) == b"alice"


def test_search_missing_key():
    tree = make_tree()
    tree.insert(1, b"alice")
    assert tree.search(99) is None


def test_insert_multiple_search_all():
    tree = make_tree()
    data = {1: b"a", 5: b"b", 3: b"c", 7: b"d", 2: b"e"}
    for k, v in data.items():
        tree.insert(k, v)
    for k, v in data.items():
        assert tree.search(k) == v


def test_update_existing_key():
    tree = make_tree()
    tree.insert(10, b"old")
    tree.insert(10, b"new")
    assert tree.search(10) == b"new"


# ---------------------------------------------------------------------------
# Leaf split (order=2 → split after 3rd key in a leaf)
# ---------------------------------------------------------------------------

def test_leaf_split_creates_two_leaves():
    """After enough inserts to force a split, both halves must be searchable."""
    tree = make_tree(order=2)
    for k in [1, 2, 3]:
        tree.insert(k, f"v{k}".encode())

    # All three keys must still be findable
    assert tree.search(1) == b"v1"
    assert tree.search(2) == b"v2"
    assert tree.search(3) == b"v3"


def test_leaf_split_root_becomes_index():
    """When the root leaf splits, the new root must be an IndexPage."""
    tree = make_tree(order=2)
    for k in [1, 2, 3]:
        tree.insert(k, f"v{k}".encode())

    root = tree._pager.get_page(tree._root_id)
    assert isinstance(root, IndexPage)


def test_leaf_split_root_has_one_key():
    tree = make_tree(order=2)
    for k in [1, 2, 3]:
        tree.insert(k, f"v{k}".encode())

    root = tree._pager.get_page(tree._root_id)
    assert root.num_keys() == 1


# ---------------------------------------------------------------------------
# Root split → tree grows a level
# ---------------------------------------------------------------------------

def test_root_split_all_keys_searchable():
    """
    With order=2:
      3 inserts → 1st leaf split (root becomes IndexPage with 1 key)
      Insert enough to fill the root index and force a 2nd-level split.
    """
    tree = make_tree(order=2)
    keys = list(range(1, 10))
    for k in keys:
        tree.insert(k, f"v{k}".encode())

    for k in keys:
        assert tree.search(k) == f"v{k}".encode()


def test_root_split_root_is_still_index():
    tree = make_tree(order=2)
    for k in range(1, 10):
        tree.insert(k, f"v{k}".encode())

    root = tree._pager.get_page(tree._root_id)
    assert isinstance(root, IndexPage)


# ---------------------------------------------------------------------------
# Larger tree — multi-level splits
# ---------------------------------------------------------------------------

def test_many_inserts_all_searchable():
    tree = make_tree(order=4)
    n    = 100
    for k in range(n):
        tree.insert(k, f"record_{k}".encode())
    for k in range(n):
        assert tree.search(k) == f"record_{k}".encode()


def test_reverse_order_insert():
    tree = make_tree(order=4)
    keys = list(range(50, 0, -1))
    for k in keys:
        tree.insert(k, f"v{k}".encode())
    for k in keys:
        assert tree.search(k) == f"v{k}".encode()


def test_random_order_insert():
    tree = make_tree(order=4)
    keys = list(range(1, 51))
    random.seed(42)
    random.shuffle(keys)
    for k in keys:
        tree.insert(k, f"v{k}".encode())
    for k in keys:
        assert tree.search(k) == f"v{k}".encode()


# ---------------------------------------------------------------------------
# range_scan — same leaf
# ---------------------------------------------------------------------------

def test_range_scan_empty():
    tree = make_tree()
    assert tree.range_scan(1, 10) == []


def test_range_scan_single_leaf_full_range():
    tree = make_tree(order=100)   # large order → no splits
    for k in [1, 2, 3, 4, 5]:
        tree.insert(k, f"v{k}".encode())
    result = tree.range_scan(1, 5)
    assert [k for k, _ in result] == [1, 2, 3, 4, 5]


def test_range_scan_single_leaf_partial():
    tree = make_tree(order=100)
    for k in [1, 2, 3, 4, 5]:
        tree.insert(k, f"v{k}".encode())
    result = tree.range_scan(2, 4)
    assert [k for k, _ in result] == [2, 3, 4]


def test_range_scan_no_match():
    tree = make_tree(order=100)
    for k in [1, 2, 3]:
        tree.insert(k, b"x")
    assert tree.range_scan(10, 20) == []


def test_range_scan_start_below_min():
    tree = make_tree(order=100)
    for k in [5, 10, 15]:
        tree.insert(k, f"v{k}".encode())
    result = tree.range_scan(0, 10)
    assert [k for k, _ in result] == [5, 10]


def test_range_scan_end_above_max():
    tree = make_tree(order=100)
    for k in [5, 10, 15]:
        tree.insert(k, f"v{k}".encode())
    result = tree.range_scan(10, 99)
    assert [k for k, _ in result] == [10, 15]


# ---------------------------------------------------------------------------
# range_scan — across multiple leaves (linked list traversal)
# ---------------------------------------------------------------------------

def test_range_scan_across_leaves():
    """
    order=2 forces splits and creates multiple leaves linked together.
    The range scan must follow next_page_id links.
    """
    tree = make_tree(order=2)
    for k in range(1, 11):
        tree.insert(k, f"v{k}".encode())

    result = tree.range_scan(3, 8)
    assert [k for k, _ in result] == [3, 4, 5, 6, 7, 8]
    for k, v in result:
        assert v == f"v{k}".encode()


def test_range_scan_full_tree():
    tree  = make_tree(order=2)
    keys  = list(range(1, 21))
    for k in keys:
        tree.insert(k, f"v{k}".encode())

    result = tree.range_scan(1, 20)
    assert [k for k, _ in result] == list(range(1, 21))


def test_range_scan_large_tree():
    tree = make_tree(order=4)
    n    = 200
    for k in range(n):
        tree.insert(k, f"r{k}".encode())

    lo, hi = 50, 149
    result  = tree.range_scan(lo, hi)
    assert [k for k, _ in result] == list(range(lo, hi + 1))
    for k, v in result:
        assert v == f"r{k}".encode()


# ---------------------------------------------------------------------------
# Leaf linked list integrity
# ---------------------------------------------------------------------------

def test_leaf_linked_list_covers_all_keys():
    """
    Walk the leaf linked list from the leftmost leaf to the end and verify
    every inserted key is found exactly once, in sorted order.
    """
    tree = make_tree(order=3)
    keys = list(range(1, 31))
    random.seed(7)
    random.shuffle(keys)
    for k in keys:
        tree.insert(k, f"v{k}".encode())

    # Find the leftmost leaf by searching for key 0 (below all keys)
    leaf = tree._find_leaf(0)
    collected = []
    while leaf is not None:
        collected.extend(k for k, _ in leaf._entries())
        leaf = (tree._pager.get_page(leaf.next_page_id)
                if leaf.next_page_id is not None else None)

    assert collected == sorted(keys)
    assert len(set(collected)) == len(keys)   # no duplicates


# ---------------------------------------------------------------------------
# record.py integration
# ---------------------------------------------------------------------------

def test_insert_encoded_records():
    from src.record import encode_record, decode_record

    tree = make_tree(order=4)
    rows = {
        1: [1,  "alice", None],
        2: [2,  "bob",   42  ],
        3: [3,  "carol", None],
    }
    for k, row in rows.items():
        tree.insert(k, encode_record(row))

    for k, expected in rows.items():
        raw = tree.search(k)
        assert raw is not None
        assert decode_record(raw) == expected
