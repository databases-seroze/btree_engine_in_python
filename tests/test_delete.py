"""
Tests for BPlusTree.delete().

Each section documents the exact tree shape being used so it is clear
which underflow-handling case is being exercised.

order=4  →  max_keys=4, min_keys=2
order=2  →  max_keys=2, min_keys=1
"""

import random
import pytest
from src.bplus_tree  import BPlusTree
from src.bplus_page  import LeafPage, IndexPage


def make_tree(order=4):
    return BPlusTree(order=order)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_keys(tree) -> list[int]:
    """Collect every key in the tree via the leaf linked list."""
    leaf = tree._find_leaf(0)
    keys = []
    while leaf is not None:
        keys.extend(k for k, _ in leaf._entries())
        leaf = (tree._pager.get_page(leaf.next_page_id)
                if leaf.next_page_id is not None else None)
    return keys


def insert_many(tree, keys):
    for k in keys:
        tree.insert(k, f"v{k}".encode())


# ---------------------------------------------------------------------------
# Basic delete
# ---------------------------------------------------------------------------

def test_delete_returns_false_if_missing():
    tree = make_tree()
    assert tree.delete(99) is False


def test_delete_single_key():
    tree = make_tree()
    tree.insert(1, b"a")
    assert tree.delete(1) is True
    assert tree.search(1) is None


def test_delete_one_of_many_in_single_leaf():
    tree = make_tree(order=10)
    for k in [1, 2, 3, 4, 5]:
        tree.insert(k, f"v{k}".encode())
    assert tree.delete(3) is True
    assert tree.search(3) is None
    # others untouched
    for k in [1, 2, 4, 5]:
        assert tree.search(k) == f"v{k}".encode()


def test_delete_all_keys_leaves_empty_tree():
    tree = make_tree(order=4)
    for k in [1, 2, 3]:
        tree.insert(k, f"v{k}".encode())
    for k in [1, 2, 3]:
        tree.delete(k)
    assert all_keys(tree) == []


def test_delete_is_idempotent():
    tree = make_tree()
    tree.insert(5, b"x")
    assert tree.delete(5) is True
    assert tree.delete(5) is False


# ---------------------------------------------------------------------------
# Case 1 — simple delete, no underflow
#
# order=4, insert 1..8 → tree shape after inserts:
#   leaf0=[1,2]  leaf1=[3,4]  leaf2=[5,6,7,8]
#   root: keys=[3,5], children=[leaf0, leaf1, leaf2]
#
# Deleting 6 from leaf2 leaves leaf2=[5,7,8] (3 keys ≥ min=2) — no underflow.
# ---------------------------------------------------------------------------

def test_simple_delete_no_underflow():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 9))

    assert tree.delete(6) is True
    assert tree.search(6) is None

    for k in [1, 2, 3, 4, 5, 7, 8]:
        assert tree.search(k) == f"v{k}".encode()


# ---------------------------------------------------------------------------
# Case 2 — borrow from right sibling (leaf)
#
# Build a state where deleting from a leaf causes underflow but the
# right sibling has a spare.
#
# order=4, insert 1..9:
#   After all inserts the tree has ≥3 leaves; deleting the first key of
#   the leftmost leaf triggers a borrow.  We verify all remaining keys
#   are still found.
# ---------------------------------------------------------------------------

def test_borrow_from_right_leaf():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 10))

    # Find the leftmost leaf and delete one key to cause underflow,
    # while its right sibling still has spares.
    leftmost = tree._find_leaf(0)
    first_key = leftmost._entries()[0][0]

    assert tree.delete(first_key) is True
    assert tree.search(first_key) is None

    remaining = [k for k in range(1, 10) if k != first_key]
    for k in remaining:
        assert tree.search(k) == f"v{k}".encode()


def test_borrow_from_right_leaf_linked_list_intact():
    """After a borrow, range_scan must still return a contiguous sequence."""
    tree = make_tree(order=4)
    insert_many(tree, range(1, 10))

    leftmost  = tree._find_leaf(0)
    first_key = leftmost._entries()[0][0]
    tree.delete(first_key)

    expected = sorted(k for k in range(1, 10) if k != first_key)
    result   = tree.range_scan(0, 100)
    assert [k for k, _ in result] == expected


# ---------------------------------------------------------------------------
# Case 3 — borrow from left sibling (leaf)
#
# Insert keys so that the rightmost leaf underflows on a delete but its
# left sibling has a spare.
# ---------------------------------------------------------------------------

def test_borrow_from_left_leaf():
    tree = make_tree(order=4)
    # Insert 1..9, then delete from the right end to trigger left-borrow
    insert_many(tree, range(1, 10))

    rightmost_key = 9
    # Keep deleting from right until right leaf underflows while left has spare
    # Deleting 9 leaves leaf with fewer keys than min.
    assert tree.delete(rightmost_key) is True
    assert tree.search(rightmost_key) is None

    for k in range(1, rightmost_key):
        assert tree.search(k) == f"v{k}".encode()


# ---------------------------------------------------------------------------
# Case 4 — merge two leaf siblings → parent loses a separator key
#
# order=4, insert 1..8 → leaf0=[1,2], leaf1=[3,4], leaf2=[5,6,7,8]
# Delete 1 → leaf0=[2], underflow; right sibling leaf1=[3,4] is at min=2
# → can't borrow → merge leaf0+leaf1 → [2,3,4]
# → parent goes from keys=[3,5] to keys=[5]  (1 key, still ≥1 as root child)
# ---------------------------------------------------------------------------

def test_leaf_merge_parent_loses_key():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 9))   # 1..8

    assert tree.delete(1) is True
    assert tree.search(1) is None

    for k in range(2, 9):
        assert tree.search(k) == f"v{k}".encode()


def test_leaf_merge_range_scan_correct():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 9))
    tree.delete(1)

    result = tree.range_scan(1, 8)
    assert [k for k, _ in result] == list(range(2, 9))


# ---------------------------------------------------------------------------
# Case 5 — merge causes root collapse (tree height shrinks)
#
# order=2 forces early splits.  After inserting 1..3:
#   leaf0=[1]  leaf1=[2,3]   root: keys=[2], children=[leaf0, leaf1]
#
# Delete 1 → leaf0=[], underflow; leaf1=[2,3] is at min=1 → can't borrow
# → merge → merged leaf=[2,3]
# → root has 0 keys → root collapses to merged leaf
# ---------------------------------------------------------------------------

def test_root_collapse_after_merge():
    # order=2 → insert [1,2,3] produces:
    #   leaf0=[1]  leaf1=[2,3]   root(IndexPage): keys=[2]
    #
    # Delete 3 → leaf1=[2]   (both leaves now at min=1)
    # Delete 2 → leaf1=[]   underflow; leaf0=[1] is at min → can't borrow
    #            → merge leaf0+leaf1 → root.keys=[] → root collapses to leaf0
    tree = make_tree(order=2)
    insert_many(tree, [1, 2, 3])

    assert isinstance(tree._pager.get_page(tree._root_id), IndexPage)

    tree.delete(3)   # bring right leaf to minimum first
    tree.delete(2)   # now right leaf underflows, left is at min → merge

    assert isinstance(tree._pager.get_page(tree._root_id), LeafPage)
    assert tree.search(2) is None
    assert tree.search(3) is None
    assert tree.search(1) == b"v1"


def test_root_collapse_all_remaining_searchable():
    tree = make_tree(order=2)
    insert_many(tree, range(1, 8))

    to_delete = [1, 2]
    for k in to_delete:
        tree.delete(k)

    for k in range(3, 8):
        assert tree.search(k) == f"v{k}".encode()


# ---------------------------------------------------------------------------
# Case 6 — index node borrow from right
#
# Use a deep enough tree (order=2, many keys) that after leaf operations
# an index node itself needs to borrow from its right index sibling.
# We verify correctness via all-keys scan.
# ---------------------------------------------------------------------------

def test_index_borrow_tree_stays_correct():
    tree = make_tree(order=2)
    keys = list(range(1, 20))
    insert_many(tree, keys)

    # Delete the first quarter of keys — this stresses the left side of the
    # tree and will exercise index-level borrows/merges.
    for k in keys[:5]:
        tree.delete(k)

    remaining = sorted(set(keys) - set(keys[:5]))
    assert all_keys(tree) == remaining
    for k in remaining:
        assert tree.search(k) == f"v{k}".encode()


# ---------------------------------------------------------------------------
# Case 7 — index node merge (multi-level propagation)
# ---------------------------------------------------------------------------

def test_index_merge_propagates_correctly():
    tree = make_tree(order=2)
    keys = list(range(1, 15))
    insert_many(tree, keys)

    # Delete enough keys to force index-level merges
    for k in keys[:7]:
        tree.delete(k)

    remaining = sorted(set(keys) - set(keys[:7]))
    assert all_keys(tree) == remaining


# ---------------------------------------------------------------------------
# Range scan after deletes
# ---------------------------------------------------------------------------

def test_range_scan_after_deletes():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 21))

    for k in [2, 5, 8, 11, 14, 17]:
        tree.delete(k)

    expected = [k for k in range(1, 21) if k not in {2, 5, 8, 11, 14, 17}]
    result   = tree.range_scan(1, 20)
    assert [k for k, _ in result] == expected


def test_range_scan_after_delete_empty_range():
    tree = make_tree(order=4)
    insert_many(tree, [1, 2, 3])
    tree.delete(2)
    assert tree.range_scan(2, 2) == []


# ---------------------------------------------------------------------------
# Re-insert after delete
# ---------------------------------------------------------------------------

def test_reinsert_after_delete():
    tree = make_tree(order=4)
    tree.insert(1, b"original")
    tree.delete(1)
    tree.insert(1, b"reinserted")
    assert tree.search(1) == b"reinserted"


def test_delete_and_reinsert_many():
    tree = make_tree(order=4)
    insert_many(tree, range(1, 21))
    for k in range(1, 11):
        tree.delete(k)
    for k in range(1, 11):
        tree.insert(k, f"new{k}".encode())
    for k in range(1, 21):
        assert tree.search(k) is not None


# ---------------------------------------------------------------------------
# Large random delete
# ---------------------------------------------------------------------------

def test_large_random_delete():
    tree = make_tree(order=4)
    n    = 200
    insert_many(tree, range(n))

    random.seed(0)
    to_delete = random.sample(range(n), 100)
    for k in to_delete:
        assert tree.delete(k) is True

    remaining = sorted(set(range(n)) - set(to_delete))
    assert all_keys(tree) == remaining
    for k in remaining:
        assert tree.search(k) == f"v{k}".encode()


def test_delete_all_keys_large():
    tree = make_tree(order=4)
    keys = list(range(50))
    insert_many(tree, keys)

    random.seed(1)
    random.shuffle(keys)
    for k in keys:
        assert tree.delete(k) is True

    assert all_keys(tree) == []
    assert tree.search(0) is None


# ---------------------------------------------------------------------------
# Leaf linked-list integrity after many deletes
# ---------------------------------------------------------------------------

def test_leaf_linked_list_after_deletes():
    """Every key reachable via the linked list must equal what search() returns."""
    tree = make_tree(order=3)
    insert_many(tree, range(1, 31))

    for k in range(1, 31, 2):   # delete all odd keys
        tree.delete(k)

    via_list = all_keys(tree)
    expected = list(range(2, 31, 2))
    assert via_list == expected

    for k in expected:
        assert tree.search(k) == f"v{k}".encode()
