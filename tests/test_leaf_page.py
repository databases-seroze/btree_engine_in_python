"""
Unit tests for LeafPage.

Covers: insert, search, update, split, linked-list stitching, is_full.
"""

import pytest
from src.bplus_page import LeafPage


def make_leaf(page_id=0, max_keys=4):
    return LeafPage(page_id=page_id, max_keys=max_keys)


# ---------------------------------------------------------------------------
# insert + search
# ---------------------------------------------------------------------------

def test_search_empty_returns_none():
    leaf = make_leaf()
    assert leaf.search(1) is None


def test_insert_and_search():
    leaf = make_leaf()
    leaf.insert(10, b"alice")
    assert leaf.search(10) == b"alice"


def test_insert_multiple_sorted_search():
    leaf = make_leaf()
    leaf.insert(30, b"carol")
    leaf.insert(10, b"alice")
    leaf.insert(20, b"bob")

    assert leaf.search(10) == b"alice"
    assert leaf.search(20) == b"bob"
    assert leaf.search(30) == b"carol"


def test_search_missing_key_returns_none():
    leaf = make_leaf()
    leaf.insert(10, b"alice")
    assert leaf.search(99) is None


def test_insert_updates_existing_key():
    leaf = make_leaf()
    leaf.insert(10, b"old")
    leaf.insert(10, b"new")
    assert leaf.search(10) == b"new"
    assert leaf.num_keys() == 1   # no duplicate slot


# ---------------------------------------------------------------------------
# entries are kept sorted
# ---------------------------------------------------------------------------

def test_entries_sorted():
    leaf = make_leaf()
    for k in [5, 3, 1, 4, 2]:
        leaf.insert(k, f"v{k}".encode())
    keys = [k for k, _ in leaf._entries()]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------

def test_is_full_below_limit():
    leaf = make_leaf(max_keys=3)
    for k in range(3):
        leaf.insert(k, b"x")
    assert not leaf.is_full()


def test_is_full_above_limit():
    leaf = make_leaf(max_keys=3)
    for k in range(4):          # one past max_keys
        leaf.insert(k, b"x")
    assert leaf.is_full()


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def test_split_divides_entries_evenly():
    leaf  = make_leaf(page_id=0, max_keys=4)
    right = make_leaf(page_id=1, max_keys=4)

    for k in [1, 2, 3, 4]:
        leaf.insert(k, f"v{k}".encode())

    split_key = leaf.split(right)

    left_keys  = [k for k, _ in leaf._entries()]
    right_keys = [k for k, _ in right._entries()]

    assert split_key == right_keys[0]
    assert left_keys  == [1, 2]
    assert right_keys == [3, 4]


def test_split_copies_key_to_right_leaf():
    """split_key must stay as the first entry of the right leaf (B+Tree copy-up rule)."""
    leaf  = make_leaf(page_id=0, max_keys=4)
    right = make_leaf(page_id=1, max_keys=4)

    for k in [10, 20, 30, 40]:
        leaf.insert(k, b"data")

    split_key = leaf.split(right)

    # split_key is copied up — it is still the first key of right
    right_first_key = right._entries()[0][0]
    assert split_key == right_first_key


def test_split_stitches_linked_list():
    """After split: self.next_page_id == right.page_id == old next."""
    leaf  = make_leaf(page_id=0, max_keys=4)
    right = make_leaf(page_id=1, max_keys=4)

    leaf.next_page_id = 99   # simulate an existing right sibling

    for k in [1, 2, 3, 4]:
        leaf.insert(k, b"x")

    leaf.split(right)

    assert leaf.next_page_id  == right.page_id   # self → right
    assert right.next_page_id == 99              # right → old next


def test_split_preserves_all_data():
    leaf  = make_leaf(page_id=0, max_keys=6)
    right = make_leaf(page_id=1, max_keys=6)

    data = {k: f"val{k}".encode() for k in range(1, 7)}
    for k, v in data.items():
        leaf.insert(k, v)

    leaf.split(right)

    combined = dict(leaf._entries()) | dict(right._entries())
    assert combined == data
