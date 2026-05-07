"""
Unit tests for IndexPage.

Covers: find_child routing, insert_key, is_full, split (push-up rule).
"""

import pytest
from src.bplus_page import IndexPage


def make_index(page_id=0, max_keys=4):
    return IndexPage(page_id=page_id, max_keys=max_keys)


def build_index(keys, children, page_id=0, max_keys=4):
    """Helper: return an IndexPage pre-loaded with keys and children."""
    page = make_index(page_id=page_id, max_keys=max_keys)
    page.keys     = list(keys)
    page.children = list(children)
    return page


# ---------------------------------------------------------------------------
# find_child routing
# ---------------------------------------------------------------------------

def test_find_child_single_key_left():
    # keys=[10], children=[p0, p1]
    page = build_index([10], [0, 1])
    assert page.find_child(5) == 0    # 5 < 10 → p0


def test_find_child_single_key_right():
    page = build_index([10], [0, 1])
    assert page.find_child(10) == 1   # 10 >= 10 → p1
    assert page.find_child(15) == 1


def test_find_child_multiple_keys():
    # keys=[10, 20, 30], children=[p0, p1, p2, p3]
    page = build_index([10, 20, 30], [0, 1, 2, 3])
    assert page.find_child(5)  == 0   # < 10
    assert page.find_child(10) == 1   # >= 10, < 20
    assert page.find_child(15) == 1
    assert page.find_child(20) == 2   # >= 20, < 30
    assert page.find_child(30) == 3   # >= 30
    assert page.find_child(99) == 3


def test_find_child_boundary_exact_match():
    page = build_index([5, 10, 15], [0, 1, 2, 3])
    assert page.find_child(5)  == 1
    assert page.find_child(10) == 2
    assert page.find_child(15) == 3


# ---------------------------------------------------------------------------
# insert_key
# ---------------------------------------------------------------------------

def test_insert_key_into_empty():
    page = make_index()
    page.children = [0]          # leftmost child exists
    page.insert_key(10, 1)
    assert page.keys     == [10]
    assert page.children == [0, 1]


def test_insert_key_appends_rightmost():
    page = build_index([10, 20], [0, 1, 2])
    page.insert_key(30, 3)
    assert page.keys     == [10, 20, 30]
    assert page.children == [0, 1, 2, 3]


def test_insert_key_prepends_leftmost():
    page = build_index([20, 30], [0, 1, 2])
    page.insert_key(10, 99)
    assert page.keys     == [10, 20, 30]
    assert page.children == [0, 99, 1, 2]


def test_insert_key_in_middle():
    page = build_index([10, 30], [0, 1, 2])
    page.insert_key(20, 99)
    assert page.keys     == [10, 20, 30]
    assert page.children == [0, 1, 99, 2]


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------

def test_is_full_at_limit():
    page = make_index(max_keys=3)
    page.keys = [1, 2, 3]
    assert not page.is_full()


def test_is_full_over_limit():
    page = make_index(max_keys=3)
    page.keys = [1, 2, 3, 4]    # one past max
    assert page.is_full()


# ---------------------------------------------------------------------------
# split — push-up rule
# ---------------------------------------------------------------------------

def test_split_pushes_middle_key_up():
    """The middle key must NOT remain in either child after split."""
    left  = build_index([10, 20, 30, 40, 50], [0, 1, 2, 3, 4, 5], page_id=0)
    right = make_index(page_id=1)

    push_up = left.split(right)

    assert push_up == 30                 # middle key pushed up
    assert push_up not in left.keys      # NOT in left
    assert push_up not in right.keys     # NOT in right (push-up, not copy-up)


def test_split_left_gets_lower_half():
    left  = build_index([10, 20, 30, 40, 50], [0, 1, 2, 3, 4, 5], page_id=0)
    right = make_index(page_id=1)
    left.split(right)

    assert left.keys     == [10, 20]
    assert left.children == [0, 1, 2]


def test_split_right_gets_upper_half():
    left  = build_index([10, 20, 30, 40, 50], [0, 1, 2, 3, 4, 5], page_id=0)
    right = make_index(page_id=1)
    left.split(right)

    assert right.keys     == [40, 50]
    assert right.children == [3, 4, 5]


def test_split_children_count_invariant():
    """After split, len(children) == len(keys) + 1 for both halves."""
    left  = build_index([10, 20, 30, 40], [0, 1, 2, 3, 4], page_id=0)
    right = make_index(page_id=1)
    left.split(right)

    assert len(left.children)  == len(left.keys)  + 1
    assert len(right.children) == len(right.keys) + 1
