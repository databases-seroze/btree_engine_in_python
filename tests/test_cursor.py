"""
Tests for Cursor (next / prev / reset / iterator protocol).
"""

import pytest
from src.engine import Engine
from src.cursor import Cursor


def make_db(rows: dict, order=10) -> Engine:
    """Return an in-memory Engine pre-loaded with {key: row} rows."""
    db = Engine(order=order)
    for k, row in rows.items():
        db.put(k, row)
    return db


# ---------------------------------------------------------------------------
# Empty / boundary
# ---------------------------------------------------------------------------

def test_cursor_empty_tree():
    db = Engine()
    c  = db.scan()
    assert c.key   is None
    assert c.value is None
    db.close()


def test_cursor_range_with_no_matches():
    db = make_db({1: [1], 2: [2], 3: [3]})
    c  = db.scan(10, 20)
    assert c.key is None
    db.close()


def test_cursor_single_entry_in_range():
    db = make_db({1: [1, "a"], 5: [5, "b"], 10: [10, "c"]})
    c  = db.scan(5, 5)
    assert c.key   == 5
    assert c.value == [5, "b"]
    db.close()


# ---------------------------------------------------------------------------
# next()
# ---------------------------------------------------------------------------

def test_cursor_next_advances():
    db = make_db({1: [1], 2: [2], 3: [3]})
    c  = db.scan(1, 3)
    assert c.key == 1
    c.next()
    assert c.key == 2
    c.next()
    assert c.key == 3
    db.close()


def test_cursor_next_returns_true_while_valid():
    db = make_db({1: [1], 2: [2]})
    c  = db.scan(1, 2)
    assert c.next() is True    # moved to key=2
    assert c.next() is False   # exhausted
    db.close()


def test_cursor_next_on_exhausted_returns_false():
    db = make_db({1: [1]})
    c  = db.scan(1, 1)
    c.next()
    assert c.next() is False
    db.close()


def test_cursor_next_stops_at_end_bound():
    db = make_db({1: [1], 2: [2], 3: [3], 4: [4]})
    c  = db.scan(1, 2)
    assert c.key == 1
    c.next()
    assert c.key == 2
    c.next()
    assert c.key is None   # 3 and 4 are outside range
    db.close()


# ---------------------------------------------------------------------------
# prev()
# ---------------------------------------------------------------------------

def test_cursor_prev_at_first_returns_false():
    db = make_db({1: [1], 2: [2], 3: [3]})
    c  = db.scan(1, 3)
    assert c.key == 1
    assert c.prev() is False
    assert c.key == 1     # position unchanged
    db.close()


def test_cursor_prev_moves_backward():
    db = make_db({1: [1], 2: [2], 3: [3]})
    c  = db.scan(1, 3)
    c.next()              # key=2
    assert c.key == 2
    c.prev()              # back to key=1
    assert c.key == 1
    db.close()


def test_cursor_prev_multiple_steps():
    db = make_db({k: [k] for k in range(1, 6)})
    c  = db.scan(1, 5)
    for _ in range(4):
        c.next()          # advance to key=5
    assert c.key == 5
    c.prev();  assert c.key == 4
    c.prev();  assert c.key == 3
    c.prev();  assert c.key == 2
    c.prev();  assert c.key == 1
    assert c.prev() is False
    db.close()


def test_cursor_prev_respects_start_bound():
    db = make_db({1: [1], 2: [2], 3: [3], 4: [4], 5: [5]})
    c  = db.scan(3, 5)    # range [3..5]
    assert c.key == 3
    assert c.prev() is False   # 3 is the start of this range
    db.close()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_cursor_reset_returns_to_start():
    db = make_db({1: [1], 2: [2], 3: [3]})
    c  = db.scan(1, 3)
    c.next()
    c.next()              # key=3
    c.reset()
    assert c.key == 1
    db.close()


def test_cursor_reset_on_empty_returns_false():
    db = Engine()
    c  = db.scan()
    assert c.reset() is False
    db.close()


def test_cursor_reset_after_exhaustion():
    db = make_db({1: [1], 2: [2]})
    c  = db.scan(1, 2)
    list(c)               # exhaust the cursor
    assert c.key is None
    c.reset()
    assert c.key == 1
    db.close()


# ---------------------------------------------------------------------------
# Iterator protocol
# ---------------------------------------------------------------------------

def test_cursor_for_loop():
    db = make_db({1: [1, "a"], 2: [2, "b"], 3: [3, "c"]})
    result = list(db.scan(1, 3))
    assert result == [(1, [1, "a"]), (2, [2, "b"]), (3, [3, "c"])]
    db.close()


def test_cursor_for_loop_partial_range():
    db = make_db({k: [k] for k in range(1, 11)})
    keys = [k for k, _ in db.scan(4, 7)]
    assert keys == [4, 5, 6, 7]
    db.close()


def test_cursor_list_conversion():
    db = make_db({k: [k, f"name{k}"] for k in range(1, 6)})
    rows = list(db.scan())
    assert len(rows) == 5
    assert rows[0] == (1, [1, "name1"])
    assert rows[-1] == (5, [5, "name5"])
    db.close()


def test_cursor_empty_range_for_loop():
    db = make_db({1: [1], 2: [2]})
    result = [k for k, _ in db.scan(10, 20)]
    assert result == []
    db.close()


# ---------------------------------------------------------------------------
# Multi-leaf traversal
# ---------------------------------------------------------------------------

def test_cursor_spans_multiple_leaves():
    db = Engine(order=4)
    n  = 40
    for k in range(1, n + 1):
        db.put(k, [k, f"v{k}"])
    keys = [k for k, _ in db.scan(1, n)]
    assert keys == list(range(1, n + 1))
    db.close()


def test_cursor_prev_across_leaves():
    """prev() must cross a leaf boundary correctly."""
    db = Engine(order=4)
    for k in range(1, 20):
        db.put(k, [k])
    c = db.scan(1, 19)

    # Advance past the first leaf (order=4 → each leaf ~2 keys)
    for _ in range(6):
        c.next()

    key_after  = c.key
    c.prev()
    key_before = c.key
    assert key_before == key_after - 1
    db.close()


def test_cursor_next_then_prev_round_trip():
    db = Engine(order=4)
    for k in range(1, 30):
        db.put(k, [k])

    c = db.scan(1, 29)
    visited_forward = []
    while c.key is not None:
        visited_forward.append(c.key)
        c.next()

    # Walk backward from the last valid position
    c.reset()
    for _ in range(len(visited_forward) - 1):
        c.next()
    # c is now at the last key

    visited_backward = [c.key]
    while c.prev():
        visited_backward.append(c.key)

    assert visited_forward == list(reversed(visited_backward))
    db.close()


# ---------------------------------------------------------------------------
# Values are correctly decoded
# ---------------------------------------------------------------------------

def test_cursor_value_decoded_correctly():
    db = Engine()
    rows = {
        10: [10, "alice", None],
        20: [20, "bob",   42  ],
    }
    for k, row in rows.items():
        db.put(k, row)

    for key, row in db.scan():
        assert row == rows[key]
    db.close()


# ---------------------------------------------------------------------------
# scan returns independent cursors
# ---------------------------------------------------------------------------

def test_two_independent_cursors():
    db = make_db({k: [k] for k in range(1, 6)})
    c1 = db.scan(1, 5)
    c2 = db.scan(1, 5)

    c1.next()   # c1 at key=2
    # c2 should still be at key=1
    assert c2.key == 1
    db.close()
