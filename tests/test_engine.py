"""
Tests for Engine (put / get / delete / scan / file persistence).
"""

import pytest
from src.engine import Engine


# ---------------------------------------------------------------------------
# put / get
# ---------------------------------------------------------------------------

def test_get_missing_returns_none():
    db = Engine()
    assert db.get(1) is None
    db.close()


def test_put_and_get():
    db = Engine()
    db.put(1, [1, "alice", 30])
    assert db.get(1) == [1, "alice", 30]
    db.close()


def test_put_multiple_rows():
    db = Engine()
    rows = {
        1: [1, "alice", 30],
        2: [2, "bob",   25],
        3: [3, "carol", None],
    }
    for k, row in rows.items():
        db.put(k, row)
    for k, expected in rows.items():
        assert db.get(k) == expected
    db.close()


def test_put_updates_existing_row():
    db = Engine()
    db.put(1, [1, "old", 0])
    db.put(1, [1, "new", 99])
    assert db.get(1) == [1, "new", 99]
    db.close()


def test_put_null_and_int_and_string():
    db = Engine()
    db.put(42, [None, 0, ""])
    assert db.get(42) == [None, 0, ""]
    db.close()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_missing_returns_false():
    db = Engine()
    assert db.delete(99) is False
    db.close()


def test_delete_existing_returns_true():
    db = Engine()
    db.put(1, [1, "alice"])
    assert db.delete(1) is True
    db.close()


def test_get_after_delete_returns_none():
    db = Engine()
    db.put(1, [1, "alice"])
    db.delete(1)
    assert db.get(1) is None
    db.close()


def test_delete_only_removes_target_key():
    db = Engine()
    db.put(1, [1, "a"])
    db.put(2, [2, "b"])
    db.put(3, [3, "c"])
    db.delete(2)
    assert db.get(1) == [1, "a"]
    assert db.get(2) is None
    assert db.get(3) == [3, "c"]
    db.close()


# ---------------------------------------------------------------------------
# scan (basic)
# ---------------------------------------------------------------------------

def test_scan_empty_db():
    db = Engine()
    result = list(db.scan())
    assert result == []
    db.close()


def test_scan_full_range():
    db = Engine()
    for k in range(1, 6):
        db.put(k, [k, f"v{k}"])
    result = list(db.scan(1, 5))
    assert [k for k, _ in result] == [1, 2, 3, 4, 5]
    assert [row[1] for _, row in result] == ["v1", "v2", "v3", "v4", "v5"]
    db.close()


def test_scan_partial_range():
    db = Engine()
    for k in range(1, 11):
        db.put(k, [k])
    result = list(db.scan(3, 7))
    assert [k for k, _ in result] == [3, 4, 5, 6, 7]
    db.close()


def test_scan_start_above_all_keys():
    db = Engine()
    db.put(1, [1])
    db.put(2, [2])
    assert list(db.scan(99, 200)) == []
    db.close()


def test_scan_end_below_all_keys():
    db = Engine()
    db.put(10, [10])
    db.put(20, [20])
    assert list(db.scan(0, 5)) == []
    db.close()


def test_scan_single_match():
    db = Engine()
    for k in [1, 5, 10]:
        db.put(k, [k])
    result = list(db.scan(5, 5))
    assert [k for k, _ in result] == [5]
    db.close()


def test_scan_default_range_covers_everything():
    db = Engine()
    for k in range(1, 6):
        db.put(k, [k])
    result = list(db.scan())
    assert [k for k, _ in result] == [1, 2, 3, 4, 5]
    db.close()


# ---------------------------------------------------------------------------
# scan after deletes
# ---------------------------------------------------------------------------

def test_scan_after_delete():
    db = Engine()
    for k in range(1, 8):
        db.put(k, [k])
    db.delete(3)
    db.delete(5)
    result = list(db.scan(1, 7))
    assert [k for k, _ in result] == [1, 2, 4, 6, 7]
    db.close()


# ---------------------------------------------------------------------------
# scan with splits (multiple leaves)
# ---------------------------------------------------------------------------

def test_scan_across_many_leaves():
    db = Engine(order=4)
    n = 50
    for k in range(1, n + 1):
        db.put(k, [k, f"row{k}"])
    result = list(db.scan(1, n))
    assert len(result) == n
    assert [k for k, _ in result] == list(range(1, n + 1))
    db.close()


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------

def test_context_manager_closes_on_exit():
    with Engine() as db:
        db.put(1, [1, "alice"])
        assert db.get(1) == [1, "alice"]
    # No exception → __exit__ called close() successfully


def test_context_manager_file_backed(tmp_path):
    path = str(tmp_path / "ctx.db")
    with Engine(path) as db:
        db.put(1, [1, "alice"])

    with Engine(path) as db:
        assert db.get(1) == [1, "alice"]


# ---------------------------------------------------------------------------
# file-backed persistence
# ---------------------------------------------------------------------------

def test_file_backed_put_flush_reopen_get(tmp_path):
    path = str(tmp_path / "test.db")
    with Engine(path, order=10) as db:
        db.put(1, [1, "alice", 30])
        db.put(2, [2, "bob",   25])

    with Engine(path) as db:
        assert db.get(1) == [1, "alice", 30]
        assert db.get(2) == [2, "bob",   25]


def test_file_backed_delete_persists(tmp_path):
    path = str(tmp_path / "test.db")
    with Engine(path, order=10) as db:
        db.put(1, [1, "alice"])
        db.put(2, [2, "bob"])
        db.delete(1)

    with Engine(path) as db:
        assert db.get(1) is None
        assert db.get(2) == [2, "bob"]


def test_file_backed_scan_after_reopen(tmp_path):
    path = str(tmp_path / "test.db")
    with Engine(path, order=4) as db:
        for k in range(1, 11):
            db.put(k, [k, f"r{k}"])

    with Engine(path) as db:
        result = list(db.scan(3, 7))
        assert [k for k, _ in result] == [3, 4, 5, 6, 7]


def test_file_backed_large_dataset(tmp_path):
    path = str(tmp_path / "large.db")
    n    = 300
    with Engine(path, order=20) as db:
        for k in range(n):
            db.put(k, [k, f"name_{k}", None])

    with Engine(path) as db:
        for k in range(n):
            assert db.get(k) == [k, f"name_{k}", None]
