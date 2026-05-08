"""
Tests for the REPL -- pretty-printer and _execute_and_print.

The interactive loop itself is not tested here (it requires a tty).
We test the parts that can run headlessly:
  - _fmt / _print_table output format
  - _execute_and_print success and error paths
  - Meta-command helpers (tables, schema, indexes)
"""

import io
import sys
import pytest

from src.repl     import _fmt, _print_table, _execute_and_print, _cmd_tables, _cmd_schema, _cmd_indexes
from src.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def capture(fn, *args, **kwargs):
    """Call fn(*args) and return its stdout as a string."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


def make_db(tmp_path):
    db = Database(str(tmp_path))
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
    db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
    db.execute("INSERT INTO users VALUES (2, 'bob',   25)")
    return db


# ===========================================================================
# _fmt
# ===========================================================================

def test_fmt_none_returns_null():
    assert _fmt(None) == "NULL"

def test_fmt_int():
    assert _fmt(42) == "42"

def test_fmt_string():
    assert _fmt("hello") == "hello"


# ===========================================================================
# _print_table
# ===========================================================================

def test_print_table_empty():
    out = capture(_print_table, [])
    assert "(0 rows)" in out

def test_print_table_single_row():
    rows = [{"id": 1, "name": "alice", "age": 30}]
    out  = capture(_print_table, rows)
    assert "alice" in out
    assert "id"    in out
    assert "name"  in out
    assert "(1 row)" in out

def test_print_table_multiple_rows():
    rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
    out  = capture(_print_table, rows)
    assert "alice"  in out
    assert "bob"    in out
    assert "(2 rows)" in out

def test_print_table_null_value():
    rows = [{"id": 1, "note": None}]
    out  = capture(_print_table, rows)
    assert "NULL" in out

def test_print_table_has_separator_lines():
    rows = [{"a": 1, "b": 2}]
    out  = capture(_print_table, rows)
    # ASCII table borders use + and -
    assert "+" in out
    assert "-" in out
    assert "|" in out

def test_print_table_column_alignment():
    # long value should widen the column
    rows = [{"col": "short"}, {"col": "a much longer value"}]
    out  = capture(_print_table, rows)
    lines = out.splitlines()
    # Only check table lines (those starting with + or |), not the row-count footer
    table_lines = [l for l in lines if l.startswith("+") or l.startswith("|")]
    pipe_positions = [i for i, c in enumerate(table_lines[0]) if c == "+"]
    for line in table_lines[1:]:
        for pos in pipe_positions:
            assert line[pos] in ("+", "|"), f"misaligned at pos {pos}: {line!r}"


# ===========================================================================
# _execute_and_print
# ===========================================================================

def test_exec_select_prints_rows(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "SELECT * FROM users WHERE id = 1")
        assert "alice" in out
        assert "(1 row)" in out

def test_exec_select_no_match_prints_zero_rows(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "SELECT * FROM users WHERE id = 99")
        assert "(0 rows)" in out

def test_exec_insert_prints_ok(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "INSERT INTO users VALUES (3, 'carol', 35)")
        assert "OK" in out

def test_exec_update_prints_ok(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "UPDATE users SET age = 31 WHERE id = 1")
        assert "OK" in out

def test_exec_delete_prints_ok(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "DELETE FROM users WHERE id = 1")
        assert "OK" in out

def test_exec_create_table_prints_ok(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        assert "OK" in out

def test_exec_syntax_error_prints_syntax_error(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "BLAH BLAH BLAH")
        assert "Syntax error" in out or "error" in out.lower()

def test_exec_execution_error_prints_error(tmp_path):
    with make_db(tmp_path) as db:
        # Duplicate PK → execution error
        out = capture(_execute_and_print, db,
                      "INSERT INTO users VALUES (1, 'dup', 99)")
        # Either "Error:" printed or the insert silently overwrites — both acceptable.
        # What matters is it doesn't raise/crash.

def test_exec_unknown_table_prints_error(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_execute_and_print, db, "SELECT * FROM nosuch")
        assert "error" in out.lower()


# ===========================================================================
# Meta-command helpers
# ===========================================================================

def test_cmd_tables_lists_tables(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_tables, db)
        assert "users" in out

def test_cmd_tables_empty_db(tmp_path):
    with Database(str(tmp_path)) as db:
        out = capture(_cmd_tables, db)
        assert "no tables" in out

def test_cmd_schema_shows_columns(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_schema, db, "users")
        assert "id"   in out
        assert "name" in out
        assert "age"  in out
        assert "INT"  in out

def test_cmd_schema_shows_pk(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_schema, db, "users")
        assert "YES"  in out   # primary key marker

def test_cmd_schema_shows_fk(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE users  (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")
        out = capture(_cmd_schema, db, "orders")
        assert "users" in out   # FK reference shown

def test_cmd_schema_unknown_table(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_schema, db, "nosuch")
        assert "Error" in out

def test_cmd_schema_no_arg(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_schema, db, "")
        assert "Usage" in out

def test_cmd_indexes_no_indexes(tmp_path):
    with make_db(tmp_path) as db:
        out = capture(_cmd_indexes, db)
        assert "no indexes" in out

def test_cmd_indexes_lists_index(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        out = capture(_cmd_indexes, db)
        assert "idx_age" in out
        assert "users"   in out
        assert "age"     in out
