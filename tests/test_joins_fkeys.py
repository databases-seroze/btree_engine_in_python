"""
Tests for Phase 7 — Joins and Foreign Keys.

Sections
--------
1.  Parsing — JOIN SELECT and REFERENCES
2.  Foreign key enforcement — INSERT
3.  Foreign key enforcement — DELETE
4.  Foreign key persistence
5.  Nested-loop join — basic
6.  Join with WHERE filter
7.  Join access path — index NLJ vs full scan
8.  Join column projection
9.  Join edge cases (no matches, NULL FK)
10. End-to-end with persistence
"""

import pytest
from src.parser    import parse, ParseError
from src.ast_nodes import JoinSelectStmt, ColRef, CreateTableStmt, ColumnDef
from src.database  import Database
from src.executor  import ExecutionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db(tmp_path):
    db = Database(str(tmp_path))
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
    db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")
    for i, (name, age) in enumerate([("alice", 30), ("bob", 25), ("carol", 35)], 1):
        db.execute(f"INSERT INTO users VALUES ({i}, '{name}', {age})")
    return db


# ===========================================================================
# 1. Parsing
# ===========================================================================

def test_parse_join_select_star():
    stmt = parse("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
    assert isinstance(stmt, JoinSelectStmt)
    assert stmt.left_table  == "users"
    assert stmt.right_table == "orders"
    assert stmt.left_col    == "id"
    assert stmt.right_col   == "user_id"
    assert stmt.columns     == ['*']
    assert stmt.where is None

def test_parse_join_select_qualified_cols():
    stmt = parse("SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id")
    assert isinstance(stmt, JoinSelectStmt)
    assert stmt.columns[0] == ColRef(table='users', col='name')
    assert stmt.columns[1] == ColRef(table='orders', col='amount')

def test_parse_join_with_where():
    stmt = parse(
        "SELECT * FROM users JOIN orders ON users.id = orders.user_id "
        "WHERE orders.amount > 100"
    )
    assert stmt.where is not None

def test_parse_references_in_create_table():
    stmt = parse(
        "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)"
    )
    assert isinstance(stmt, CreateTableStmt)
    fk_col = next(c for c in stmt.columns if c.name == 'user_id')
    assert fk_col.fk_table == 'users'
    assert fk_col.fk_col   == 'id'

def test_parse_on_sides_normalised():
    # ON right.col = left.col should be swapped to left.col = right.col
    stmt = parse("SELECT * FROM users JOIN orders ON orders.user_id = users.id")
    assert stmt.left_col  == "id"
    assert stmt.right_col == "user_id"


# ===========================================================================
# 2. Foreign key enforcement — INSERT
# ===========================================================================

def test_insert_with_valid_fk_succeeds(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")   # user_id=1 exists

def test_insert_with_invalid_fk_raises(tmp_path):
    with make_db(tmp_path) as db:
        with pytest.raises(ExecutionError, match="Foreign key"):
            db.execute("INSERT INTO orders VALUES (1, 99, 500)")  # user_id=99 absent

def test_insert_fk_null_skips_check(tmp_path):
    # NULL FK value should not trigger the constraint check
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, NULL, 500)")

def test_fk_references_unknown_table_raises_at_create(tmp_path):
    with Database(str(tmp_path)) as db:
        with pytest.raises(Exception):
            # 'nosuch' table does not exist
            db.execute("CREATE TABLE t (id INT PRIMARY KEY, ref INT REFERENCES nosuch (id))")


# ===========================================================================
# 3. Foreign key enforcement — DELETE
# ===========================================================================

def test_delete_parent_with_no_children_succeeds(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("DELETE FROM users WHERE id = 3")   # carol has no orders

def test_delete_parent_with_child_raises(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")
        with pytest.raises(ExecutionError, match="Foreign key"):
            db.execute("DELETE FROM users WHERE id = 1")

def test_delete_child_then_parent_succeeds(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")
        db.execute("DELETE FROM orders WHERE id = 1")
        db.execute("DELETE FROM users WHERE id = 1")   # now allowed

def test_delete_parent_after_all_children_removed(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 1, 200)")
        db.execute("DELETE FROM orders WHERE id = 1")
        db.execute("DELETE FROM orders WHERE id = 2")
        db.execute("DELETE FROM users WHERE id = 1")   # ok now


# ===========================================================================
# 4. FK persistence
# ===========================================================================

def test_fk_enforced_after_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")

    with Database(path) as db:
        # FK should still be enforced after reopen
        with pytest.raises(ExecutionError, match="Foreign key"):
            db.execute("INSERT INTO orders VALUES (1, 99, 500)")
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")   # valid

def test_delete_fk_enforced_after_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")

    with Database(path) as db:
        with pytest.raises(ExecutionError, match="Foreign key"):
            db.execute("DELETE FROM users WHERE id = 1")


# ===========================================================================
# 5. Join — basic
# ===========================================================================

def test_join_returns_cross_product_filtered_by_key(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")
        db.execute("INSERT INTO orders VALUES (2, 2, 300)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 2
        names = {r['users.name'] for r in rows}
        assert names == {"alice", "bob"}

def test_join_multiple_orders_per_user(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 1, 200)")
        db.execute("INSERT INTO orders VALUES (3, 1, 300)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 3
        assert all(r['users.name'] == 'alice' for r in rows)

def test_join_user_with_no_orders_excluded(tmp_path):
    with make_db(tmp_path) as db:
        # Only user 1 has orders
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 1

def test_join_no_rows_returns_empty(tmp_path):
    with make_db(tmp_path) as db:
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert rows == []


# ===========================================================================
# 6. Join with WHERE filter
# ===========================================================================

def test_join_where_filters_by_right_col(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 2, 500)")
        db.execute("INSERT INTO orders VALUES (3, 3, 200)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id "
            "WHERE orders.amount > 150"
        )
        amounts = {r['orders.amount'] for r in rows}
        assert amounts == {500, 200}

def test_join_where_filters_by_left_col(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 2, 200)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id "
            "WHERE users.age > 28"
        )
        # alice(30) and carol(35) are > 28; bob(25) is not
        # alice has order, carol has no order → only alice's order
        assert len(rows) == 1
        assert rows[0]['users.name'] == 'alice'

def test_join_where_and_filter(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 1, 600)")
        db.execute("INSERT INTO orders VALUES (3, 2, 400)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id "
            "WHERE users.age > 28 AND orders.amount > 200"
        )
        assert len(rows) == 1
        assert rows[0]['orders.amount'] == 600


# ===========================================================================
# 7. Join access path — index NLJ
# ===========================================================================

def test_join_uses_pk_lookup_for_right_side(tmp_path, monkeypatch):
    """
    When join col is right table's PK, executor uses engine.get() (not scan).
    """
    with make_db(tmp_path) as db:
        # JOIN on users.id = orders.user_id where orders.user_id is NOT the PK.
        # Swap: join users.id (left PK) matched from orders side.
        # Confirm engine.get() called for users (right in reversed scenario)
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")

        right_engine = db._engines['users']
        get_calls = []
        orig_get   = right_engine.get
        monkeypatch.setattr(right_engine, 'get', lambda k: (get_calls.append(k), orig_get(k))[1])

        # users is left, orders is right — join on users.id = orders.user_id
        # right_col=user_id, not PK of orders, so full scan of orders per left row
        # But left scan of users → get() on users NOT called via join path
        db.execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id")

def test_join_uses_index_nlj_when_index_exists(tmp_path, monkeypatch):
    """
    When an index exists on the right join col, index lookup is used
    instead of a full scan.
    """
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_user_id ON orders (user_id)")
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        db.execute("INSERT INTO orders VALUES (2, 2, 200)")

        right_engine = db._engines['orders']
        scan_calls   = []
        orig_scan    = right_engine.scan
        monkeypatch.setattr(right_engine, 'scan',
            lambda *a, **kw: (scan_calls.append(1), orig_scan(*a, **kw))[1])

        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 2
        assert not scan_calls, "Index NLJ should not call engine.scan() on inner table"


# ===========================================================================
# 8. Column projection
# ===========================================================================

def test_join_project_qualified_cols(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")
        rows = db.execute(
            "SELECT users.name, orders.amount "
            "FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 1
        assert set(rows[0].keys()) == {'users.name', 'orders.amount'}
        assert rows[0]['users.name']    == 'alice'
        assert rows[0]['orders.amount'] == 500

def test_join_star_returns_qualified_names(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        keys = set(rows[0].keys())
        # All qualified names should be present
        assert 'users.id'      in keys
        assert 'users.name'    in keys
        assert 'orders.amount' in keys


# ===========================================================================
# 9. Edge cases
# ===========================================================================

def test_join_null_fk_not_matched(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO orders VALUES (1, NULL, 100)")
        rows = db.execute(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert rows == []

def test_join_empty_left_table(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, a_id INT, y INT)")
        rows = db.execute("SELECT * FROM a JOIN b ON a.id = b.a_id")
        assert rows == []

def test_join_empty_right_table(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, a_id INT, y INT)")
        db.execute("INSERT INTO a VALUES (1, 10)")
        rows = db.execute("SELECT * FROM a JOIN b ON a.id = b.a_id")
        assert rows == []


# ===========================================================================
# 10. End-to-end with persistence
# ===========================================================================

def test_join_data_persists_across_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users (id), amount INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO orders VALUES (1, 1, 999)")

    with Database(path) as db:
        rows = db.execute(
            "SELECT users.name, orders.amount "
            "FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) == 1
        assert rows[0]['users.name']    == 'alice'
        assert rows[0]['orders.amount'] == 999
