"""
Tests for secondary indexes (Phase 6).

Sections
--------
1.  CREATE INDEX / DROP INDEX — parsing and catalog
2.  Index used for equality lookup (access path)
3.  Index used for range lookup
4.  Index kept in sync — INSERT
5.  Index kept in sync — DELETE
6.  Index kept in sync — UPDATE (indexed column changed)
7.  Index kept in sync — UPDATE (non-indexed column changed)
8.  Index built from pre-existing data (CREATE INDEX on non-empty table)
9.  DROP INDEX — access falls back to full scan
10. Persistence — index survives close/reopen
11. Multiple indexes on the same table
12. Error cases
"""

import pytest

from src.parser    import parse, ParseError
from src.ast_nodes import CreateIndexStmt, DropIndexStmt
from src.catalog   import Catalog, CatalogError
from src.database  import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db(tmp_path):
    db = Database(str(tmp_path))
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
    for i, (name, age) in enumerate(
        [("alice",30),("bob",25),("carol",35),("dave",28),("eve",30)], 1
    ):
        db.execute(f"INSERT INTO users VALUES ({i}, '{name}', {age})")
    return db


# ===========================================================================
# 1. Parsing
# ===========================================================================

def test_parse_create_index():
    stmt = parse("CREATE INDEX idx_age ON users (age)")
    assert isinstance(stmt, CreateIndexStmt)
    assert stmt.name  == "idx_age"
    assert stmt.table == "users"
    assert stmt.col   == "age"

def test_parse_drop_index():
    stmt = parse("DROP INDEX idx_age")
    assert isinstance(stmt, DropIndexStmt)
    assert stmt.name == "idx_age"

def test_parse_create_index_bad_syntax():
    with pytest.raises(ParseError):
        parse("CREATE INDEX ON users (age)")   # missing index name


# ===========================================================================
# 2. Catalog index registration
# ===========================================================================

def test_catalog_create_index(tmp_path):
    cat  = Catalog(str(tmp_path))
    from src.ast_nodes import ColumnDef
    cat.create_table("users", [
        ColumnDef("id", "INT", True),
        ColumnDef("age", "INT", False),
    ])
    idx = cat.create_index("idx_age", "users", "age")
    assert idx.name    == "idx_age"
    assert idx.table   == "users"
    assert idx.col     == "age"
    assert idx.col_idx == 1

def test_catalog_index_persists(tmp_path):
    from src.ast_nodes import ColumnDef
    cat1 = Catalog(str(tmp_path))
    cat1.create_table("t", [ColumnDef("id", "INT", True), ColumnDef("x", "INT", False)])
    cat1.create_index("idx_x", "t", "x")

    cat2 = Catalog(str(tmp_path))
    idx  = cat2.get_index("idx_x")
    assert idx.col == "x"

def test_catalog_index_on_string_column_raises(tmp_path):
    from src.ast_nodes import ColumnDef
    cat = Catalog(str(tmp_path))
    cat.create_table("t", [ColumnDef("id", "INT", True), ColumnDef("name", "STRING", False)])
    with pytest.raises(CatalogError, match="INT"):
        cat.create_index("idx_name", "t", "name")

def test_catalog_duplicate_index_raises(tmp_path):
    from src.ast_nodes import ColumnDef
    cat = Catalog(str(tmp_path))
    cat.create_table("t", [ColumnDef("id", "INT", True), ColumnDef("x", "INT", False)])
    cat.create_index("idx_x", "t", "x")
    with pytest.raises(CatalogError):
        cat.create_index("idx_x", "t", "x")

def test_catalog_index_on_unknown_table_raises(tmp_path):
    cat = Catalog(str(tmp_path))
    with pytest.raises(CatalogError):
        cat.create_index("idx_x", "nosuch", "x")

def test_catalog_drop_index(tmp_path):
    from src.ast_nodes import ColumnDef
    cat = Catalog(str(tmp_path))
    cat.create_table("t", [ColumnDef("id", "INT", True), ColumnDef("x", "INT", False)])
    cat.create_index("idx_x", "t", "x")
    cat.drop_index("idx_x")
    assert not cat.index_exists("idx_x")

def test_catalog_drop_unknown_index_raises(tmp_path):
    cat = Catalog(str(tmp_path))
    with pytest.raises(CatalogError):
        cat.drop_index("nosuch")


# ===========================================================================
# 3. Index equality lookup
# ===========================================================================

def test_index_equality_returns_correct_rows(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        names = {r['name'] for r in rows}
        assert names == {"alice", "eve"}

def test_index_equality_no_match_returns_empty(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        rows = db.execute("SELECT * FROM users WHERE age = 99")
        assert rows == []

def test_index_equality_uses_index_not_full_scan(tmp_path, monkeypatch):
    """Verify engine.scan() is NOT called when an index covers the WHERE col."""
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        engine     = db._engines['users']
        scan_calls = []
        orig_scan  = engine.scan
        monkeypatch.setattr(engine, 'scan', lambda *a, **kw: (scan_calls.append(1), orig_scan(*a, **kw))[1])
        db.execute("SELECT * FROM users WHERE age = 30")
        assert not scan_calls, "Full scan should NOT be used when index is available"


# ===========================================================================
# 4. Index range lookup
# ===========================================================================

def test_index_range_returns_correct_rows(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        rows = db.execute("SELECT * FROM users WHERE age BETWEEN 28 AND 32")
        ages = sorted(r['age'] for r in rows)
        assert ages == [28, 30, 30]

def test_index_range_no_match(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        rows = db.execute("SELECT * FROM users WHERE age BETWEEN 50 AND 99")
        assert rows == []


# ===========================================================================
# 5. Index in sync — INSERT
# ===========================================================================

def test_insert_after_create_index_is_indexed(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("INSERT INTO users VALUES (10, 'frank', 30)")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        names = {r['name'] for r in rows}
        assert "frank" in names

def test_new_value_only_appears_in_its_bucket(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("INSERT INTO users VALUES (10, 'frank', 42)")
        assert db.execute("SELECT * FROM users WHERE age = 42") == [
            {'id': 10, 'name': 'frank', 'age': 42}
        ]
        # Should NOT appear in other age buckets
        names_30 = {r['name'] for r in db.execute("SELECT * FROM users WHERE age = 30")}
        assert "frank" not in names_30


# ===========================================================================
# 6. Index in sync — DELETE
# ===========================================================================

def test_delete_removes_from_index(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        # alice (id=1) and eve (id=5) both have age=30
        db.execute("DELETE FROM users WHERE id = 1")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        assert len(rows) == 1
        assert rows[0]['name'] == "eve"

def test_delete_last_row_empties_index_bucket(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("DELETE FROM users WHERE id = 2")   # bob, age=25 (unique)
        rows = db.execute("SELECT * FROM users WHERE age = 25")
        assert rows == []


# ===========================================================================
# 7. Index in sync — UPDATE (indexed column changes)
# ===========================================================================

def test_update_indexed_column_moves_entry(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("UPDATE users SET age = 40 WHERE id = 1")  # alice: 30 → 40

        # alice no longer in age=30 bucket
        names_30 = {r['name'] for r in db.execute("SELECT * FROM users WHERE age = 30")}
        assert "alice" not in names_30

        # alice now in age=40 bucket
        rows_40 = db.execute("SELECT * FROM users WHERE age = 40")
        assert len(rows_40) == 1
        assert rows_40[0]['name'] == "alice"

def test_update_non_indexed_column_leaves_index_intact(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("UPDATE users SET name = 'ALICE' WHERE id = 1")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        names = {r['name'] for r in rows}
        assert "ALICE" in names


# ===========================================================================
# 8. CREATE INDEX on non-empty table (back-fill)
# ===========================================================================

def test_create_index_backfills_existing_rows(tmp_path):
    with make_db(tmp_path) as db:
        # Table already has 5 rows; create index after the fact
        db.execute("CREATE INDEX idx_age ON users (age)")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        assert len(rows) == 2   # alice + eve


# ===========================================================================
# 9. DROP INDEX — fallback to full scan
# ===========================================================================

def test_drop_index_then_query_still_works(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("DROP INDEX idx_age")
        # Query should still work via full scan
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        names = {r['name'] for r in rows}
        assert names == {"alice", "eve"}

def test_drop_index_removes_index_file(tmp_path):
    import os
    with make_db(tmp_path) as db:
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("DROP INDEX idx_age")
    assert not os.path.exists(str(tmp_path / "idx_age.db"))

def test_drop_nonexistent_index_raises(tmp_path):
    with make_db(tmp_path) as db:
        with pytest.raises(CatalogError):
            db.execute("DROP INDEX nosuch")


# ===========================================================================
# 10. Persistence
# ===========================================================================

def test_index_persists_across_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   25)")
        db.execute("CREATE INDEX idx_age ON users (age)")

    with Database(path) as db:
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        assert rows == [{'id': 1, 'name': 'alice', 'age': 30}]

def test_inserts_after_reopen_are_indexed(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("CREATE INDEX idx_age ON users (age)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")

    with Database(path) as db:
        db.execute("INSERT INTO users VALUES (2, 'bob', 30)")
        rows = db.execute("SELECT * FROM users WHERE age = 30")
        assert len(rows) == 2


# ===========================================================================
# 11. Multiple indexes on the same table
# ===========================================================================

def test_two_indexes_on_same_table(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, a INT, b INT)")
        db.execute("INSERT INTO t VALUES (1, 10, 100)")
        db.execute("INSERT INTO t VALUES (2, 20, 100)")
        db.execute("INSERT INTO t VALUES (3, 10, 200)")
        db.execute("CREATE INDEX idx_a ON t (a)")
        db.execute("CREATE INDEX idx_b ON t (b)")

        rows_a = db.execute("SELECT * FROM t WHERE a = 10")
        assert sorted(r['id'] for r in rows_a) == [1, 3]

        rows_b = db.execute("SELECT * FROM t WHERE b = 100")
        assert sorted(r['id'] for r in rows_b) == [1, 2]

def test_both_indexes_updated_on_insert(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, a INT, b INT)")
        db.execute("CREATE INDEX idx_a ON t (a)")
        db.execute("CREATE INDEX idx_b ON t (b)")
        db.execute("INSERT INTO t VALUES (1, 10, 100)")
        db.execute("INSERT INTO t VALUES (2, 10, 200)")

        assert len(db.execute("SELECT * FROM t WHERE a = 10")) == 2
        assert len(db.execute("SELECT * FROM t WHERE b = 100")) == 1


# ===========================================================================
# 12. Error cases
# ===========================================================================

def test_create_index_on_string_column_raises(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        with pytest.raises(CatalogError, match="INT"):
            db.execute("CREATE INDEX idx_name ON users (name)")

def test_create_index_on_unknown_table_raises(tmp_path):
    with Database(str(tmp_path)) as db:
        with pytest.raises(CatalogError):
            db.execute("CREATE INDEX idx_x ON nosuch (x)")

def test_create_index_on_unknown_column_raises(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        with pytest.raises(CatalogError):
            db.execute("CREATE INDEX idx_y ON t (y)")
