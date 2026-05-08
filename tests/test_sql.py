"""
Tests for the SQL layer: Lexer, Parser, Catalog, Executor, Database.

Sections
--------
1.  Lexer — token types, literals, keywords, edge cases
2.  Parser — AST shape for all statement types
3.  Catalog — create, get, persist, error cases
4.  Executor / Database — end-to-end SQL execution
5.  Access path selection — point lookup vs range vs full scan
6.  Persistence — schema and data survive close/reopen
7.  Error handling — bad SQL, wrong table, wrong column
"""

import pytest

from src.lexer     import Lexer, TT, LexError
from src.parser    import Parser, ParseError, parse
from src.ast_nodes import (
    ColumnDef,
    EqExpr, CompareExpr, BetweenExpr, AndExpr,
    CreateTableStmt, InsertStmt, SelectStmt, UpdateStmt, DeleteStmt,
)
from src.catalog   import Catalog, CatalogError
from src.database  import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(sql):
    return Lexer(sql).tokenize()

def token_types(sql):
    return [t.type for t in tokenize(sql) if t.type != TT.EOF]

def make_db(tmp_path):
    db = Database(str(tmp_path))
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
    return db


# ===========================================================================
# 1. Lexer
# ===========================================================================

def test_lex_keywords_case_insensitive():
    toks = tokenize("select FROM Where")
    assert toks[0].type == TT.SELECT
    assert toks[1].type == TT.FROM
    assert toks[2].type == TT.WHERE

def test_lex_integer_literal():
    toks = tokenize("42")
    assert toks[0].type == TT.INT_LIT
    assert toks[0].value == 42

def test_lex_negative_integer():
    toks = tokenize("-7")
    assert toks[0].type == TT.INT_LIT
    assert toks[0].value == -7

def test_lex_string_literal_double_quotes():
    toks = tokenize('"hello world"')
    assert toks[0].type == TT.STR_LIT
    assert toks[0].value == "hello world"

def test_lex_string_literal_single_quotes():
    toks = tokenize("'alice'")
    assert toks[0].type == TT.STR_LIT
    assert toks[0].value == "alice"

def test_lex_string_backslash_escape():
    toks = tokenize(r"'it\'s'")
    assert toks[0].value == "it's"

def test_lex_null():
    toks = tokenize("NULL")
    assert toks[0].type == TT.NULL

def test_lex_star():
    assert token_types("*") == [TT.STAR]

def test_lex_comparison_operators():
    assert token_types("= != < > <= >=") == [
        TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE
    ]

def test_lex_identifier():
    toks = tokenize("my_table")
    assert toks[0].type == TT.IDENT
    assert toks[0].value == "my_table"

def test_lex_ident_preserves_case():
    toks = tokenize("myTable")
    assert toks[0].value == "myTable"

def test_lex_unterminated_string_raises():
    with pytest.raises(LexError):
        tokenize('"unclosed')

def test_lex_select_star_from():
    types = token_types("SELECT * FROM users")
    assert types == [TT.SELECT, TT.STAR, TT.FROM, TT.IDENT]

def test_lex_full_insert():
    types = token_types("INSERT INTO t VALUES (1, 'x', 30)")
    assert types == [
        TT.INSERT, TT.INTO, TT.IDENT,
        TT.VALUES,
        TT.LPAREN, TT.INT_LIT, TT.COMMA, TT.STR_LIT, TT.COMMA, TT.INT_LIT, TT.RPAREN,
    ]


# ===========================================================================
# 2. Parser
# ===========================================================================

def test_parse_create_table():
    stmt = parse("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
    assert isinstance(stmt, CreateTableStmt)
    assert stmt.table == "users"
    assert len(stmt.columns) == 3
    assert stmt.columns[0] == ColumnDef("id", "INT", True)
    assert stmt.columns[1] == ColumnDef("name", "STRING", False)
    assert stmt.columns[2] == ColumnDef("age", "INT", False)

def test_parse_create_table_requires_pk():
    with pytest.raises(ParseError):
        parse("CREATE TABLE t (id INT, name STRING)")

def test_parse_create_table_pk_must_be_int():
    with pytest.raises(ParseError):
        parse("CREATE TABLE t (id STRING PRIMARY KEY)")

def test_parse_insert():
    stmt = parse("INSERT INTO users VALUES (1, 'alice', 30)")
    assert isinstance(stmt, InsertStmt)
    assert stmt.table  == "users"
    assert stmt.values == [1, "alice", 30]

def test_parse_insert_null():
    stmt = parse("INSERT INTO t VALUES (1, NULL)")
    assert stmt.values == [1, None]

def test_parse_select_star():
    stmt = parse("SELECT * FROM users")
    assert isinstance(stmt, SelectStmt)
    assert stmt.columns == ['*']
    assert stmt.where   is None

def test_parse_select_columns():
    stmt = parse("SELECT id, name FROM users")
    assert stmt.columns == ['id', 'name']

def test_parse_select_where_eq():
    stmt = parse("SELECT * FROM users WHERE id = 1")
    assert isinstance(stmt.where, EqExpr)
    assert stmt.where.col   == 'id'
    assert stmt.where.value == 1

def test_parse_select_where_compare():
    stmt = parse("SELECT * FROM users WHERE age > 25")
    assert isinstance(stmt.where, CompareExpr)
    assert stmt.where.col   == 'age'
    assert stmt.where.op    == '>'
    assert stmt.where.value == 25

def test_parse_select_where_between():
    stmt = parse("SELECT * FROM users WHERE id BETWEEN 1 AND 100")
    assert isinstance(stmt.where, BetweenExpr)
    assert stmt.where.col  == 'id'
    assert stmt.where.low  == 1
    assert stmt.where.high == 100

def test_parse_select_where_and():
    stmt = parse("SELECT * FROM users WHERE age > 20 AND age < 40")
    assert isinstance(stmt.where, AndExpr)
    assert isinstance(stmt.where.left,  CompareExpr)
    assert isinstance(stmt.where.right, CompareExpr)

def test_parse_update():
    stmt = parse("UPDATE users SET age = 31 WHERE id = 1")
    assert isinstance(stmt, UpdateStmt)
    assert stmt.table == "users"
    assert stmt.assignments == {'age': 31}
    assert isinstance(stmt.where, EqExpr)

def test_parse_update_multiple_cols():
    stmt = parse("UPDATE users SET age = 31, name = 'carol'")
    assert stmt.assignments == {'age': 31, 'name': 'carol'}
    assert stmt.where is None

def test_parse_delete():
    stmt = parse("DELETE FROM users WHERE id = 5")
    assert isinstance(stmt, DeleteStmt)
    assert stmt.table == "users"
    assert isinstance(stmt.where, EqExpr)

def test_parse_delete_no_where():
    stmt = parse("DELETE FROM users")
    assert stmt.where is None

def test_parse_unexpected_token_raises():
    with pytest.raises(ParseError):
        parse("BLAH users")


# ===========================================================================
# 3. Catalog
# ===========================================================================

def test_catalog_create_and_get(tmp_path):
    cat = Catalog(str(tmp_path))
    cols = [
        ColumnDef("id", "INT", True),
        ColumnDef("name", "STRING", False),
    ]
    schema = cat.create_table("users", cols)
    assert schema.pk_col == "id"
    assert schema.pk_idx == 0

    got = cat.get_table("users")
    assert got.name == "users"
    assert len(got.columns) == 2

def test_catalog_duplicate_table_raises(tmp_path):
    cat  = Catalog(str(tmp_path))
    cols = [ColumnDef("id", "INT", True)]
    cat.create_table("t", cols)
    with pytest.raises(CatalogError):
        cat.create_table("t", cols)

def test_catalog_unknown_table_raises(tmp_path):
    cat = Catalog(str(tmp_path))
    with pytest.raises(CatalogError):
        cat.get_table("nonexistent")

def test_catalog_persists_across_reload(tmp_path):
    cat1 = Catalog(str(tmp_path))
    cols = [ColumnDef("id", "INT", True), ColumnDef("x", "STRING", False)]
    cat1.create_table("t", cols)

    cat2 = Catalog(str(tmp_path))
    schema = cat2.get_table("t")
    assert schema.pk_col == "id"
    assert len(schema.columns) == 2

def test_catalog_table_exists(tmp_path):
    cat  = Catalog(str(tmp_path))
    cols = [ColumnDef("id", "INT", True)]
    assert not cat.table_exists("t")
    cat.create_table("t", cols)
    assert cat.table_exists("t")


# ===========================================================================
# 4. Executor / Database — end-to-end
# ===========================================================================

def test_insert_and_select_star(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        rows = db.execute("SELECT * FROM users WHERE id = 1")
        assert rows == [{'id': 1, 'name': 'alice', 'age': 30}]

def test_insert_multiple_and_select_all(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   25)")
        db.execute("INSERT INTO users VALUES (3, 'carol', 35)")
        rows = db.execute("SELECT * FROM users")
        assert len(rows) == 3

def test_select_specific_columns(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        rows = db.execute("SELECT name, age FROM users WHERE id = 1")
        assert rows == [{'name': 'alice', 'age': 30}]
        assert 'id' not in rows[0]

def test_select_where_gt(tmp_path):
    with make_db(tmp_path) as db:
        for i, (name, age) in enumerate([("alice", 30), ("bob", 25), ("carol", 35)], 1):
            db.execute(f"INSERT INTO users VALUES ({i}, '{name}', {age})")
        rows = db.execute("SELECT * FROM users WHERE age > 28")
        ages = {r['age'] for r in rows}
        assert ages == {30, 35}

def test_select_where_between_pk(tmp_path):
    with make_db(tmp_path) as db:
        for i in range(1, 11):
            db.execute(f"INSERT INTO users VALUES ({i}, 'user{i}', {20+i})")
        rows = db.execute("SELECT * FROM users WHERE id BETWEEN 3 AND 7")
        assert [r['id'] for r in rows] == [3, 4, 5, 6, 7]

def test_select_no_match_returns_empty(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        rows = db.execute("SELECT * FROM users WHERE id = 99")
        assert rows == []

def test_update_single_row(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("UPDATE users SET age = 31 WHERE id = 1")
        rows = db.execute("SELECT * FROM users WHERE id = 1")
        assert rows[0]['age'] == 31

def test_update_multiple_columns(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("UPDATE users SET name = 'alicia', age = 32 WHERE id = 1")
        row = db.execute("SELECT * FROM users WHERE id = 1")[0]
        assert row['name'] == 'alicia'
        assert row['age']  == 32

def test_update_no_where_updates_all(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   25)")
        db.execute("UPDATE users SET age = 99")
        rows = db.execute("SELECT * FROM users")
        assert all(r['age'] == 99 for r in rows)

def test_delete_single_row(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   25)")
        db.execute("DELETE FROM users WHERE id = 1")
        rows = db.execute("SELECT * FROM users")
        assert len(rows) == 1
        assert rows[0]['id'] == 2

def test_delete_no_where_clears_table(tmp_path):
    with make_db(tmp_path) as db:
        for i in range(1, 6):
            db.execute(f"INSERT INTO users VALUES ({i}, 'u{i}', {20+i})")
        db.execute("DELETE FROM users")
        rows = db.execute("SELECT * FROM users")
        assert rows == []

def test_delete_by_non_pk_column(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   30)")
        db.execute("INSERT INTO users VALUES (3, 'carol', 25)")
        db.execute("DELETE FROM users WHERE age = 30")
        rows = db.execute("SELECT * FROM users")
        assert len(rows) == 1
        assert rows[0]['name'] == 'carol'

def test_select_and_compound_where(tmp_path):
    with make_db(tmp_path) as db:
        for i, (name, age) in enumerate([("alice",30),("bob",25),("carol",35),("dave",28)], 1):
            db.execute(f"INSERT INTO users VALUES ({i}, '{name}', {age})")
        rows = db.execute("SELECT * FROM users WHERE age > 25 AND age < 35")
        ages = sorted(r['age'] for r in rows)
        assert ages == [28, 30]

def test_null_value_insert_and_select(tmp_path):
    with Database(str(tmp_path)) as db:
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, note STRING)")
        db.execute("INSERT INTO t VALUES (1, NULL)")
        rows = db.execute("SELECT * FROM t WHERE id = 1")
        assert rows[0]['note'] is None

def test_string_values_round_trip(tmp_path):
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'hello world', 0)")
        rows = db.execute("SELECT name FROM users WHERE id = 1")
        assert rows[0]['name'] == 'hello world'


# ===========================================================================
# 5. Access path selection
# ===========================================================================

def test_pk_equality_uses_point_lookup(tmp_path, monkeypatch):
    """Verify engine.get() is called for PK equality (not scan)."""
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        engine = db._engines['users']
        calls  = []
        orig   = engine.get
        monkeypatch.setattr(engine, 'get', lambda k: (calls.append(k), orig(k))[1])
        rows = db.execute("SELECT * FROM users WHERE id = 1")
        assert 1 in calls, "engine.get() should have been called for PK equality"
        assert rows[0]['name'] == 'alice'

def test_pk_between_uses_scan(tmp_path, monkeypatch):
    """Verify engine.scan() is called for PK BETWEEN."""
    with make_db(tmp_path) as db:
        for i in range(1, 6):
            db.execute(f"INSERT INTO users VALUES ({i}, 'u{i}', {20+i})")
        engine = db._engines['users']
        scan_calls = []
        orig_scan  = engine.scan
        def mock_scan(*args, **kwargs):
            scan_calls.append(args)
            return orig_scan(*args, **kwargs)
        monkeypatch.setattr(engine, 'scan', mock_scan)
        rows = db.execute("SELECT * FROM users WHERE id BETWEEN 2 AND 4")
        assert scan_calls, "engine.scan() should have been called"
        assert [r['id'] for r in rows] == [2, 3, 4]

def test_non_pk_filter_uses_full_scan(tmp_path):
    """Full scan + filter for non-PK WHERE clause."""
    with make_db(tmp_path) as db:
        for i, age in enumerate([30, 25, 35, 22, 28], 1):
            db.execute(f"INSERT INTO users VALUES ({i}, 'u{i}', {age})")
        rows = db.execute("SELECT * FROM users WHERE age >= 28")
        assert sorted(r['age'] for r in rows) == [28, 30, 35]


# ===========================================================================
# 6. Persistence
# ===========================================================================

def test_schema_persists_across_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")

    with Database(path) as db:
        rows = db.execute("SELECT * FROM users WHERE id = 1")
        assert rows == [{'id': 1, 'name': 'alice', 'age': 30}]

def test_data_persists_across_reopen(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        for i in range(1, 6):
            db.execute(f"INSERT INTO users VALUES ({i}, 'user{i}', {20+i})")

    with Database(path) as db:
        rows = db.execute("SELECT * FROM users")
        assert len(rows) == 5

def test_multiple_tables_persist(tmp_path):
    path = str(tmp_path)
    with Database(path) as db:
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, label STRING)")
        db.execute("INSERT INTO a VALUES (1, 100)")
        db.execute("INSERT INTO b VALUES (1, 'hello')")

    with Database(path) as db:
        assert db.execute("SELECT * FROM a WHERE id = 1") == [{'id': 1, 'val': 100}]
        assert db.execute("SELECT * FROM b WHERE id = 1") == [{'id': 1, 'label': 'hello'}]


# ===========================================================================
# 7. Error handling
# ===========================================================================

def test_insert_unknown_table_raises(tmp_path):
    from src.catalog import CatalogError
    with Database(str(tmp_path)) as db:
        with pytest.raises(CatalogError):
            db.execute("INSERT INTO nosuch VALUES (1, 'x')")

def test_insert_wrong_number_of_values_raises(tmp_path):
    from src.executor import ExecutionError
    with make_db(tmp_path) as db:
        with pytest.raises(ExecutionError):
            db.execute("INSERT INTO users VALUES (1, 'alice')")  # missing age

def test_update_unknown_column_raises(tmp_path):
    from src.executor import ExecutionError
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        with pytest.raises(ExecutionError):
            db.execute("UPDATE users SET nosuchcol = 1 WHERE id = 1")

def test_update_pk_raises(tmp_path):
    from src.executor import ExecutionError
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        with pytest.raises(ExecutionError):
            db.execute("UPDATE users SET id = 99 WHERE id = 1")

def test_create_duplicate_table_raises(tmp_path):
    with make_db(tmp_path) as db:
        with pytest.raises(CatalogError):
            db.execute("CREATE TABLE users (id INT PRIMARY KEY, x STRING)")

def test_select_unknown_column_raises(tmp_path):
    from src.executor import ExecutionError
    with make_db(tmp_path) as db:
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        with pytest.raises(ExecutionError):
            db.execute("SELECT nosuchcol FROM users WHERE id = 1")
