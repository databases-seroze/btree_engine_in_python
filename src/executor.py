"""
Executor — evaluates SQL AST nodes against the storage engine.

Access path selection
---------------------
SELECT / UPDATE / DELETE all route through _fetch_rows(), which picks:

  PK equality   WHERE pk = v         → engine.get(v)          O(log n)
  PK range      WHERE pk BETWEEN l AND h → engine.scan(l, h)  O(log n + k)
  Full scan     anything else         → engine.scan() + filter O(n)

For UPDATE the matched rows are re-inserted with the mutated column values.
For DELETE the matched rows' PK values are passed to engine.delete().

Rows are represented as plain Python dicts: {column_name: value}.
"""

import os

from src.ast_nodes import (
    ColumnDef,
    EqExpr, CompareExpr, BetweenExpr, AndExpr, OrExpr,
    CreateTableStmt, InsertStmt, SelectStmt, UpdateStmt, DeleteStmt,
)
from src.catalog import Catalog, TableSchema, CatalogError
from src.engine  import Engine


class ExecutionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def _eval_expr(expr, row: dict) -> bool:
    """Return True if *row* satisfies *expr* (None expr → always True)."""
    if expr is None:
        return True
    if isinstance(expr, EqExpr):
        return row.get(expr.col) == expr.value
    if isinstance(expr, CompareExpr):
        val = row.get(expr.col)
        if val is None:
            return False
        op = expr.op
        if op == '<':  return val <  expr.value
        if op == '>':  return val >  expr.value
        if op == '<=': return val <= expr.value
        if op == '>=': return val >= expr.value
        if op == '!=': return val != expr.value
    if isinstance(expr, BetweenExpr):
        val = row.get(expr.col)
        return val is not None and expr.low <= val <= expr.high
    if isinstance(expr, AndExpr):
        return _eval_expr(expr.left, row) and _eval_expr(expr.right, row)
    if isinstance(expr, OrExpr):
        return _eval_expr(expr.left, row) or _eval_expr(expr.right, row)
    raise ExecutionError(f"Unknown expression type {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _to_dict(schema: TableSchema, row_list: list) -> dict:
    return {col.name: val for col, val in zip(schema.columns, row_list)}


def _project(row: dict, columns: list) -> dict:
    if columns == ['*']:
        return row
    missing = [c for c in columns if c not in row]
    if missing:
        raise ExecutionError(f"Unknown column(s): {missing}")
    return {c: row[c] for c in columns}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Stateless SQL executor.

    Parameters
    ----------
    catalog : Catalog
    engines : dict          Mutable dict (table_name → Engine); modified in-place
                            when CREATE TABLE opens a new engine.
    dirpath : str           Directory for new Engine files.
    """

    def __init__(self, catalog: Catalog, engines: dict, dirpath: str):
        self._catalog = catalog
        self._engines = engines
        self._dirpath = dirpath

    def execute(self, stmt) -> list:
        if isinstance(stmt, CreateTableStmt):
            return self._exec_create(stmt)
        if isinstance(stmt, InsertStmt):
            return self._exec_insert(stmt)
        if isinstance(stmt, SelectStmt):
            return self._exec_select(stmt)
        if isinstance(stmt, UpdateStmt):
            return self._exec_update(stmt)
        if isinstance(stmt, DeleteStmt):
            return self._exec_delete(stmt)
        raise ExecutionError(f"Unknown statement type {type(stmt).__name__}")

    # ------------------------------------------------------------------
    # CREATE TABLE
    # ------------------------------------------------------------------

    def _exec_create(self, stmt: CreateTableStmt) -> list:
        self._catalog.create_table(stmt.table, stmt.columns)
        path   = os.path.join(self._dirpath, f"{stmt.table}.db")
        engine = Engine(path)
        self._engines[stmt.table] = engine
        return []

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------

    def _exec_insert(self, stmt: InsertStmt) -> list:
        schema = self._catalog.get_table(stmt.table)
        if len(stmt.values) != len(schema.columns):
            raise ExecutionError(
                f"INSERT into '{stmt.table}': expected {len(schema.columns)} "
                f"values, got {len(stmt.values)}"
            )
        pk_val = stmt.values[schema.pk_idx]
        if not isinstance(pk_val, int):
            raise ExecutionError(
                f"PRIMARY KEY value must be INT, got {type(pk_val).__name__}"
            )
        engine = self._get_engine(stmt.table)
        engine.put(pk_val, stmt.values)
        return []

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------

    def _exec_select(self, stmt: SelectStmt) -> list:
        schema = self._catalog.get_table(stmt.table)
        engine = self._get_engine(stmt.table)
        rows   = self._fetch_rows(schema, engine, stmt.where)
        return [_project(r, stmt.columns) for r in rows]

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def _exec_update(self, stmt: UpdateStmt) -> list:
        schema    = self._catalog.get_table(stmt.table)
        engine    = self._get_engine(stmt.table)
        col_names = [c.name for c in schema.columns]

        for col in stmt.assignments:
            if col not in col_names:
                raise ExecutionError(
                    f"Column '{col}' does not exist in table '{stmt.table}'"
                )
            if col == schema.pk_col:
                raise ExecutionError("Cannot UPDATE the primary key column")

        rows = self._fetch_rows(schema, engine, stmt.where)
        for row_dict in rows:
            updated = [row_dict[c] for c in col_names]
            for col, val in stmt.assignments.items():
                updated[col_names.index(col)] = val
            engine.put(row_dict[schema.pk_col], updated)

        return []

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def _exec_delete(self, stmt: DeleteStmt) -> list:
        schema = self._catalog.get_table(stmt.table)
        engine = self._get_engine(stmt.table)
        rows   = self._fetch_rows(schema, engine, stmt.where)
        for row_dict in rows:
            engine.delete(row_dict[schema.pk_col])
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_rows(self, schema: TableSchema, engine: Engine, where) -> list:
        """
        Return a list of row dicts matching *where*, using the cheapest path.
        """
        # PK equality → point lookup
        if isinstance(where, EqExpr) and where.col == schema.pk_col:
            raw = engine.get(where.value)
            if raw is None:
                return []
            return [_to_dict(schema, raw)]

        # PK BETWEEN → index range scan
        if isinstance(where, BetweenExpr) and where.col == schema.pk_col:
            return [
                _to_dict(schema, row)
                for _, row in engine.scan(where.low, where.high)
            ]

        # Full scan + filter
        return [
            _to_dict(schema, row)
            for _, row in engine.scan()
            if _eval_expr(where, _to_dict(schema, row))
        ]

    def _get_engine(self, table: str) -> Engine:
        if table not in self._engines:
            raise ExecutionError(
                f"Engine for table '{table}' is not open — was CREATE TABLE run?"
            )
        return self._engines[table]
