"""
Executor — evaluates SQL AST nodes against the storage engine.

Access path selection
---------------------
SELECT / UPDATE / DELETE all route through _fetch_rows(), which picks:

  PK equality        WHERE pk = v            → engine.get(v)         O(log n)
  PK range           WHERE pk BETWEEN l AND h → engine.scan(l, h)    O(log n + k)
  Index equality     WHERE col = v  (indexed) → index lookup + fetch  O(log n + k)
  Index range        WHERE col BETWEEN l AND h (indexed)              O(log n + k)
  Full scan          anything else            → engine.scan() + filter O(n)

Rows are represented as plain Python dicts: {column_name: value}.
"""

import os

from src.ast_nodes import (
    ColumnDef, ColRef,
    EqExpr, CompareExpr, BetweenExpr, AndExpr, OrExpr,
    CreateTableStmt, InsertStmt, SelectStmt, UpdateStmt, DeleteStmt,
    CreateIndexStmt, DropIndexStmt, JoinSelectStmt,
)
from src.catalog       import Catalog, TableSchema, CatalogError
from src.engine        import Engine
from src.index_manager import IndexManager


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


def _eval_join_expr(expr, row: dict) -> bool:
    """Like _eval_expr but col may be a ColRef (table.col) or plain str."""
    if expr is None:
        return True
    if isinstance(expr, EqExpr):
        key = f"{expr.col.table}.{expr.col.col}" if isinstance(expr.col, ColRef) else expr.col
        return row.get(key) == expr.value
    if isinstance(expr, CompareExpr):
        key = f"{expr.col.table}.{expr.col.col}" if isinstance(expr.col, ColRef) else expr.col
        val = row.get(key)
        if val is None:
            return False
        op = expr.op
        if op == '<':  return val <  expr.value
        if op == '>':  return val >  expr.value
        if op == '<=': return val <= expr.value
        if op == '>=': return val >= expr.value
        if op == '!=': return val != expr.value
    if isinstance(expr, BetweenExpr):
        key = f"{expr.col.table}.{expr.col.col}" if isinstance(expr.col, ColRef) else expr.col
        val = row.get(key)
        return val is not None and expr.low <= val <= expr.high
    if isinstance(expr, AndExpr):
        return _eval_join_expr(expr.left, row) and _eval_join_expr(expr.right, row)
    if isinstance(expr, OrExpr):
        return _eval_join_expr(expr.left, row) or _eval_join_expr(expr.right, row)
    raise ExecutionError(f"Unknown expression type {type(expr).__name__}")


def _project_join(row: dict, columns: list, left_table: str, right_table: str) -> dict:
    """Project a merged join row.  '*' returns all qualified columns."""
    if columns == ['*']:
        # Return only the qualified names to avoid duplicates
        return {k: v for k, v in row.items() if '.' in k}
    result = {}
    for col in columns:
        if isinstance(col, ColRef):
            qualified = f"{col.table}.{col.col}"
            if qualified not in row:
                raise ExecutionError(f"Unknown column: {col.table}.{col.col}")
            result[qualified] = row[qualified]
        else:
            if col not in row:
                raise ExecutionError(f"Unknown column: {col}")
            result[col] = row[col]
    return result


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    SQL executor.

    Parameters
    ----------
    catalog   : Catalog
    engines   : dict            table_name → Engine; modified when CREATE TABLE runs.
    dirpath   : str             Directory for new Engine files.
    index_mgr : IndexManager    Optional; enables secondary index support.
    """

    def __init__(self, catalog: Catalog, engines: dict, dirpath: str,
                 index_mgr: IndexManager | None = None):
        self._catalog   = catalog
        self._engines   = engines
        self._dirpath   = dirpath
        self._index_mgr = index_mgr

    def execute(self, stmt) -> list:
        if isinstance(stmt, CreateTableStmt):
            return self._exec_create_table(stmt)
        if isinstance(stmt, CreateIndexStmt):
            return self._exec_create_index(stmt)
        if isinstance(stmt, DropIndexStmt):
            return self._exec_drop_index(stmt)
        if isinstance(stmt, InsertStmt):
            return self._exec_insert(stmt)
        if isinstance(stmt, JoinSelectStmt):
            return self._exec_join_select(stmt)
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

    def _exec_create_table(self, stmt: CreateTableStmt) -> list:
        self._catalog.create_table(stmt.table, stmt.columns)
        path   = os.path.join(self._dirpath, f"{stmt.table}.db")
        engine = Engine(path)
        self._engines[stmt.table] = engine
        return []

    # ------------------------------------------------------------------
    # CREATE INDEX / DROP INDEX
    # ------------------------------------------------------------------

    def _exec_create_index(self, stmt: CreateIndexStmt) -> list:
        if self._index_mgr is None:
            raise ExecutionError("IndexManager not available")
        idx_schema   = self._catalog.create_index(stmt.name, stmt.table, stmt.col)
        table_schema = self._catalog.get_table(stmt.table)
        table_engine = self._get_engine(stmt.table)
        self._index_mgr.create_index(idx_schema, table_schema, table_engine)
        return []

    def _exec_drop_index(self, stmt: DropIndexStmt) -> list:
        if self._index_mgr is None:
            raise ExecutionError("IndexManager not available")
        self._catalog.drop_index(stmt.name)
        self._index_mgr.drop_index(stmt.name)
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
        # FK check: every REFERENCES column must point to an existing parent row
        for fk in self._catalog.get_fkeys_from(stmt.table):
            fk_val = stmt.values[fk.child_col_idx]
            if fk_val is not None:
                parent_engine = self._get_engine(fk.parent_table)
                if parent_engine.get(fk_val) is None:
                    raise ExecutionError(
                        f"Foreign key violation: '{stmt.table}.{fk.child_col}' = {fk_val} "
                        f"does not exist in '{fk.parent_table}.{fk.parent_col}'"
                    )
        engine = self._get_engine(stmt.table)
        engine.put(pk_val, stmt.values)
        if self._index_mgr:
            self._index_mgr.on_insert(schema, stmt.values)
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
            old_row = [row_dict[c] for c in col_names]
            new_row = old_row[:]
            for col, val in stmt.assignments.items():
                new_row[col_names.index(col)] = val
            engine.put(row_dict[schema.pk_col], new_row)
            if self._index_mgr:
                self._index_mgr.on_update(schema, old_row, new_row)

        return []

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def _exec_delete(self, stmt: DeleteStmt) -> list:
        schema    = self._catalog.get_table(stmt.table)
        engine    = self._get_engine(stmt.table)
        col_names = [c.name for c in schema.columns]
        rows      = self._fetch_rows(schema, engine, stmt.where)
        for row_dict in rows:
            pk = row_dict[schema.pk_col]
            # FK check: no child row may reference this PK
            for fk in self._catalog.get_fkeys_to(stmt.table):
                child_engine = self._get_engine(fk.child_table)
                child_schema = self._catalog.get_table(fk.child_table)
                # Use secondary index on FK column if available, else full scan
                child_rows = self._fetch_rows(
                    child_schema, child_engine,
                    EqExpr(col=fk.child_col, value=pk)
                )
                if child_rows:
                    raise ExecutionError(
                        f"Foreign key violation: cannot delete '{stmt.table}' row "
                        f"with {schema.pk_col}={pk} — "
                        f"referenced by '{fk.child_table}.{fk.child_col}'"
                    )
            engine.delete(pk)
            if self._index_mgr:
                old_row = [row_dict[c] for c in col_names]
                self._index_mgr.on_delete(schema, old_row)
        return []

    # ------------------------------------------------------------------
    # JOIN SELECT
    # ------------------------------------------------------------------

    def _exec_join_select(self, stmt: JoinSelectStmt) -> list:
        left_schema  = self._catalog.get_table(stmt.left_table)
        right_schema = self._catalog.get_table(stmt.right_table)
        left_engine  = self._get_engine(stmt.left_table)
        right_engine = self._get_engine(stmt.right_table)

        # Fetch all left rows (full scan of outer table)
        left_rows = [
            _to_dict(left_schema, row)
            for _, row in left_engine.scan()
        ]

        # For each left row, probe the right table using the join key.
        # Use secondary index on right join col if available (index NLJ),
        # otherwise fall back to equality fetch (if it's the PK) or full scan.
        result = []
        for lrow in left_rows:
            join_val = lrow.get(stmt.left_col)
            if join_val is None:
                continue

            # Access path for right side
            if stmt.right_col == right_schema.pk_col:
                raw = right_engine.get(join_val)
                right_matches = [_to_dict(right_schema, raw)] if raw else []
            elif self._index_mgr:
                idx = self._index_mgr.has_index_on(stmt.right_table, stmt.right_col)
                if idx:
                    pks = self._index_mgr.lookup(idx.name, join_val)
                    right_matches = [
                        _to_dict(right_schema, row)
                        for pk in pks
                        if (row := right_engine.get(pk)) is not None
                    ]
                else:
                    right_matches = self._fetch_rows(
                        right_schema, right_engine,
                        EqExpr(col=stmt.right_col, value=join_val)
                    )
            else:
                right_matches = self._fetch_rows(
                    right_schema, right_engine,
                    EqExpr(col=stmt.right_col, value=join_val)
                )

            for rrow in right_matches:
                # Merge rows: prefix ambiguous columns with table name
                merged = {}
                for k, v in lrow.items():
                    merged[f"{stmt.left_table}.{k}"] = v
                    merged[k] = v   # unqualified alias (left wins on conflict)
                for k, v in rrow.items():
                    merged[f"{stmt.right_table}.{k}"] = v
                    if k not in merged:
                        merged[k] = v

                # Apply post-join WHERE filter
                if stmt.where is not None and not _eval_join_expr(stmt.where, merged):
                    continue

                result.append(_project_join(merged, stmt.columns,
                                            stmt.left_table, stmt.right_table))

        return result

    # ------------------------------------------------------------------
    # Access path selection
    # ------------------------------------------------------------------

    def _fetch_rows(self, schema: TableSchema, engine: Engine, where) -> list:
        """Return row dicts matching *where* using the cheapest available path."""

        # 1. PK equality → O(log n) point lookup
        if isinstance(where, EqExpr) and where.col == schema.pk_col:
            raw = engine.get(where.value)
            if raw is None:
                return []
            return [_to_dict(schema, raw)]

        # 2. PK BETWEEN → O(log n + k) range scan
        if isinstance(where, BetweenExpr) and where.col == schema.pk_col:
            return [
                _to_dict(schema, row)
                for _, row in engine.scan(where.low, where.high)
            ]

        # 3. Secondary index equality → O(log n + k)
        if isinstance(where, EqExpr) and self._index_mgr:
            idx = self._index_mgr.has_index_on(schema.name, where.col)
            if idx:
                pks = self._index_mgr.lookup(idx.name, where.value)
                return [
                    _to_dict(schema, row)
                    for pk in pks
                    if (row := engine.get(pk)) is not None
                ]

        # 4. Secondary index range → O(log n + k)
        if isinstance(where, BetweenExpr) and self._index_mgr:
            idx = self._index_mgr.has_index_on(schema.name, where.col)
            if idx:
                pks = self._index_mgr.range_lookup(idx.name, where.low, where.high)
                rows = [
                    _to_dict(schema, row)
                    for pk in pks
                    if (row := engine.get(pk)) is not None
                ]
                return sorted(rows, key=lambda r: r[schema.pk_col])

        # 5. Full scan + filter — O(n)
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
