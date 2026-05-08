"""
IndexManager — maintains secondary index B+Tree files.

Each secondary index is stored in its own Engine file:
    <dirpath>/<index_name>.db

Index entry format
------------------
key   = indexed column value (INT, uint32)
value = encoded record containing the list of primary keys that share this value

    age=30 → [pk=1, pk=5, pk=9]   stored as encode_record([1, 5, 9])

This supports non-unique indexes naturally: many rows may have the same
value for an indexed column.

Sync protocol
-------------
on_insert(table_schema, row_list)
    Called after engine.put().  Adds each non-NULL indexed column value
    to its index, appending the row's PK to the existing PK list.

on_delete(table_schema, row_list)
    Called before engine.delete().  Removes the row's PK from each index.
    If the PK list becomes empty, the index entry is deleted entirely.

on_update(table_schema, old_row_list, new_row_list)
    Called when engine.put() replaces an existing row.  For each indexed
    column where the value changed, removes the old entry and adds the new one.

Limitation
----------
Only INT columns may be indexed (B+Tree keys are uint32).  This is enforced
at CREATE INDEX time by the Catalog.  NULL values in an indexed column are
silently skipped — they are not indexed.
"""

import os

from src.catalog import Catalog, IndexSchema


class IndexManager:
    """
    Opens and manages Engine instances for all secondary indexes.

    Parameters
    ----------
    catalog : Catalog
    dirpath : str      Directory containing index Engine files.
    """

    def __init__(self, catalog: Catalog, dirpath: str):
        self._catalog = catalog
        self._dirpath = dirpath
        self._engines = {}   # index_name → Engine

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_all(self) -> None:
        """Open Engine files for every index currently in the catalog."""
        for idx in self._catalog.all_indexes():
            self._open_engine(idx.name)

    def create_index(self, idx: IndexSchema, table_schema, table_engine) -> None:
        """
        Create a new index Engine and populate it from the existing table data.

        Parameters
        ----------
        idx          : IndexSchema  — already registered in catalog
        table_schema : TableSchema
        table_engine : Engine       — the table's Engine (for scanning existing rows)
        """
        engine = self._open_engine(idx.name)
        for _, row_list in table_engine.scan():
            col_val = row_list[idx.col_idx]
            pk_val  = row_list[table_schema.pk_idx]
            if col_val is not None:
                self._add(idx.name, col_val, pk_val)

    def drop_index(self, name: str) -> None:
        """Close and delete the index Engine file."""
        engine = self._engines.pop(name, None)
        if engine is not None:
            engine.close()
        for suffix in ('', '.wal'):
            path = os.path.join(self._dirpath, f"{name}.db{suffix}")
            if os.path.exists(path):
                os.remove(path)

    def close(self) -> None:
        """Flush and close all index Engine files."""
        for engine in self._engines.values():
            engine.close()
        self._engines.clear()

    # ------------------------------------------------------------------
    # Sync operations (called by Executor on every write)
    # ------------------------------------------------------------------

    def on_insert(self, table_schema, row_list: list) -> None:
        pk_val = row_list[table_schema.pk_idx]
        for idx in self._catalog.get_indexes_for_table(table_schema.name):
            col_val = row_list[idx.col_idx]
            if col_val is not None:
                self._add(idx.name, col_val, pk_val)

    def on_delete(self, table_schema, row_list: list) -> None:
        pk_val = row_list[table_schema.pk_idx]
        for idx in self._catalog.get_indexes_for_table(table_schema.name):
            col_val = row_list[idx.col_idx]
            if col_val is not None:
                self._remove(idx.name, col_val, pk_val)

    def on_update(self, table_schema, old_row: list, new_row: list) -> None:
        pk_val = old_row[table_schema.pk_idx]   # PK cannot change
        for idx in self._catalog.get_indexes_for_table(table_schema.name):
            old_val = old_row[idx.col_idx]
            new_val = new_row[idx.col_idx]
            if old_val != new_val:
                if old_val is not None:
                    self._remove(idx.name, old_val, pk_val)
                if new_val is not None:
                    self._add(idx.name, new_val, pk_val)

    # ------------------------------------------------------------------
    # Lookup (called by Executor._fetch_rows)
    # ------------------------------------------------------------------

    def has_index_on(self, table: str, col: str) -> 'IndexSchema | None':
        """Return the IndexSchema if *table* has an index on *col*, else None."""
        for idx in self._catalog.get_indexes_for_table(table):
            if idx.col == col:
                return idx
        return None

    def lookup(self, index_name: str, value: int) -> list:
        """Return list of PKs whose indexed column equals *value*."""
        engine = self._engines.get(index_name)
        if engine is None:
            return []
        pks = engine.get(value)
        return pks if pks is not None else []

    def range_lookup(self, index_name: str, low: int, high: int) -> list:
        """Return list of PKs whose indexed column is in [low, high]."""
        engine = self._engines.get(index_name)
        if engine is None:
            return []
        pks = []
        for _, pk_list in engine.scan(low, high):
            pks.extend(pk_list)
        return pks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_engine(self, name: str):
        from src.engine import Engine
        path   = os.path.join(self._dirpath, f"{name}.db")
        engine = Engine(path)
        self._engines[name] = engine
        return engine

    def _add(self, index_name: str, col_val: int, pk_val: int) -> None:
        engine   = self._engines[index_name]
        existing = engine.get(col_val)
        if existing is None:
            engine.put(col_val, [pk_val])
        elif pk_val not in existing:
            engine.put(col_val, existing + [pk_val])

    def _remove(self, index_name: str, col_val: int, pk_val: int) -> None:
        engine   = self._engines[index_name]
        existing = engine.get(col_val)
        if existing is None:
            return
        updated = [pk for pk in existing if pk != pk_val]
        if updated:
            engine.put(col_val, updated)
        else:
            engine.delete(col_val)
