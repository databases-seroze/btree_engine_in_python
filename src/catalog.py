"""
Catalog — persistent schema registry for the SQL layer.

Stores table definitions and index definitions in a JSON sidecar file at
<dirpath>/catalog.json.  The file is rewritten on every schema change.

Schema format on disk (v2)
--------------------------
{
  "tables": {
    "users": {
      "columns": [
        {"name": "id",   "type": "INT",    "pk": true},
        {"name": "name", "type": "STRING", "pk": false},
        {"name": "age",  "type": "INT",    "pk": false}
      ],
      "pk_col": "id",
      "pk_idx": 0
    }
  },
  "indexes": {
    "idx_age": {"table": "users", "col": "age", "col_idx": 2}
  }
}

Backward compatibility
----------------------
Files written by the old format (tables at top level, no "tables" wrapper)
are loaded transparently; they are re-saved in the new format on the first
schema change.
"""

import json
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    name:        str
    col_type:    str    # 'INT' | 'STRING'
    primary_key: bool


@dataclass
class TableSchema:
    name:    str
    columns: list   # list[ColumnSchema]
    pk_col:  str    # name of the primary key column
    pk_idx:  int    # index of pk column in columns list


@dataclass
class IndexSchema:
    name:    str   # index name
    table:   str   # table it covers
    col:     str   # indexed column name
    col_idx: int   # position of that column in the table's column list


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class CatalogError(Exception):
    pass


class Catalog:
    """
    In-memory schema registry backed by a JSON file.

    Parameters
    ----------
    dirpath : str
        Directory that contains (or will contain) catalog.json.
    """

    def __init__(self, dirpath: str):
        self._path    = os.path.join(dirpath, 'catalog.json')
        self._tables: dict = {}   # name → TableSchema
        self._indexes: dict = {}  # name → IndexSchema
        if os.path.exists(self._path):
            self._load()

    # ------------------------------------------------------------------
    # Table API
    # ------------------------------------------------------------------

    def create_table(self, name: str, columns: list) -> TableSchema:
        if name in self._tables:
            raise CatalogError(f"Table '{name}' already exists")

        pk_col = next(c for c in columns if c.primary_key)
        pk_idx = next(i for i, c in enumerate(columns) if c.primary_key)

        schema = TableSchema(
            name    = name,
            columns = [ColumnSchema(c.name, c.col_type, c.primary_key) for c in columns],
            pk_col  = pk_col.name,
            pk_idx  = pk_idx,
        )
        self._tables[name] = schema
        self._save()
        return schema

    def get_table(self, name: str) -> TableSchema:
        if name not in self._tables:
            raise CatalogError(f"Table '{name}' does not exist")
        return self._tables[name]

    def table_exists(self, name: str) -> bool:
        return name in self._tables

    def all_tables(self) -> list:
        return list(self._tables.keys())

    # ------------------------------------------------------------------
    # Index API
    # ------------------------------------------------------------------

    def create_index(self, name: str, table: str, col: str) -> IndexSchema:
        """
        Register a new index.

        Raises CatalogError if:
        - *name* already taken
        - *table* does not exist
        - *col* does not exist in *table*
        - *col* is not of type INT (B+Tree keys are uint32)
        """
        if name in self._indexes:
            raise CatalogError(f"Index '{name}' already exists")
        schema = self.get_table(table)
        col_schema = next((c for c in schema.columns if c.name == col), None)
        if col_schema is None:
            raise CatalogError(f"Column '{col}' does not exist in table '{table}'")
        if col_schema.col_type != 'INT':
            raise CatalogError(
                f"Column '{col}' is of type {col_schema.col_type}; "
                f"secondary indexes are only supported on INT columns"
            )
        col_idx = next(i for i, c in enumerate(schema.columns) if c.name == col)

        idx = IndexSchema(name=name, table=table, col=col, col_idx=col_idx)
        self._indexes[name] = idx
        self._save()
        return idx

    def drop_index(self, name: str) -> None:
        if name not in self._indexes:
            raise CatalogError(f"Index '{name}' does not exist")
        del self._indexes[name]
        self._save()

    def get_index(self, name: str) -> IndexSchema:
        if name not in self._indexes:
            raise CatalogError(f"Index '{name}' does not exist")
        return self._indexes[name]

    def index_exists(self, name: str) -> bool:
        return name in self._indexes

    def get_indexes_for_table(self, table: str) -> list:
        """Return all IndexSchema objects for the given table."""
        return [idx for idx in self._indexes.values() if idx.table == table]

    def all_indexes(self) -> list:
        return list(self._indexes.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        tables_data = {}
        for name, s in self._tables.items():
            tables_data[name] = {
                'columns': [
                    {'name': c.name, 'type': c.col_type, 'pk': c.primary_key}
                    for c in s.columns
                ],
                'pk_col': s.pk_col,
                'pk_idx': s.pk_idx,
            }
        indexes_data = {}
        for name, idx in self._indexes.items():
            indexes_data[name] = {
                'table':   idx.table,
                'col':     idx.col,
                'col_idx': idx.col_idx,
            }
        with open(self._path, 'w') as f:
            json.dump({'tables': tables_data, 'indexes': indexes_data}, f, indent=2)

    def _load(self):
        with open(self._path) as f:
            data = json.load(f)

        # Backward-compatible: old format has tables at top level (no 'tables' key)
        if 'tables' in data:
            tables_data  = data['tables']
            indexes_data = data.get('indexes', {})
        else:
            tables_data  = data
            indexes_data = {}

        for name, td in tables_data.items():
            cols = [
                ColumnSchema(c['name'], c['type'], c['pk'])
                for c in td['columns']
            ]
            self._tables[name] = TableSchema(
                name    = name,
                columns = cols,
                pk_col  = td['pk_col'],
                pk_idx  = td['pk_idx'],
            )

        for name, id in indexes_data.items():
            self._indexes[name] = IndexSchema(
                name    = name,
                table   = id['table'],
                col     = id['col'],
                col_idx = id['col_idx'],
            )
