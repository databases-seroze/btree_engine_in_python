"""
Catalog — persistent schema registry for the SQL layer.

Stores table definitions (column names, types, primary key) in a JSON sidecar
file at <dirpath>/catalog.json.  The file is rewritten on every schema change
(CREATE TABLE); it is small enough that this is fine.

Schema format on disk
---------------------
{
  "users": {
    "columns": [
      {"name": "id",   "type": "INT",    "pk": true},
      {"name": "name", "type": "STRING", "pk": false},
      {"name": "age",  "type": "INT",    "pk": false}
    ],
    "pk_col": "id",
    "pk_idx": 0
  }
}
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
        self._path   = os.path.join(dirpath, 'catalog.json')
        self._tables: dict = {}   # name → TableSchema
        if os.path.exists(self._path):
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_table(self, name: str, columns: list) -> TableSchema:
        """
        Register a new table.

        Parameters
        ----------
        name    : str           Table name.
        columns : list[ColumnDef]   Column definitions from the AST.

        Raises CatalogError if the table already exists.
        """
        if name in self._tables:
            raise CatalogError(f"Table '{name}' already exists")

        pk_cols = [c for c in columns if c.primary_key]
        pk_col  = pk_cols[0]
        pk_idx  = next(i for i, c in enumerate(columns) if c.primary_key)

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
        """Return the schema for *name*.  Raises CatalogError if not found."""
        if name not in self._tables:
            raise CatalogError(f"Table '{name}' does not exist")
        return self._tables[name]

    def table_exists(self, name: str) -> bool:
        return name in self._tables

    def all_tables(self) -> list:
        """Return a list of all registered table names."""
        return list(self._tables.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        data = {}
        for name, s in self._tables.items():
            data[name] = {
                'columns': [
                    {'name': c.name, 'type': c.col_type, 'pk': c.primary_key}
                    for c in s.columns
                ],
                'pk_col': s.pk_col,
                'pk_idx': s.pk_idx,
            }
        with open(self._path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        with open(self._path) as f:
            data = json.load(f)
        for name, td in data.items():
            cols = [
                ColumnSchema(c['name'], c['type'], c['pk'])
                for c in td['columns']
            ]
            self._tables[name] = TableSchema(
                name   = name,
                columns= cols,
                pk_col = td['pk_col'],
                pk_idx = td['pk_idx'],
            )
