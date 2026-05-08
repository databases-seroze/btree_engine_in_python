"""
Database — top-level SQL interface.

Creates a directory that holds per-table Engine files and a catalog.json.
All SQL goes through execute(); results are returned as lists of dicts.

Usage
-----
    from src.database import Database

    with Database("mydb/") as db:
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name STRING, age INT)")
        db.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        db.execute("INSERT INTO users VALUES (2, 'bob',   25)")

        rows = db.execute("SELECT * FROM users WHERE id = 1")
        # [{'id': 1, 'name': 'alice', 'age': 30}]

        rows = db.execute("SELECT name FROM users WHERE age > 20")
        # [{'name': 'alice'}, {'name': 'bob'}]

        db.execute("UPDATE users SET age = 31 WHERE id = 1")
        db.execute("DELETE FROM users WHERE id = 2")

    # Re-open — schema and data are restored from disk:
    with Database("mydb/") as db:
        rows = db.execute("SELECT * FROM users")

Parameters
----------
dirpath : str
    Path to the database directory.  Created if it does not exist.
"""

import os

from src.catalog       import Catalog
from src.executor      import Executor
from src.index_manager import IndexManager
from src.lexer         import Lexer
from src.parser        import Parser


class Database:
    """
    Top-level SQL interface backed by the B+Tree storage engine.

    Thread safety: a single Database instance should not be shared
    across threads without external locking.  For concurrent access,
    use individual Transaction objects on a shared Engine directly.
    """

    def __init__(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        self._dirpath   = dirpath
        self._catalog   = Catalog(dirpath)
        self._engines   = {}   # table_name → Engine
        self._index_mgr = IndexManager(self._catalog, dirpath)

        # Re-open engines for tables and indexes that already exist on disk.
        for table in self._catalog.all_tables():
            self._reopen_engine(table)
        self._index_mgr.open_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> list:
        """
        Parse and execute *sql*.

        Returns
        -------
        list[dict]
            For SELECT: one dict per matching row.
            For INSERT / UPDATE / DELETE / CREATE TABLE: empty list.
        """
        tokens = Lexer(sql).tokenize()
        stmt   = Parser(tokens).parse()
        return Executor(
            self._catalog, self._engines, self._dirpath, self._index_mgr
        ).execute(stmt)

    def close(self) -> None:
        """Flush and close all open engines and index files."""
        for engine in self._engines.values():
            engine.close()
        self._engines.clear()
        self._index_mgr.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'Database':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reopen_engine(self, table: str) -> None:
        from src.engine import Engine
        path = os.path.join(self._dirpath, f"{table}.db")
        self._engines[table] = Engine(path)
