"""
REPL — interactive SQL shell for py_btree_engine.

Usage
-----
    python -m py_btree_engine <dbdir>
    python -m py_btree_engine .          # use current directory

Prompt
------
    db> SELECT * FROM users WHERE age > 25;

Multi-line input is supported — the statement is executed when a line
ending with ';' is received:

    db> SELECT *
    ... FROM users
    ... WHERE age > 25;

Meta-commands (no semicolon)
-----------------------------
    \\tables              list all tables
    \\schema <table>      show column definitions for a table
    \\indexes             list all secondary indexes
    \\help                show this help
    \\quit  or  \\q        exit

Results
-------
Rows are displayed as an aligned ASCII table with column headers and a
row count footer.  Empty result sets print "(0 rows)".
"""

import os
import sys
import readline   # noqa: F401  — side-effect: enables arrow-key history

from src.database  import Database
from src.catalog   import CatalogError
from src.executor  import ExecutionError
from src.lexer     import LexError
from src.parser    import ParseError


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def _print_table(rows: list) -> None:
    """Print *rows* (list of dicts) as an aligned ASCII table."""
    if not rows:
        print("(0 rows)")
        return

    # Collect all column names preserving first-appearance order
    cols = list(dict.fromkeys(k for row in rows for k in row))

    # Column widths = max(header, all values)
    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(row.get(c))))

    sep  = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    hdr  = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"

    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "|" + "|".join(f" {_fmt(row.get(c)):<{widths[c]}} " for c in cols) + "|"
        print(line)
    print(sep)
    n = len(rows)
    print(f"({n} row{'s' if n != 1 else ''})")


def _fmt(value) -> str:
    if value is None:
        return "NULL"
    return str(value)


# ---------------------------------------------------------------------------
# Meta-command handlers
# ---------------------------------------------------------------------------

def _cmd_tables(db: Database) -> None:
    tables = db._catalog.all_tables()
    if not tables:
        print("(no tables)")
        return
    for t in sorted(tables):
        print(f"  {t}")


def _cmd_schema(db: Database, args: str) -> None:
    table = args.strip()
    if not table:
        print("Usage: \\schema <table>")
        return
    try:
        schema = db._catalog.get_table(table)
    except CatalogError as e:
        print(f"Error: {e}")
        return
    rows = [
        {
            "column": c.name,
            "type":   c.col_type,
            "pk":     "YES" if c.primary_key else "",
            "fk":     f"→ {c2.parent_table}.{c2.parent_col}"
                      if (c2 := next(
                          (fk for fk in db._catalog.get_fkeys_from(table)
                           if fk.child_col == c.name), None
                      )) else "",
        }
        for c in schema.columns
    ]
    _print_table(rows)


def _cmd_indexes(db: Database) -> None:
    indexes = db._catalog.all_indexes()
    if not indexes:
        print("(no indexes)")
        return
    rows = [
        {"name": idx.name, "table": idx.table, "column": idx.col}
        for idx in sorted(indexes, key=lambda i: (i.table, i.name))
    ]
    _print_table(rows)


def _cmd_help() -> None:
    print("""
SQL statements (end with ';'):
  CREATE TABLE t (col1 INT PRIMARY KEY, col2 STRING, col3 INT)
  CREATE TABLE t (id INT PRIMARY KEY, fk INT REFERENCES other (id))
  INSERT INTO t VALUES (1, 'alice', 30)
  SELECT * FROM t [WHERE ...]
  SELECT * FROM t JOIN other ON t.col = other.col [WHERE ...]
  UPDATE t SET col = value [WHERE ...]
  DELETE FROM t [WHERE ...]
  CREATE INDEX idx ON t (col)
  DROP INDEX idx

WHERE operators: = != < > <= >= BETWEEN ... AND ...
Compound:        AND (multi-condition)

Meta-commands (no semicolon):
  \\tables              list all tables
  \\schema <table>      show column definitions
  \\indexes             list all secondary indexes
  \\help                show this help
  \\quit  \\q            exit
""".strip())


# ---------------------------------------------------------------------------
# REPL main loop
# ---------------------------------------------------------------------------

def run(dirpath: str) -> None:
    """Open *dirpath* as a Database and start the interactive loop."""
    print(f"py_btree_engine  —  database: {os.path.abspath(dirpath)}")
    print("Type SQL statements ending with ';', or \\help for commands.\n")

    try:
        db = Database(dirpath)
    except Exception as e:
        print(f"Failed to open database: {e}", file=sys.stderr)
        sys.exit(1)

    buf = []   # accumulated lines for the current statement

    with db:
        while True:
            prompt = "db> " if not buf else "... "
            try:
                line = input(prompt)
            except EOFError:
                # Ctrl-D
                print()
                break
            except KeyboardInterrupt:
                # Ctrl-C cancels current input buffer
                print()
                buf.clear()
                continue

            stripped = line.strip()

            # --- Meta-commands (never need a semicolon) ---
            if not buf and stripped.startswith("\\"):
                parts   = stripped.split(None, 1)
                command = parts[0].lower()
                args    = parts[1] if len(parts) > 1 else ""

                if command in ("\\quit", "\\q"):
                    break
                elif command == "\\help":
                    _cmd_help()
                elif command == "\\tables":
                    _cmd_tables(db)
                elif command == "\\schema":
                    _cmd_schema(db, args)
                elif command == "\\indexes":
                    _cmd_indexes(db)
                else:
                    print(f"Unknown command {command!r}. Try \\help.")
                continue

            # --- Accumulate SQL lines ---
            if stripped:
                buf.append(line)

            # Execute when the accumulated buffer ends with ';'
            combined = " ".join(buf).strip()
            if combined.endswith(";"):
                sql = combined[:-1].strip()   # strip trailing semicolon
                buf.clear()
                if not sql:
                    continue
                _execute_and_print(db, sql)

    print("Bye.")


def _execute_and_print(db: Database, sql: str) -> None:
    try:
        rows = db.execute(sql)
        verb = sql.strip().split()[0].upper()
        if verb == "SELECT":
            _print_table(rows)   # always show table, even if empty
        elif rows:
            _print_table(rows)
        else:
            # DDL / DML with no result rows — just confirm
            if verb in ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP"):
                print("OK")
    except (ParseError, LexError) as e:
        print(f"Syntax error: {e}")
    except (ExecutionError, CatalogError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
