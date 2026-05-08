"""
AST node definitions for the SQL parser.

Expressions
-----------
EqExpr        col = value
CompareExpr   col <|>|<=|>=|!= value
BetweenExpr   col BETWEEN low AND high
AndExpr       left AND right
OrExpr        left OR right

Statements
----------
CreateTableStmt   CREATE TABLE ...
InsertStmt        INSERT INTO ... VALUES (...)
SelectStmt        SELECT ... FROM ... WHERE ...
UpdateStmt        UPDATE ... SET ... WHERE ...
DeleteStmt        DELETE FROM ... WHERE ...
"""

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Schema helpers (used in CreateTableStmt)
# ---------------------------------------------------------------------------

@dataclass
class ColumnDef:
    name:        str
    col_type:    str   # 'INT' | 'STRING'
    primary_key: bool = False


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class EqExpr:
    col:   str
    value: Any


@dataclass
class CompareExpr:
    col:   str
    op:    str   # '<' | '>' | '<=' | '>=' | '!='
    value: Any


@dataclass
class BetweenExpr:
    col:  str
    low:  Any
    high: Any


@dataclass
class AndExpr:
    left:  Any
    right: Any


@dataclass
class OrExpr:
    left:  Any
    right: Any


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class CreateTableStmt:
    table:   str
    columns: list   # list[ColumnDef]


@dataclass
class InsertStmt:
    table:  str
    values: list    # raw Python values in column order


@dataclass
class SelectStmt:
    table:   str
    columns: list   # ['*'] or list of column name strings
    where:   Any = None


@dataclass
class UpdateStmt:
    table:       str
    assignments: dict   # col_name → new_value
    where:       Any = None


@dataclass
class DeleteStmt:
    table: str
    where: Any = None


@dataclass
class CreateIndexStmt:
    name:  str   # index name (e.g. "idx_age")
    table: str
    col:   str   # single column being indexed


@dataclass
class DropIndexStmt:
    name: str
