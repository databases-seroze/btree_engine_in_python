"""
Parser — converts a flat list of Tokens into an AST.

Entry point
-----------
    from src.parser import parse

    stmt = parse("SELECT * FROM users WHERE id = 1")

Or manually:

    tokens = Lexer(sql).tokenize()
    stmt   = Parser(tokens).parse()

Supported statements
--------------------
    CREATE TABLE t (col1 INT PRIMARY KEY, col2 STRING, col3 INT)
    INSERT INTO t VALUES (1, 'alice', 30)
    SELECT * FROM t
    SELECT col1, col2 FROM t WHERE col1 = 1
    SELECT * FROM t WHERE col1 BETWEEN 1 AND 100
    SELECT * FROM t WHERE col1 > 5 AND col2 = 'bob'
    UPDATE t SET col2 = 'carol' WHERE col1 = 1
    UPDATE t SET col2 = 'x', col3 = 99 WHERE col1 = 1
    DELETE FROM t WHERE col1 = 1
    CREATE INDEX idx_age ON users (age)
    DROP INDEX idx_age
"""

from src.lexer    import Lexer, Token, TT
from src.ast_nodes import (
    ColumnDef,
    EqExpr, CompareExpr, BetweenExpr, AndExpr, OrExpr,
    CreateTableStmt, InsertStmt, SelectStmt, UpdateStmt, DeleteStmt,
    CreateIndexStmt, DropIndexStmt,
)


class ParseError(Exception):
    pass


class Parser:

    def __init__(self, tokens: list):
        self._tokens = tokens
        self._pos    = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TT.EOF:
            self._pos += 1
        return tok

    def _expect(self, tt: TT) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) "
                f"at position {tok.pos}"
            )
        return tok

    def _match(self, *types) -> bool:
        return self._peek().type in types

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse(self):
        tt = self._peek().type
        if tt == TT.SELECT:
            return self._parse_select()
        if tt == TT.INSERT:
            return self._parse_insert()
        if tt == TT.UPDATE:
            return self._parse_update()
        if tt == TT.DELETE:
            return self._parse_delete()
        if tt == TT.DROP:
            return self._parse_drop_index()
        if tt == TT.CREATE:
            self._advance()   # consume CREATE
            if self._match(TT.TABLE):
                return self._parse_create_table_body()
            if self._match(TT.INDEX):
                return self._parse_create_index_body()
            tok = self._peek()
            raise ParseError(
                f"Expected TABLE or INDEX after CREATE, got {tok.type.name} at pos {tok.pos}"
            )
        raise ParseError(
            f"Unexpected token {self._peek().type.name!r} at position {self._peek().pos}"
        )

    # ------------------------------------------------------------------
    # CREATE TABLE
    # ------------------------------------------------------------------

    def _parse_create_table(self) -> CreateTableStmt:
        # Legacy entry point (CREATE not yet consumed).
        self._expect(TT.CREATE)
        return self._parse_create_table_body()

    def _parse_create_table_body(self) -> CreateTableStmt:
        self._expect(TT.TABLE)
        table = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)

        columns = []
        while not self._match(TT.RPAREN, TT.EOF):
            columns.append(self._parse_column_def())
            if self._match(TT.COMMA):
                self._advance()

        self._expect(TT.RPAREN)

        pks = [c for c in columns if c.primary_key]
        if len(pks) != 1:
            raise ParseError(
                f"Table must have exactly one PRIMARY KEY column, got {len(pks)}"
            )
        if pks[0].col_type != 'INT':
            raise ParseError("PRIMARY KEY column must be of type INT")

        return CreateTableStmt(table=table, columns=columns)

    def _parse_create_index_body(self) -> CreateIndexStmt:
        self._expect(TT.INDEX)
        name  = self._expect(TT.IDENT).value
        self._expect(TT.ON)
        table = self._expect(TT.IDENT).value
        self._expect(TT.LPAREN)
        col   = self._expect(TT.IDENT).value
        self._expect(TT.RPAREN)
        return CreateIndexStmt(name=name, table=table, col=col)

    def _parse_drop_index(self) -> DropIndexStmt:
        self._expect(TT.DROP)
        self._expect(TT.INDEX)
        name = self._expect(TT.IDENT).value
        return DropIndexStmt(name=name)

    def _parse_column_def(self) -> ColumnDef:
        name = self._expect(TT.IDENT).value

        if self._match(TT.INT):
            col_type = 'INT'
            self._advance()
        elif self._match(TT.STRING):
            col_type = 'STRING'
            self._advance()
        else:
            tok = self._peek()
            raise ParseError(
                f"Expected column type INT or STRING, got {tok.type.name} at pos {tok.pos}"
            )

        pk = False
        if self._match(TT.PRIMARY):
            self._advance()
            self._expect(TT.KEY)
            pk = True

        return ColumnDef(name=name, col_type=col_type, primary_key=pk)

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------

    def _parse_insert(self) -> InsertStmt:
        self._expect(TT.INSERT)
        self._expect(TT.INTO)
        table = self._expect(TT.IDENT).value
        self._expect(TT.VALUES)
        self._expect(TT.LPAREN)

        values = []
        while not self._match(TT.RPAREN, TT.EOF):
            values.append(self._parse_literal())
            if self._match(TT.COMMA):
                self._advance()

        self._expect(TT.RPAREN)
        return InsertStmt(table=table, values=values)

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------

    def _parse_select(self) -> SelectStmt:
        self._expect(TT.SELECT)
        columns = self._parse_column_list()
        self._expect(TT.FROM)
        table = self._expect(TT.IDENT).value

        where = None
        if self._match(TT.WHERE):
            self._advance()
            where = self._parse_where()

        return SelectStmt(table=table, columns=columns, where=where)

    def _parse_column_list(self) -> list:
        if self._match(TT.STAR):
            self._advance()
            return ['*']
        cols = [self._expect(TT.IDENT).value]
        while self._match(TT.COMMA):
            self._advance()
            cols.append(self._expect(TT.IDENT).value)
        return cols

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def _parse_update(self) -> UpdateStmt:
        self._expect(TT.UPDATE)
        table = self._expect(TT.IDENT).value
        self._expect(TT.SET)

        assignments = {}
        col = self._expect(TT.IDENT).value
        self._expect(TT.EQ)
        assignments[col] = self._parse_literal()
        while self._match(TT.COMMA):
            self._advance()
            col = self._expect(TT.IDENT).value
            self._expect(TT.EQ)
            assignments[col] = self._parse_literal()

        where = None
        if self._match(TT.WHERE):
            self._advance()
            where = self._parse_where()

        return UpdateStmt(table=table, assignments=assignments, where=where)

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def _parse_delete(self) -> DeleteStmt:
        self._expect(TT.DELETE)
        self._expect(TT.FROM)
        table = self._expect(TT.IDENT).value

        where = None
        if self._match(TT.WHERE):
            self._advance()
            where = self._parse_where()

        return DeleteStmt(table=table, where=where)

    # ------------------------------------------------------------------
    # WHERE clause
    # ------------------------------------------------------------------

    def _parse_where(self):
        left = self._parse_predicate()
        while self._match(TT.AND):
            self._advance()
            # Peek: if this AND is part of BETWEEN it will have been consumed
            # already inside _parse_predicate. Here we're at a top-level AND.
            right = self._parse_predicate()
            left  = AndExpr(left=left, right=right)
        return left

    def _parse_predicate(self):
        col = self._expect(TT.IDENT).value

        if self._match(TT.BETWEEN):
            self._advance()
            low = self._parse_literal()
            self._expect(TT.AND)
            high = self._parse_literal()
            return BetweenExpr(col=col, low=low, high=high)

        if self._match(TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE):
            op_tok = self._advance()
            value  = self._parse_literal()
            if op_tok.type == TT.EQ:
                return EqExpr(col=col, value=value)
            return CompareExpr(col=col, op=op_tok.value, value=value)

        tok = self._peek()
        raise ParseError(
            f"Expected comparison operator after column '{col}', "
            f"got {tok.type.name} at position {tok.pos}"
        )

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def _parse_literal(self):
        tok = self._peek()
        if tok.type == TT.INT_LIT:
            self._advance()
            return tok.value
        if tok.type == TT.STR_LIT:
            self._advance()
            return tok.value
        if tok.type == TT.NULL:
            self._advance()
            return None
        raise ParseError(
            f"Expected a literal value (integer, string, or NULL), "
            f"got {tok.type.name} at position {tok.pos}"
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def parse(sql: str):
    """Lex and parse *sql* in one call; returns the AST root node."""
    tokens = Lexer(sql).tokenize()
    return Parser(tokens).parse()
