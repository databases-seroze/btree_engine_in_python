"""
Lexer — tokenises a SQL string into a flat list of Tokens.

Supported tokens
----------------
Keywords : SELECT FROM WHERE INSERT INTO VALUES UPDATE SET DELETE
           CREATE TABLE PRIMARY KEY BETWEEN AND OR NOT NULL INT STRING
Symbols  : * ( ) , = != < > <= >=
Literals : integer (signed), string (single or double quoted, backslash escape)
Other    : IDENT (table / column names), EOF

Case-insensitive for keywords; identifiers preserve case.
"""

from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TT(Enum):
    # Keywords
    SELECT  = auto()
    FROM    = auto()
    WHERE   = auto()
    INSERT  = auto()
    INTO    = auto()
    VALUES  = auto()
    UPDATE  = auto()
    SET     = auto()
    DELETE  = auto()
    CREATE  = auto()
    TABLE   = auto()
    PRIMARY = auto()
    KEY     = auto()
    BETWEEN = auto()
    AND     = auto()
    OR      = auto()
    NOT     = auto()
    NULL    = auto()
    INT     = auto()
    STRING  = auto()
    # Symbols
    STAR   = auto()   # *
    LPAREN = auto()   # (
    RPAREN = auto()   # )
    COMMA  = auto()   # ,
    EQ     = auto()   # =
    NEQ    = auto()   # !=
    LT     = auto()   # <
    GT     = auto()   # >
    LTE    = auto()   # <=
    GTE    = auto()   # >=
    # Literals / names
    INT_LIT = auto()
    STR_LIT = auto()
    IDENT   = auto()
    # End of input
    EOF = auto()


_KEYWORDS: dict[str, TT] = {
    'SELECT':  TT.SELECT,
    'FROM':    TT.FROM,
    'WHERE':   TT.WHERE,
    'INSERT':  TT.INSERT,
    'INTO':    TT.INTO,
    'VALUES':  TT.VALUES,
    'UPDATE':  TT.UPDATE,
    'SET':     TT.SET,
    'DELETE':  TT.DELETE,
    'CREATE':  TT.CREATE,
    'TABLE':   TT.TABLE,
    'PRIMARY': TT.PRIMARY,
    'KEY':     TT.KEY,
    'BETWEEN': TT.BETWEEN,
    'AND':     TT.AND,
    'OR':      TT.OR,
    'NOT':     TT.NOT,
    'NULL':    TT.NULL,
    'INT':     TT.INT,
    'STRING':  TT.STRING,
}


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type:  TT
    value: object   # str | int | None
    pos:   int      # byte offset in source string (for error messages)


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class LexError(Exception):
    pass


class Lexer:
    """
    Tokenise *text* into a list of Token objects ending with EOF.

    Usage
    -----
        tokens = Lexer("SELECT * FROM users").tokenize()
    """

    def __init__(self, text: str):
        self._text = text
        self._pos  = 0

    def tokenize(self) -> list:
        tokens = []
        while True:
            tok = self._next_token()
            tokens.append(tok)
            if tok.type == TT.EOF:
                break
        return tokens

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_token(self) -> Token:
        self._skip_ws()
        if self._pos >= len(self._text):
            return Token(TT.EOF, None, self._pos)

        pos = self._pos
        ch  = self._text[pos]

        # String literal
        if ch in ('"', "'"):
            return self._read_string(ch)

        # Negative integer: '-' followed by digit
        if ch == '-' and self._pos + 1 < len(self._text) and self._text[self._pos + 1].isdigit():
            return self._read_int()

        # Positive integer
        if ch.isdigit():
            return self._read_int()

        # Identifier or keyword
        if ch.isalpha() or ch == '_':
            return self._read_ident()

        # Single- and double-character symbols
        self._pos += 1

        if ch == '*':
            return Token(TT.STAR,   '*',  pos)
        if ch == '(':
            return Token(TT.LPAREN, '(',  pos)
        if ch == ')':
            return Token(TT.RPAREN, ')',  pos)
        if ch == ',':
            return Token(TT.COMMA,  ',',  pos)
        if ch == '=':
            return Token(TT.EQ,     '=',  pos)
        if ch == '<':
            if self._pos < len(self._text) and self._text[self._pos] == '=':
                self._pos += 1
                return Token(TT.LTE, '<=', pos)
            return Token(TT.LT, '<', pos)
        if ch == '>':
            if self._pos < len(self._text) and self._text[self._pos] == '=':
                self._pos += 1
                return Token(TT.GTE, '>=', pos)
            return Token(TT.GT, '>', pos)
        if ch == '!':
            if self._pos < len(self._text) and self._text[self._pos] == '=':
                self._pos += 1
                return Token(TT.NEQ, '!=', pos)
            raise LexError(f"Unexpected '!' without '=' at position {pos}")

        raise LexError(f"Unexpected character {ch!r} at position {pos}")

    def _skip_ws(self):
        while self._pos < len(self._text) and self._text[self._pos] in ' \t\n\r':
            self._pos += 1

    def _read_string(self, quote: str) -> Token:
        pos = self._pos
        self._pos += 1  # skip opening quote
        buf = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == quote:
                self._pos += 1
                return Token(TT.STR_LIT, ''.join(buf), pos)
            if ch == '\\' and self._pos + 1 < len(self._text):
                self._pos += 1
                buf.append(self._text[self._pos])
            else:
                buf.append(ch)
            self._pos += 1
        raise LexError(f"Unterminated string literal starting at position {pos}")

    def _read_int(self) -> Token:
        pos = self._pos
        if self._text[self._pos] == '-':
            self._pos += 1
        while self._pos < len(self._text) and self._text[self._pos].isdigit():
            self._pos += 1
        return Token(TT.INT_LIT, int(self._text[pos:self._pos]), pos)

    def _read_ident(self) -> Token:
        pos = self._pos
        while self._pos < len(self._text) and (
            self._text[self._pos].isalnum() or self._text[self._pos] == '_'
        ):
            self._pos += 1
        word = self._text[pos:self._pos]
        tt   = _KEYWORDS.get(word.upper())
        if tt is not None:
            return Token(tt, word.upper(), pos)
        return Token(TT.IDENT, word, pos)
