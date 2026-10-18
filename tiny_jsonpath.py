"""
tiny-jsonpath — RFC 9535 (JSONPath 1.0) parser and evaluator, stdlib only.

JSONPath is a query language for JSON, analogous to XPath for XML. The IETF
finalized JSONPath 1.0 as RFC 9535 in 2024, and it is now supported across
most JSON-processing tooling (jq, Postgres 16+, jq replacements, kubectl
templates). This module is a pure-Python, zero-dependency implementation
of the core RFC 9535 syntax.

Scope:
  * Root selector: ``$``
  * Child member: ``.name`` (and bracket form ``['name']``)
  * Array index: ``[i]`` (positive and negative)
  * Array slice: ``[start:end:step]``
  * Wildcard: ``[*]`` and ``.*``
  * Descendant: ``..name`` (recursive descent through objects and arrays)
  * Union: ``[a, b, c]`` and ``.a.b.c`` (multi-name shorthand)
  * Filters: ``[?(@.x > 5)]`` — supports comparisons, ``&&`` / ``||``,
    ``@``, and the safe navigation subset (no function calls).

Out of scope (deliberately):
  * Script expressions (the original Stefan Gössner extension).
  * Function calls inside filters (``length()``, ``match()``, etc.).
  * Anything outside the RFC 9535 grammar.

Usage:
    >>> from tiny_jsonpath import find, parse, find_one
    >>> doc = {"users": [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]}
    >>> find("$.users[*].name", doc)
    ['alice', 'bob']
    >>> find_one("$.users[?(@.age >= 28)].name", doc)
    'alice'

Errors:
    All parse / evaluation errors raise ``JSONPathError`` with a position
    pointer so the user can fix their query.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, Union

__version__ = "0.1.0"
__all__ = [
    "find",
    "find_one",
    "parse",
    "JSONPathError",
    "Selector",
    "Root",
    "Name",
    "Index",
    "Slice",
    "Wildcard",
    "Descendant",
    "Union",
    "Filter",
]


# Sentinel for slices that mean "from start" / "to end".
_UNSET = object()


class JSONPathError(Exception):
    """Raised for any parse or evaluation error.

    The ``position`` attribute (when present) is the 1-based character
    offset in the original JSONPath expression. The message includes a
    caret-pointed excerpt so the caller can show it inline.
    """

    def __init__(self, message: str, source: str = "", position: int = -1):
        self.position = position
        self.source = source
        if position >= 0 and source:
            line = source[:position].count("\n") + 1
            col = position - source.rfind("\n", 0, position)
            pointer = "\n" + "\n".join(source.splitlines()[:line]) + "\n" + (" " * (col - 1)) + "^"
            super().__init__(f"{message} (line {line}, col {col}){pointer}")
        else:
            super().__init__(message)


# ── AST nodes ─────────────────────────────────────────────────────────────────

class Selector:
    """Base class for JSONPath selectors."""


class Root(Selector):
    """The ``$`` root selector."""

    def __repr__(self) -> str:
        return "$"


class Name(Selector):
    """``.name`` selects the named field from each input object."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f".{self.name}"


class Index(Selector):
    """``[i]`` selects the array element at index ``i``."""

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def __repr__(self) -> str:
        return f"[{self.index}]"


class Slice(Selector):
    """``[start:end:step]`` selects an array slice."""

    __slots__ = ("start", "end", "step")

    def __init__(self, start: Any, end: Any, step: int) -> None:
        # Each of ``start`` / ``end`` may be ``None`` (omitted) or an int.
        # ``step`` is always an int (default 1) but never 0.
        self.start = start
        self.end = end
        self.step = step

    def __repr__(self) -> str:
        parts = []
        if self.start is not None:
            parts.append(str(self.start))
        parts.append(":")
        if self.end is not None:
            parts.append(str(self.end))
        if self.step != 1:
            parts.append(":" + str(self.step))
        return "[" + "".join(parts) + "]"


class Wildcard(Selector):
    """``[*]`` or ``.*`` — selects all values from the current node."""

    def __repr__(self) -> str:
        return "[*]"


class Descendant(Selector):
    """``..name`` — recursive descent, then a name.

    The ``member`` attribute is ``None`` for the bare descendant operator
    used at the start of a path (``$..foo`` is equivalent to
    ``$..[*].foo``).
    """

    __slots__ = ("member",)

    def __init__(self, member: Optional[str]) -> None:
        self.member = member

    def __repr__(self) -> str:
        return f"..{self.member}" if self.member else "..[*]"


class Union(Selector):
    """``[a, b, c]`` — union of selectors, evaluated as a single step."""

    __slots__ = ("selectors",)

    def __init__(self, selectors: List[Selector]) -> None:
        self.selectors = selectors

    def __repr__(self) -> str:
        return "[" + ", ".join(repr(s) for s in self.selectors) + "]"


class Filter(Selector):
    """``[?expr]`` — filter on the current array.

    ``expr`` is a parsed ``FilterExpr`` AST (see ``_parse_filter``).
    """

    __slots__ = ("expr",)

    def __init__(self, expr: "FilterExpr") -> None:
        self.expr = expr

    def __repr__(self) -> str:
        return f"[?{self.expr}]"


class FilterExpr:
    """Base class for filter AST nodes."""


class LogicalOp(FilterExpr):
    """``&&`` / ``||`` with short-circuit semantics."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: FilterExpr, right: FilterExpr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class ComparisonOp(FilterExpr):
    """``==`` / ``!=`` / ``<`` / ``<=`` / ``>`` / ``>=`` / ``=~``."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: FilterExpr, right: FilterExpr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class Path(FilterExpr):
    """A path inside a filter expression. ``@`` refers to the current node;
    ``@.foo`` selects its ``foo`` member; ``@.foo[0]`` indexes into an
    array member.
    """

    __slots__ = ("segments",)

    def __init__(self, segments: List[Selector]) -> None:
        self.segments = segments

    def __repr__(self) -> str:
        return "@" + "".join(repr(s) for s in self.segments)


class Literal(FilterExpr):
    """A literal value — string, number, true, false, or null."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


# ── Parser ────────────────────────────────────────────────────────────────────

# Tokeniser for non-path parts of a filter expression.
_LITERAL_RE = re.compile(r"""
    \s*(?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | \s*(?P<true>true)
    | \s*(?P<false>false)
    | \s*(?P<null>null)
    | \s*\"(?P<str>(?:\\.|[^\\"])*)\"
    | \s*'(?P<sstr>(?:\\.|[^\\'])*)'
""", re.VERBOSE)


def _parse_literal(src: str, pos: int) -> Tuple[FilterExpr, int]:
    m = _LITERAL_RE.match(src, pos)
    if not m:
        raise JSONPathError("Expected literal", src, pos)
    if m["num"]:
        s = m["num"]
        val: Any = int(s) if "." not in s and "e" not in s.lower() else float(s)
    elif m["true"]:
        val = True
    elif m["false"]:
        val = False
    elif m["null"]:
        val = None
    elif m["str"] is not None:
        val = _unescape_string(m["str"])
    else:  # sstr
        val = _unescape_string(m["sstr"])
    return Literal(val), m.end()


def _unescape_string(s: str) -> str:
    """Process JSON string escapes."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 1
            if i >= len(s):
                raise JSONPathError("Trailing backslash in string literal", "")
            esc = s[i]
            mapping = {
                "\"": "\"",
                "'": "'",
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if esc in mapping:
                out.append(mapping[esc])
            elif esc == "u":
                hexpart = s[i + 1:i + 5]
                if len(hexpart) != 4:
                    raise JSONPathError("Bad unicode escape", "")
                out.append(chr(int(hexpart, 16)))
                i += 4
            else:
                raise JSONPathError(f"Bad string escape: \\{esc}", "")
        else:
            out.append(c)
        i += 1
    return "".join(out)


class _Parser:
    """Recursive descent parser for JSONPath expressions.

    Grammar (simplified):
        path      = '$' segment*
        segment   = '.' child | '..' child | '[' bracket ']'
        child     = NAME | '*'
        bracket   = index | slice | '*' | filter | union
        filter    = '?' logical
        logical   = comparison ( (&&|\|\|) logical )?
        comparison= primary (op primary)?
        primary   = '@' segment* | literal
        op        = '==' | '!=' | '<' | '<=' | '>' | '>=' | '=~'
    """

    __slots__ = ("src", "pos", "length")

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0
        self.length = len(src)

    def _eof(self) -> bool:
        return self.pos >= self.length

    def _peek(self) -> str:
        return "" if self._eof() else self.src[self.pos]

    def _consume(self) -> str:
        c = self.src[self.pos]
        self.pos += 1
        return c

    def _expect(self, s: str) -> None:
        if self.src[self.pos:self.pos + len(s)] != s:
            raise JSONPathError(f"Expected '{s}'", self.src, self.pos)
        self.pos += len(s)

    def _skip_ws(self) -> None:
        while not self._eof() and self.src[self.pos].isspace():
            self.pos += 1

    def _match_name(self) -> Optional[str]:
        # RFC 9531: member-name-shorthand =/ FIRST-CHAR *( NON-FIRST-CHAR *("." NON-FIRST-CHAR ) )
        # In English: a member name starts with a letter or _ and continues
        # with letters, digits, underscores, or dots (for dotted keys).
        # RFC 9535 member-name-shorthand: a single name (letters, digits,
        # underscores). Names with dots must use the bracket form
        # ``['a.b']`` because the dot is the segment separator.
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.src[self.pos:])
        if not m:
            return None
        # Reject reserved words like 'true' / 'false' / 'null' at this
        # position: they should never appear as member names without a
        # quoting bracket form.
        if m.group(0) in ("true", "false", "null"):
            return None
        self.pos += len(m.group(0))
        return m.group(0)

    def parse(self) -> List[Selector]:
        if self._eof() or self._peek() != "$":
            raise JSONPathError("JSONPath must start with '$'", self.src, self.pos)
        self.pos += 1
        segments: List[Selector] = [Root()]
        while not self._eof():
            seg = self._parse_segment()
            if seg is not None:
                segments.append(seg)
            else:
                break
        if not self._eof():
            raise JSONPathError("Unexpected trailing input", self.src, self.pos)
        return segments

    def _parse_segment(self) -> Optional[Selector]:
        c = self._peek()
        if c == ".":
            self.pos += 1
            if not self._eof() and self.src[self.pos] == ".":
                # Recursive descent
                self.pos += 1
                name = self._match_name()
                return Descendant(name)
            # Member name shorthand or wildcard
            if self._peek() == "*":
                self.pos += 1
                return Name("*")  # we represent ``.*`` as Name('*') so the
                                  # evaluator knows to fan out into values.
            name = self._match_name()
            if name is None:
                raise JSONPathError("Expected member name after '.'", self.src, self.pos)
            return Name(name)
        if c == "[":
            return self._parse_bracket()
        if c == "..":
            self.pos += 2
            name = self._match_name()
            return Descendant(name)
        return None

    def _parse_bracket(self) -> Selector:
        self._expect("[")
        self._skip_ws()
        if self._peek() == "*":
            self.pos += 1
            self._skip_ws()
            self._expect("]")
            return Wildcard()
        if self._peek() == "?":
            self.pos += 1
            # The filter expression itself supports grouping via parens
            # (see ``_parse_filter_primary``), so we always delegate to
            # ``_parse_logical`` and let it parse the parens if present.
            # RFC 9535: ``[?(expr)]`` and ``[?expr]`` are both legal;
            # ``_parse_filter_primary`` handles the optional ``(`` itself.
            expr = self._parse_logical()
            self._skip_ws()
            self._expect("]")
            return Filter(expr)
        # Otherwise: index, slice, union, or quoted name.
        # Try to parse a union (comma-separated selectors) first: a union
        # always has more than one item, separated by a comma not adjacent
        # to a colon or comma.
        return self._parse_union_or_simple()

    def _parse_union_or_simple(self) -> Selector:
        """Parse bracketed content: an int, a slice, a union of ints/
        names, or a single quoted name. Dispatches to a dedicated
        union handler when the content has more than one comma-
        separated element; otherwise returns a single selector.
        """
        # The whole function is structured to never rewind the parser
        # so the original ``pos`` at entry is preserved as a fallback.
        save = self.pos
        # Detect union by peeking for a comma AFTER the first element.
        first_end = self._peek_bracket_element_end()
        if first_end == -1:
            # No terminator found before end-of-input.
            raise JSONPathError("Unterminated bracket", self.src, self.pos)
        if self.src[first_end] == ",":
            # Union: parse each comma-separated element.
            self.pos = save
            return self._parse_union()
        # Not a union. Either a slice/index (next char is `:` or digit
        # followed by `]`) or a single quoted name.
        self.pos = save
        if self.src[self.pos] in ("'", '"'):
            quote = self.src[self.pos]
            self.pos += 1
            end = self.src.find(quote, self.pos)
            if end == -1:
                raise JSONPathError("Unterminated quoted name", self.src, self.pos)
            name = _unescape_string(self.src[self.pos:end])
            self.pos = end + 1
            self._skip_ws()
            self._expect("]")
            return Name(name)
        # Slice or int.
        sel = self._try_slice_or_index()
        if sel is None:
            raise JSONPathError("Bad bracket content", self.src, self.pos)
        return sel

    def _peek_bracket_element_end(self) -> int:
        """Return the position of the first char that terminates the
        current bracket element - either ``,`` (union) or ``]``
        (single element). Returns -1 if no terminator before EOF.

        Handles parens that might appear inside filter expressions -
        but for the bracketed-element peek we only care about top-
        level commas, so we stop at the first unbalanced ``]``.
        """
        depth = 0
        i = self.pos
        while i < self.length:
            c = self.src[i]
            if c == "[":
                depth += 1
            elif c == "]":
                if depth == 0:
                    return i
                depth -= 1
            elif c == "," and depth == 0:
                return i
            i += 1
        return -1

    def _parse_union(self) -> Selector:
        """Parse ``[a, b, c]`` and return a Union (or a single
        selector if there's only one element).
        """
        selectors: List[Selector] = []
        while True:
            self._skip_ws()
            if self._peek() == "]":
                self.pos += 1
                break
            # Parse one element: int, signed int, slice, or quoted name.
            if self.src[self.pos] in ("'", '"'):
                quote = self.src[self.pos]
                self.pos += 1
                end = self.src.find(quote, self.pos)
                if end == -1:
                    raise JSONPathError("Unterminated quoted name", self.src, self.pos)
                selectors.append(Name(_unescape_string(self.src[self.pos:end])))
                self.pos = end + 1
            elif (self.src[self.pos].isdigit()
                  or self.src[self.pos] == "-"
                  or self.src[self.pos] == ":"):
                sel = self._try_slice_or_index(consume_close=False)
                if sel is None:
                    raise JSONPathError("Bad union member", self.src, self.pos)
                selectors.append(sel)
            else:
                raise JSONPathError("Unexpected character in union", self.src, self.pos)
            self._skip_ws()
            if self._peek() == ",":
                self.pos += 1
                continue
            if self._peek() == "]":
                self.pos += 1
                break
            raise JSONPathError("Expected ',' or ']'", self.src, self.pos)
        if len(selectors) == 1:
            return selectors[0]
        return Union(selectors)

    def _try_slice_or_index(self, consume_close: bool = True) -> Optional[Selector]:
        """Parse a bracketed int, signed int, or slice.

        Handles all four slice forms:
          [start:end:step]
          [:end:step]
          [start::step]
          [start:end]
          [:end]
          [start:]
          [:]
          [::step]
          and the simple forms [int] / [-int] for index access.

        On failure (e.g. mid-bracket comma), returns None and leaves
        ``self.pos`` at the failure point so the caller can decide
        whether to enter union mode (comman after an int) or report a
        parse error (anything else).
        """
        save = self.pos
        try:
            start_val: Any = None
            # First segment: digits, signed digits, or omitted.
            if self.src[self.pos] == ":":
                # Omitted start.
                self.pos += 1
            else:
                is_negative = False
                if self.src[self.pos] == "-":
                    is_negative = True
                    self.pos += 1
                int_start = self.pos
                while not self._eof() and self.src[self.pos].isdigit():
                    self.pos += 1
                if self.pos == int_start:
                    self.pos = save
                    return None
                start_val = int(self.src[int_start:self.pos])
                if is_negative:
                    start_val = -start_val
                if self.src[self.pos] != ":":
                    # Not a slice. Could be plain index ``[3]`` OR the
                    # comma of a union ``[3, 5]``. Return None and leave
                    # pos at the comma so the union caller can tell.
                    self._skip_ws()
                    self._expect("]") if consume_close else None
                    return Index(start_val)
                self.pos += 1  # consume ':'
            # Slice mode (start consumed or omitted; ':' seen).
            self._skip_ws()
            end_val: Any = None
            if self.src[self.pos] == ":":
                # Omitted end - skip the ':' and go straight to step.
                self.pos += 1
                self._skip_ws()
            else:
                is_negative = False
                if self.src[self.pos] == "-":
                    is_negative = True
                    self.pos += 1
                int_start = self.pos
                while not self._eof() and self.src[self.pos].isdigit():
                    self.pos += 1
                if self.pos > int_start:
                    end_val = int(self.src[int_start:self.pos])
                    if is_negative:
                        end_val = -end_val
                self._skip_ws()
                if self._peek() != ":":
                    self._expect("]") if consume_close else None
                    return Slice(start_val, end_val, 1)
                self.pos += 1  # consume second ':'
                self._skip_ws()
            # Step segment (possibly omitted).
            self._skip_ws()
            step_val: int = 1
            if self.src[self.pos] != "]":
                is_negative = False
                if self.src[self.pos] == "-":
                    is_negative = True
                    self.pos += 1
                int_start = self.pos
                while not self._eof() and self.src[self.pos].isdigit():
                    self.pos += 1
                if self.pos > int_start:
                    step_val = int(self.src[int_start:self.pos])
                    if is_negative:
                        step_val = -step_val
            self._skip_ws()
            self._expect("]") if consume_close else None
            if step_val == 0:
                raise JSONPathError("Slice step cannot be zero", self.src, save)
            return Slice(start_val, end_val, step_val)
        except JSONPathError:
            # Restore to a position that lets the caller distinguish
            # "we just consumed an int and the bracket continued" from
            # "the input is malformed". If we made it past at least
            # one int digit, leave pos at the int end; otherwise
            # restore fully.
            if self.pos > save + 1 and self.src[self.pos - 1].isdigit():
                # Find the start of the int we parsed and place pos
                # at the first non-int char after it. If that char is
                # ",", the caller will detect a union.
                end = self.pos
                p = end - 1
                while p > save and (self.src[p].isdigit() or self.src[p] == "-"):
                    p -= 1
                self.pos = end  # leave pos at the int end (e.g. after `0`)
            else:
                self.pos = save
            return None

    # -- Filter expressions ----------------------------------------------------    # -- Filter expressions ----------------------------------------------------

    def _parse_logical(self) -> FilterExpr:
        left = self._parse_comparison()
        self._skip_ws()
        if self.src[self.pos:self.pos + 2] == "&&":
            self.pos += 2
            right = self._parse_logical()
            return LogicalOp("&&", left, right)
        if self.src[self.pos:self.pos + 2] == "||":
            self.pos += 2
            right = self._parse_logical()
            return LogicalOp("||", left, right)
        return left

    def _parse_comparison(self) -> FilterExpr:
        left = self._parse_filter_primary()
        self._skip_ws()
        ops = ("==", "!=", "<=", ">=", "=~", "<", ">")
        matched = None
        matched_len = 0
        for op in ops:
            if self.src[self.pos:self.pos + len(op)] == op:
                matched = op
                matched_len = len(op)
                break
        if matched is None:
            return left
        self.pos += matched_len
        right = self._parse_filter_primary()
        return ComparisonOp(matched, left, right)

    def _parse_filter_primary(self) -> FilterExpr:
        self._skip_ws()
        c = self._peek()
        if c == "(":
            # Grouping in filter expressions: ``(expr)`` returns expr.
            self.pos += 1
            self._skip_ws()
            inner = self._parse_logical()
            self._skip_ws()
            self._expect(")")
            return inner
        if c == "@":
            self.pos += 1
            segments: List[Selector] = []
            while not self._eof() and self.src[self.pos] in (".", "["):
                seg = self._parse_segment()
                if seg is None:
                    break
                segments.append(seg)
            return Path(segments)
        if c in ("'", '"', "-") or c.isdigit():
            lit, end = _parse_literal(self.src, self.pos)
            self.pos = end
            return lit
        if self.src[self.pos:self.pos + 4] == "true":
            self.pos += 4
            return Literal(True)
        if self.src[self.pos:self.pos + 5] == "false":
            self.pos += 5
            return Literal(False)
        if self.src[self.pos:self.pos + 4] == "null":
            self.pos += 4
            return Literal(None)
        raise JSONPathError(f"Unexpected character in filter: {c!r}", self.src, self.pos)


def parse(expression: str) -> List[Selector]:
    """Parse a JSONPath expression into a list of Selectors.

    Returns ``[Root(), ...selectors]``. Raises ``JSONPathError`` on
    malformed input.
    """
    return _Parser(expression).parse()


# ── Evaluator ─────────────────────────────────────────────────────────────────


def _apply_to_list(value: Any, func) -> List[Any]:
    """Apply ``func(value)`` to each element of a list, else to ``value`` itself."""
    if isinstance(value, list):
        return [r for v in value for r in func(v)]
    return func(value)


def _evaluate_path(node: Any, segments: List[Selector]) -> List[Any]:
    """Run a sequence of Selectors against a node, returning a list of matches."""
    current: List[Any] = [node]
    for sel in segments:
        nxt: List[Any] = []
        for c in current:
            nxt.extend(_evaluate_selector(c, sel))
        current = nxt
    return current


def _evaluate_selector(node: Any, sel: Selector) -> List[Any]:
    """Run a single Selector against a single node, returning its matches."""
    if isinstance(sel, Root):
        return [node]
    if isinstance(sel, Name):
        if sel.name == "*":
            if isinstance(node, dict):
                return list(node.values())
            if isinstance(node, list):
                return list(node)
            return []
        if isinstance(node, dict):
            v = node.get(sel.name, _MISSING)
            return [] if v is _MISSING else [v]
        return []
    if isinstance(sel, Index):
        if not isinstance(node, list):
            return []
        idx = sel.index
        if idx < 0:
            idx += len(node)
        if 0 <= idx < len(node):
            return [node[idx]]
        return []
    if isinstance(sel, Slice):
        if not isinstance(node, list):
            return []
        return [_slice_get(node, sel.start, sel.end, sel.step)]
    if isinstance(sel, Wildcard):
        if isinstance(node, dict):
            return list(node.values())
        if isinstance(node, list):
            return list(node)
        return []
    if isinstance(sel, Descendant):
        return _descend(node, sel.member)
    if isinstance(sel, Union):
        out: List[Any] = []
        for sub in sel.selectors:
            out.extend(_evaluate_selector(node, sub))
        return out
    if isinstance(sel, Filter):
        if not isinstance(node, list):
            return []
        return [v for v in node if _eval_filter(sel.expr, v)]
    raise JSONPathError(f"Unknown selector type: {type(sel).__name__}")


class _Missing:
    pass


_MISSING = _Missing()


def _slice_get(arr: List[Any], start: Any, end: Any, step: int) -> List[Any]:
    """Apply a JSONPath slice to a list, returning a new list."""
    n = len(arr)
    # Normalize negative indices and clamp.
    def norm(v: Any, default: int) -> int:
        if v is None:
            return default
        if v < 0:
            v = v + n
        if step > 0:
            return max(0, min(n, v))
        return max(-1, min(n - 1, v))

    if step == 0:
        raise JSONPathError("Slice step cannot be zero")
    s = norm(start, 0 if step > 0 else n - 1)
    e = norm(end, n if step > 0 else -1)
    return list(arr[s:e:step])


def _descend(node: Any, member: Optional[str]) -> List[Any]:
    """Recursive descent through objects and arrays."""
    out: List[Any] = []
    stack = [node]
    seen = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, dict):
            for k, v in cur.items():
                if member is None or k == member:
                    out.append(v)
                stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if member is None:
                    out.append(v)
                stack.append(v)
    return out


def _eval_filter(expr: FilterExpr, current: Any) -> bool:
    """Evaluate a filter expression against ``current`` (``@``)."""
    if isinstance(expr, LogicalOp):
        if expr.op == "&&":
            return _eval_filter(expr.left, current) and _eval_filter(expr.right, current)
        return _eval_filter(expr.left, current) or _eval_filter(expr.right, current)
    if isinstance(expr, ComparisonOp):
        left = _eval_filter_expr(expr.left, current)
        right = _eval_filter_expr(expr.right, current)
        # ``=~`` is regex match (string only). All others compare with operators.
        if expr.op == "=~":
            if not isinstance(left, str) or not isinstance(right, str):
                return False
            try:
                return re.search(right, left) is not None
            except re.error:
                return False
        if expr.op == "==":
            return left == right
        if expr.op == "!=":
            return left != right
        if expr.op == "<":
            try:
                return left < right  # type: ignore[operator]
            except TypeError:
                return False
        if expr.op == "<=":
            try:
                return left <= right  # type: ignore[operator]
            except TypeError:
                return False
        if expr.op == ">":
            try:
                return left > right  # type: ignore[operator]
            except TypeError:
                return False
        if expr.op == ">=":
            try:
                return left >= right  # type: ignore[operator]
            except TypeError:
                return False
    if isinstance(expr, Literal):
        return bool(expr.value) if isinstance(expr.value, bool) else False
    raise JSONPathError(f"Unknown filter expression type: {type(expr).__name__}")


def _eval_filter_expr(expr: FilterExpr, current: Any) -> Any:
    """Like ``_eval_filter`` but returns the value (used for comparisons)."""
    if isinstance(expr, LogicalOp):
        return _eval_filter(expr, current)
    if isinstance(expr, Path):
        results = _evaluate_path(current, expr.segments)
        if len(results) == 1:
            return results[0]
        if not results:
            return None
        return results
    if isinstance(expr, Literal):
        return expr.value
    raise JSONPathError(f"Cannot evaluate as value: {type(expr).__name__}")


# ── Public API ────────────────────────────────────────────────────────────────


def find(expression: str, document: Any) -> List[Any]:
    """Run ``expression`` against ``document`` and return all matches as a list."""
    segments = parse(expression)
    return _evaluate_path(document, segments)


def find_one(expression: str, document: Any, default: Any = None) -> Any:
    """Run ``expression`` against ``document`` and return the first match.

    Returns ``default`` (default ``None``) when nothing matches.
    """
    results = find(expression, document)
    return results[0] if results else default
