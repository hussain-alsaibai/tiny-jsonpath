# tiny-jsonpath

**Zero-dependency, RFC 9535 (JSONPath 1.0) parser and evaluator for Python.**

```bash
pip install tiny-jsonpath
```

---

## Why

JSONPath is the standard way to query JSON, finalized by the IETF as RFC 9535
in 2024. Most existing Python implementations either depend on a non-stdlib
JSON library, ship a several-thousand-line parser, or only support the older
Stefan Goessner syntax.

`tiny-jsonpath` is the 600-line stdlib-only implementation of RFC 9535. No
dependencies, no compile step, runs anywhere CPython runs. Pure-Python so the
AST nodes are inspectable from your own tools.

---

## Quick start

```python
from tiny_jsonpath import find, find_one

doc = {
    "store": {
        "book": [
            {"title": "Moby Dick",       "price": 8.99,  "category": "fiction"},
            {"title": "Sword of Honour", "price": 12.99, "category": "fiction"},
        ],
    },
}

find("$.store.book[*].title", doc)
# ['Moby Dick', 'Sword of Honour']

find_one("$.store.book[?(@.price < 10)].title", doc)
# 'Moby Dick'

find("$..price", doc)
# [8.99, 12.99]   (recursive descent)
```

---

## Supported features (RFC 9535 subset)

- **Root** `$`
- **Member access** `.name` and `['name']`
- **Array index** `[i]`, negative indices
- **Array slice** `[start:end:step]` (all forms: `[a:b:c]`, `[:b:c]`, `[a::c]`, `[a:b]`, `[:b]`, `[a:]`, `[:]`, `[::c]`)
- **Wildcard** `[*]` and `.*`
- **Recursive descent** `..name`
- **Union** `[a, b, c]`
- **Filter expressions** `[?(@.x > 5)]` with `==` / `!=` / `<` / `<=` / `>` / `>=` / `=~`, `&&`, `||`, optional parens, literals, paths
- **find_one()** with default value

Out of scope (deliberately):
- Script expressions (the original Goessner extension)
- Function calls inside filters (`length()`, `match()`, etc.)

---

## Errors

All parse / evaluation errors raise `JSONPathError` with a `position`
attribute and a one-line caret-pointed excerpt:

```python
from tiny_jsonpath import find, JSONPathError
try:
    find("$.foo[bar", {})
except JSONPathError as e:
    print(e)            # Unexpected character in union (line 1, col 7)
                         # $.foo[bar
                         #       ^
    print(e.position)   # 6
```

---

## License

MIT
