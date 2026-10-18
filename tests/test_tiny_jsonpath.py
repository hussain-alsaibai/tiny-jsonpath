"""
tiny-jsonpath tests

Covers RFC 9535 syntax subset:
  * Root selector $
  * Member access: .name and ['name']
  * Array index: [i] (positive and negative)
  * Array slice: [start:end:step]
  * Wildcard: [*] and .*
  * Recursive descent
  * Union
  * Filter expressions: comparisons, logical ops, regex match
"""
from __future__ import annotations

import unittest

from tiny_jsonpath import (
    find, find_one, parse, JSONPathError,
    Root, Name, Index, Slice, Wildcard, Descendant, Union, Filter,
    Path, Literal, ComparisonOp, LogicalOp,
)


SAMPLE = {
    "store": {
        "book": [
            {"title": "Sayings of the Century", "price": 8.95, "category": "reference",
             "author": {"name": "Nigel Rees"}},
            {"title": "Sword of Honour", "price": 12.99, "category": "fiction",
             "author": {"name": "Evelyn Waugh"}},
            {"title": "Moby Dick", "price": 8.99, "category": "fiction",
             "isbn": "0-553-21311-3",
             "author": {"name": "Herman Melville"}},
            {"title": "The Lord of the Rings", "price": 22.99, "category": "fiction",
             "isbn": "0-395-19395-8",
             "author": {"name": "J. R. R. Tolkien"}},
        ],
        "bicycle": {"color": "red", "price": 19.95},
    },
    "expensive": 10,
    "tags": ["a", "b", "c"],
}


class TestRootAndName(unittest.TestCase):

    def test_root_returns_document(self):
        self.assertEqual(find("$", SAMPLE), [SAMPLE])

    def test_dot_name(self):
        self.assertEqual(find("$.expensive", SAMPLE), [10])

    def test_bracket_quoted_name(self):
        self.assertEqual(find("$['expensive']", SAMPLE), [10])

    def test_double_quoted_bracket_name(self):
        self.assertEqual(find('$["expensive"]', SAMPLE), [10])

    def test_nested_path(self):
        self.assertEqual(find("$.store.bicycle.color", SAMPLE), ["red"])

    def test_missing_key_returns_empty(self):
        self.assertEqual(find("$.nonexistent", SAMPLE), [])


class TestArrayIndex(unittest.TestCase):

    def test_positive_index(self):
        self.assertEqual(find("$.tags[0]", SAMPLE), ["a"])

    def test_negative_index(self):
        self.assertEqual(find("$.tags[-1]", SAMPLE), ["c"])

    def test_out_of_bounds_returns_empty(self):
        self.assertEqual(find("$.tags[99]", SAMPLE), [])

    def test_index_into_nested_array(self):
        self.assertEqual(find("$.store.book[1].title", SAMPLE), ["Sword of Honour"])


class TestArraySlice(unittest.TestCase):

    def test_full_slice(self):
        self.assertEqual(find("$.tags[:]", SAMPLE), [["a", "b", "c"]])

    def test_first_two(self):
        self.assertEqual(find("$.tags[0:2]", SAMPLE), [["a", "b"]])

    def test_skip_first(self):
        self.assertEqual(find("$.tags[1:]", SAMPLE), [["b", "c"]])

    def test_negative_index_in_slice(self):
        self.assertEqual(find("$.tags[-2:]", SAMPLE), [["b", "c"]])

    def test_step(self):
        # Even-indexed tags: a, c
        self.assertEqual(find("$.tags[::2]", SAMPLE), [["a", "c"]])

    def test_zero_step_raises(self):
        with self.assertRaises(JSONPathError):
            parse("$.tags[::0]")


class TestWildcard(unittest.TestCase):

    def test_bracket_wildcard(self):
        self.assertEqual(
            sorted(find("$.tags[*]", SAMPLE)),
            ["a", "b", "c"],
        )

    def test_bracket_wildcard_then_member(self):
        titles = sorted(find("$.store.book[*].title", SAMPLE))
        self.assertEqual(titles, [
            "Moby Dick",
            "Sayings of the Century",
            "Sword of Honour",
            "The Lord of the Rings",
        ])

    def test_dot_wildcard(self):
        # .store.* returns all values inside store (objects + arrays).
        results = find("$.store.*", SAMPLE)
        self.assertIn(SAMPLE["store"]["book"], results)
        self.assertIn(SAMPLE["store"]["bicycle"], results)


class TestRecursiveDescent(unittest.TestCase):

    def test_descendant_member(self):
        # Every ``price`` anywhere in the document.
        prices = sorted(find("$..price", SAMPLE))
        self.assertEqual(prices, [8.95, 8.99, 12.99, 19.95, 22.99])

    def test_descendant_author_name(self):
        names = sorted(find("$..author.name", SAMPLE))
        self.assertEqual(names, [
            "Evelyn Waugh", "Herman Melville", "J. R. R. Tolkien", "Nigel Rees",
        ])

    def test_descendant_bare(self):
        # $..title matches every ``title`` value.
        result = find("$..title", SAMPLE)
        self.assertEqual(len(result), 4)


class TestUnion(unittest.TestCase):

    def test_union_of_names(self):
        # ['expensive', 'tags'] selects both members.
        result = sorted(find("$['expensive', 'tags']", SAMPLE), key=repr)
        self.assertEqual(len(result), 2)
        self.assertIn(10, result)
        self.assertEqual(sorted([v for v in result if isinstance(v, list)]), [["a", "b", "c"]])

    def test_union_of_indices(self):
        result = sorted(find("$.tags[0, 2]", SAMPLE))
        self.assertEqual(result, ["a", "c"])


class TestFilterExpressions(unittest.TestCase):

    def test_filter_price_under_ten(self):
        cheap = find("$.store.book[?(@.price < 10)]", SAMPLE)
        titles = sorted(b["title"] for b in cheap)
        self.assertEqual(titles, ["Moby Dick", "Sayings of the Century"])

    def test_filter_category_eq(self):
        fiction = find("$.store.book[?(@.category == 'fiction')]", SAMPLE)
        self.assertEqual(len(fiction), 3)

    def test_filter_and(self):
        result = find(
            "$.store.book[?(@.category == 'fiction' && @.price < 10)]",
            SAMPLE,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Moby Dick")

    def test_filter_or(self):
        result = find(
            "$.store.book[?(@.price < 9 || @.price > 20)]",
            SAMPLE,
        )
        titles = sorted(b["title"] for b in result)
        self.assertEqual(titles, ["Moby Dick", "Sayings of the Century",
                                   "The Lord of the Rings"])

    def test_filter_no_parens(self):
        # Optional parens around the filter expression.
        result = find("$.store.book[?@.category == 'fiction']", SAMPLE)
        self.assertEqual(len(result), 3)

    def test_filter_regex_match(self):
        result = find('$..book[?(@.title =~ "Moby.*")]', SAMPLE)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Moby Dick")

    def test_filter_not_equal(self):
        result = find("$.store.book[?(@.isbn != null)]", SAMPLE)
        self.assertEqual(len(result), 2)

    def test_filter_nested_member(self):
        # Compare against a nested member.
        result = find("$.store.book[?(@.author.name == 'Nigel Rees')]", SAMPLE)
        self.assertEqual(len(result), 1)


class TestFindOne(unittest.TestCase):

    def test_returns_first_match(self):
        self.assertEqual(
            find_one("$.store.book[0].title", SAMPLE),
            "Sayings of the Century",
        )

    def test_returns_default_when_no_match(self):
        self.assertIsNone(find_one("$.nonexistent", SAMPLE))
        self.assertEqual(find_one("$.nonexistent", SAMPLE, default="missing"),
                         "missing")


class TestParseAST(unittest.TestCase):
    """The AST nodes match the spec."""

    def test_simple_chain(self):
        ast = parse("$.store.book[0].title")
        self.assertIsInstance(ast[0], Root)
        self.assertIsInstance(ast[1], Name)
        self.assertEqual(ast[1].name, "store")
        self.assertIsInstance(ast[2], Name)
        self.assertEqual(ast[2].name, "book")
        self.assertIsInstance(ast[3], Index)
        self.assertEqual(ast[3].index, 0)
        self.assertIsInstance(ast[4], Name)
        self.assertEqual(ast[4].name, "title")

    def test_slice_ast(self):
        ast = parse("$.tags[1:5:2]")
        # The third selector is the slice; ast[1] is the .tags Name.
        self.assertIsInstance(ast[2], Slice)
        self.assertEqual(ast[2].start, 1)
        self.assertEqual(ast[2].end, 5)
        self.assertEqual(ast[2].step, 2)

    def test_filter_ast(self):
        ast = parse("$.x[?(@.y > 5)]")
        # ast[1] is the .x Name, ast[2] is the Filter.
        self.assertIsInstance(ast[2], Filter)
        self.assertIsInstance(ast[2].expr, ComparisonOp)
        self.assertEqual(ast[2].expr.op, ">")


class TestErrors(unittest.TestCase):

    def test_missing_root(self):
        with self.assertRaises(JSONPathError):
            parse("foo.bar")

    def test_bad_bracket(self):
        with self.assertRaises(JSONPathError):
            parse("$[")

    def test_zero_step(self):
        with self.assertRaises(JSONPathError):
            parse("$.x[::0]")

    def test_unterminated_string(self):
        with self.assertRaises(JSONPathError):
            parse("$['unterminated")

    def test_error_includes_position(self):
        try:
            parse("$.foo[bar")
        except JSONPathError as e:
            # Position 6 is the 'b' that starts the bad union element.
            self.assertEqual(e.position, 6)
            self.assertIn("line", str(e))


class TestEdgeCases(unittest.TestCase):

    def test_empty_document(self):
        self.assertEqual(find("$", {}), [{}])
        self.assertEqual(find("$.x", {}), [])

    def test_root_only(self):
        self.assertEqual(find("$", [1, 2, 3]), [[1, 2, 3]])

    def test_wildcard_on_non_object(self):
        # [*] on a non-array, non-object returns nothing.
        self.assertEqual(find("$.expensive[*]", SAMPLE), [])

    def test_index_on_non_array(self):
        self.assertEqual(find("$.expensive[0]", SAMPLE), [])

    def test_member_on_non_object(self):
        self.assertEqual(find("$.expensive.foo", SAMPLE), [])


class TestDeepNested(unittest.TestCase):
    """Stress recursive descent + filters in deeply nested structures."""

    def test_five_levels_deep(self):
        doc = {"a": {"b": {"c": {"d": {"e": [1, 2, 3, 4, 5]}}}}}
        self.assertEqual(find("$.a.b.c.d.e[2]", doc), [3])
        self.assertEqual(sorted(find("$..e[*]", doc)), [1, 2, 3, 4, 5])

    def test_filter_with_arithmetic_lookalike(self):
        # Make sure >= and <= parse correctly and don't get confused with > and <.
        doc = [{"v": 1}, {"v": 2}, {"v": 3}]
        self.assertEqual(len(find("$[?(@.v >= 2)]", doc)), 2)
        self.assertEqual(len(find("$[?(@.v <= 2)]", doc)), 2)
        self.assertEqual(len(find("$[?(@.v > 2)]", doc)), 1)
        self.assertEqual(len(find("$[?(@.v < 2)]", doc)), 1)


if __name__ == "__main__":
    unittest.main()
