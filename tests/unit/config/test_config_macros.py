"""Tests for config.macros"""

import pytest

import deepr


@pytest.mark.parametrize(
    "item, macros, expected",
    [
        (None, None, None),
        ("$macro:name", None, "$macro:name"),
        ("$macro:name", {"macro": {"other": "value"}}, "$macro:name"),
        ("$macro:name", {"other": {"name": "value"}}, "$macro:name"),
        (["$macro:name"], {"macro": {"name": "value"}}, ["value"]),
        (("$macro:name",), {"macro": {"name": "value"}}, ("value",)),
        ({"key": "$macro:name"}, {"macro": {"name": "value"}}, {"key": "value"}),
    ],
)
def test_config_fill_macros(item, macros, expected):
    assert deepr.config.fill_macros(item, macros) == expected


@pytest.mark.parametrize(
    "item, macro, expected",
    [
        ({"x": "$params:x"}, "params", ["x"]),
        ([{"x": "$params:x"}], "params", ["x"]),
        (({"x": "$params:x"},), "params", ["x"]),
        ({"a": {"x": "$params:x"}}, "params", ["x"]),
        ({"x": "$params:x", "y": "$prod:y"}, "params", ["x"]),
    ],
)
def test_find_macro_params(item, macro, expected):
    assert set(deepr.config.find_macro_params(item, macro)) == set(expected)


@pytest.mark.parametrize(
    "item, expected", [("param", False), ("macro:param", False), ("$macro:param", True), ("$macro:param:other", None)]
)
def test_ismacro(item, expected):
    if expected is not None:
        assert deepr.config.ismacro(item) == expected
    else:
        with pytest.raises(ValueError):
            deepr.config.ismacro(item)


def test_get_macro_and_param():
    assert deepr.config.get_macro_and_param("$macro:param") == ("macro", "param")


@pytest.mark.parametrize(
    "item, error",
    [
        (None, False),
        ("macro", False),
        ([1, 2], False),
        ((1, 2), False),
        ({1: 2}, False),
        ("$wrong", True),
        ("$macro:name", True),
        (["$macro:name"], True),
        (("$macro:name",), True),
        ({"key": "$macro:name"}, True),
    ],
)
def test_config_assert_no_macros(item, error: bool):
    if error:
        with pytest.raises(ValueError):
            deepr.config.assert_no_macros(item)
    else:
        deepr.config.assert_no_macros(item)


@pytest.mark.parametrize(
    "macros, expected",
    [({"1": {"x": 1}, "2": {"x": "$1:x"}}, ["1", "2"]), ({"1": {"x": "$2:x"}, "2": {"x": "$1:x"}}, None)],
)
def test_macros_eval_order(macros, expected):
    if expected is not None:
        assert deepr.config.macros_eval_order(macros) == expected
    else:
        with pytest.raises(ValueError):
            deepr.config.macros_eval_order(macros)
