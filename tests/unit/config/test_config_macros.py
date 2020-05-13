"""Tests for config.macros"""

import pytest

import deepr as dpr


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
    assert dpr.config.fill_macros(item, macros) == expected


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
            dpr.config.assert_no_macros(item)
    else:
        dpr.config.assert_no_macros(item)
