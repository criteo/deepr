"""Tests for config.base"""

from typing import Any
from dataclasses import dataclass

import pytest

import deepr


@dataclass
class A:
    x: Any


class B:
    def __init__(self, *args):
        self.args = args

    def __eq__(self, other):
        return type(self) is type(other) and self.args == other.args


@dataclass
class C:
    x: Any
    y: Any


@pytest.mark.parametrize(
    "config, macros, expected",
    [
        (None, None, None),
        ({"type": "test_config_base.A", "x": "$macro:x"}, {"macro": {"x": 1}}, {"type": "test_config_base.A", "x": 1}),
        (
            {"type": "test_config_base.A", "x": "@self"},
            None,
            {"type": "test_config_base.A", "x": {"type": "test_config_base.A", "eval": None, "x": "@self"}},
        ),
    ],
)
def test_config_parse_config(config, macros, expected):
    assert deepr.parse_config(config, macros) == expected


@pytest.mark.parametrize(
    "item, expected",
    [
        (None, None),
        (1, 1),
        (3.14, 3.14),
        (True, True),
        (False, False),
        ("hello", "hello"),
        ((1, 2), (1, 2)),
        ([1, 2], [1, 2]),
        ({1: 2}, {1: 2}),
        ({"type": "test_config_base.A", "x": 1}, A(1)),
        ({"type": "test_config_base.B", "*": (1, 2)}, B(1, 2)),
        ({"type": "test_config_base.A", "eval": None, "x": 1}, {"type": "test_config_base.A", "x": 1}),
    ],
)
def test_config_from_config(item, expected):
    assert deepr.from_config(item) == expected


@pytest.mark.parametrize(
    "item, kwargs, expected",
    [
        ({"type": "test_config_base.C", "eval": "partial", "x": 1}, {"y": 2}, C(1, 2)),
        ({"eval": "partial", "x": 1}, {"y": 2}, {"x": 1, "y": 2}),
    ],
)
def test_config_from_config_constructor(item, kwargs, expected):
    constructor = deepr.from_config(item)
    instance = constructor(**kwargs)
    assert instance == expected
