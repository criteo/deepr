"""Tests for config.deepr_experimental"""

from dataclasses import dataclass

import pytest

import deepr


@dataclass
class A:
    x: int = 1
    y: float = 1.0
    z: bool = True


@dataclass
class B:
    a: A
    b: str = "b"


class C(A):
    def __init__(self, b: str = "b", **kwargs):
        super().__init__(**kwargs)
        self.b = b


@pytest.mark.parametrize(
    "obj, cfg",
    [
        (
            B(A()),
            {
                "type": "test_config_experimental.B",
                "a": {"type": "test_config_experimental.A", "x": 1, "y": 1.0, "z": True},
                "b": "b",
            },
        ),
        (C(), {"type": "test_config_experimental.C", "x": 1, "y": 1.0, "z": True, "b": "b"}),
    ],
)
def test_to_config(obj, cfg):
    """Test to_config"""
    assert deepr.config.experimental.to_config(obj) == cfg


@pytest.mark.parametrize(
    "config, params, expected",
    [
        ({"x": 1}, ["x"], {"x": "$params:x"}),
        ([{"x": 1}], ["x"], [{"x": "$params:x"}]),
        (({"x": 1},), ["x"], ({"x": "$params:x"},)),
        ({"x": {"y": 1}}, ["y"], {"x": {"y": "$params:y"}}),
        ({"x": "$other:x"}, ["x"], {"x": "$params:x"}),
        ([{"x": 1}], ["y"], None),
    ],
)
def test_add_macro_params(config, params, expected):
    if expected is not None:
        assert deepr.config.experimental.add_macro_params(config, macro="params", params=params) == expected
    else:
        with pytest.raises(ValueError):
            deepr.config.experimental.add_macro_params(config, macro="params", params=params)


@pytest.mark.parametrize(
    "item, values, expected",
    [
        ({"x": 1}, {"x": "y"}, {"x": "y"}),
        ([{"x": 1}], {"x": "y"}, [{"x": "y"}]),
        (({"x": 1},), {"x": "y"}, ({"x": "y"},)),
        ({"a": {"x": 1}}, {"x": "y"}, {"a": {"x": "y"}}),
        ({"x": "y"}, {"y": "1"}, {"x": "y"}),
        ({"x": (1, 2)}, {"x": (2, 3)}, None),
        ({"x": {}}, {"x": {}}, None),
        ({"x": []}, {"x": []}, None),
    ],
)
def test_replace_values(item, values, expected):
    if expected is not None:
        assert deepr.config.experimental.replace_values(item, values=values) == expected
    else:
        with pytest.raises(ValueError):
            deepr.config.experimental.replace_values(item, values=values)


@pytest.mark.parametrize(
    "item, keys, expected",
    [
        ({"x": 1}, ["x"], {"x": 1}),
        ([{"x": 1}], ["x"], {"x": 1}),
        (({"x": 1},), ["x"], {"x": 1}),
        ({"a": {"x": 1}}, ["x"], {"x": 1}),
        ({"x": (1, 2)}, ["x"], None),
        ({"x": {}}, ["x"], None),
        ({"x": []}, ["x"], None),
    ],
)
def test_find_values(item, keys, expected):
    if expected is not None:
        assert deepr.config.experimental.find_values(item, keys=keys) == expected
    else:
        with pytest.raises(ValueError):
            deepr.config.experimental.find_values(item, keys=keys)
