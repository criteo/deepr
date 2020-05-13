"""Tests for config.deepr_experimental"""

from dataclasses import dataclass

import pytest

import deepr as dpr


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
    assert dpr.config.experimental.to_config(obj) == cfg
