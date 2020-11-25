"""Tests for utils.datastruct"""

import pytest

import deepr


class DummyIterable:
    """Dummy Iterable"""

    def __iter__(self):
        pass

    def __eq__(self, other):
        return other.__class__ == self.__class__


@pytest.mark.parametrize(
    "items, expected",
    [
        (1, (1,)),
        ("hello", ("hello",)),
        (DummyIterable(), (DummyIterable(),)),
        ((x for x in range(2)), (0, 1)),
        ((1, 2), (1, 2)),
        ((1, (2, 3)), (1, 2, 3)),
        (((x for x in range(2)), 2), (0, 1, 2)),
    ],
)
def test_utils_to_flat_tuple(items, expected):
    """Test to_float_tuple function"""
    assert deepr.utils.to_flat_tuple(items) == expected
