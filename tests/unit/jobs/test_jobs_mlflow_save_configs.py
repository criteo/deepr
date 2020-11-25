# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Tests for prepros.mlflow_initializer"""

import pytest

import deepr


@pytest.mark.parametrize(
    "include_keys, params, expected",
    [
        (None, {"foo": {"bar": "baz"}}, {"foo.bar": "baz"}),
        (None, {"type": "foo.Foo"}, {"type": "Foo"}),
        (None, {"foo": [{"type": "foo.Bar", "baz": "baz"}]}, {"foo.Bar.baz": "baz"}),
        (("bar", "x"), {"foo": {"type": "foo.Foo", "bar": {"x": 1, "y": 1}}}, {"bar.x": 1, "bar.y": 1, "x": 1}),
    ],
)
def test_jobs_mlflow_formatter(include_keys, params, expected):
    """Test MLFlowFormatter"""
    formatter = deepr.jobs.MLFlowFormatter(include_keys)
    assert expected == formatter(params)
