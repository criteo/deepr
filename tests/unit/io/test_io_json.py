"""Test for io.json"""

import pytest
import json

import deepr


@pytest.mark.parametrize(
    "data, is_json", [("viewfs://path", False), ("config.json", False), (json.dumps({"x": 1}), True)]
)
def test_io_is_json(data, is_json):
    """Test for io.is_json"""
    assert deepr.io.is_json(data) == is_json
