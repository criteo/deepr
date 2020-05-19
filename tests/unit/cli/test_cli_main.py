# pylint: disable=redefined-outer-name,missing-docstring
"""Tests for config.base"""

import subprocess
import json

import pytest

import deepr as dpr


@pytest.fixture
def path_config(tmpdir_factory):
    path_data = tmpdir_factory.mktemp("data")
    path = path_data.join("config.jsonnet")
    config = {
        "type": "deepr.example.jobs.BuildDataset",
        "path_dataset": f"{path_data}/data.tfrecord",
        "num_examples": 10,
    }
    with dpr.io.Path(path).open("w") as file:
        json.dump(config, file)
    return path


def test_cli_from_config_and_macros(path_config):
    subprocess.check_call(["deepr", "from_config_and_macros", path_config])


def test_cli_run(path_config):
    subprocess.check_call(["deepr", "run", path_config])
    path_data = dpr.io.Path(path_config).parent
    assert dpr.io.Path(path_data, "data.tfrecord").is_file()
