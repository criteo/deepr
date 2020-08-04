# pylint: disable=redefined-outer-name,missing-docstring
"""Tests for deepr CLI"""

import subprocess

import pytest

import deepr as dpr


@pytest.fixture
def path_config_no_macro(tmpdir_factory):
    path_data = tmpdir_factory.mktemp("data")
    path = path_data.join("config.jsonnet")
    config = {
        "type": "deepr.examples.multiply.jobs.Build",
        "path_dataset": f"{path_data}/data.tfrecord",
        "num_examples": 10,
    }
    dpr.io.write_json(config, path)
    return path


@pytest.fixture
def path_config(tmpdir_factory):
    path_data = tmpdir_factory.mktemp("data")
    path = path_data.join("config.jsonnet")
    config = {
        "type": "deepr.examples.multiply.jobs.Build",
        "path_dataset": f"{path_data}/data.tfrecord",
        "num_examples": "$params:num_examples",
    }
    dpr.io.write_json(config, path)
    return path


@pytest.fixture
def path_macros(tmpdir_factory):
    path = tmpdir_factory.mktemp("data").join("macros.jsonnet")
    macros = {"params": {"num_examples": 10}}
    dpr.io.write_json(macros, path)
    return path


def test_cli_run(path_config, path_macros):
    subprocess.check_call(["deepr", "run", path_config, path_macros])
    path_data = dpr.io.Path(path_config).parent
    assert dpr.io.Path(path_data, "data.tfrecord").is_file()


def test_cli_from_config_and_macros(path_config, path_macros):
    subprocess.check_call(["deepr", "from_config_and_macros", path_config, path_macros])


def test_cli_from_config(path_config_no_macro):
    subprocess.check_call(["deepr", "from_config", path_config_no_macro, "-", "run"])
    path_data = dpr.io.Path(path_config_no_macro).parent
    assert dpr.io.Path(path_data, "data.tfrecord").is_file()


def test_cli_add_macro(path_config_no_macro, path_macros):
    subprocess.check_call(["deepr", "add_macro", path_config_no_macro, path_macros, "num_examples"])
    with dpr.io.Path(path_config_no_macro).open() as file:
        assert "$params:num_examples" in file.read()
