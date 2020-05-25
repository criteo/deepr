"""Test for example.configs"""

import logging

import deepr as dpr
import deepr.example


logging.basicConfig(level=logging.INFO)


PATH_CONFIG = dpr.io.Path(deepr.example.__file__).parent / "configs"


def test_example_configs(tmpdir):
    """Test for example.configs"""
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    config = dpr.io.read_json(PATH_CONFIG / "config.json")
    macros = dpr.io.read_json(PATH_CONFIG / "macros.json")
    macros["paths"]["path_model"] = path_model
    macros["paths"]["path_dataset"] = path_dataset
    parsed = dpr.parse_config(config, macros)
    job = dpr.from_config(parsed)
    job.run()
