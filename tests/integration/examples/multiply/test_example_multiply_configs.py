"""Test for examples.multiply.configs"""

import logging

import deepr as dpr
import deepr.examples.multiply


logging.basicConfig(level=logging.INFO)


PATH_CONFIG = dpr.io.Path(deepr.examples.multiply.__file__).parent / "configs"


def test_example_multiply_configs(tmpdir):
    """Test for examples.multiply.configs"""
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    config = dpr.io.read_json(PATH_CONFIG / "config.json")
    macros = dpr.io.read_json(PATH_CONFIG / "macros.json")
    macros["paths"]["path_model"] = path_model
    macros["paths"]["path_dataset"] = path_dataset
    parsed = dpr.parse_config(config, macros)
    job = dpr.from_config(parsed)
    job.run()
