"""Test for examples.multiply.configs"""

import logging

import deepr
import deepr.examples.multiply


logging.basicConfig(level=logging.INFO)


PATH_CONFIG = deepr.io.Path(deepr.examples.multiply.__file__).parent / "configs"


def test_example_multiply_configs(tmpdir):
    """Test for examples.multiply.configs"""
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    config = deepr.io.read_json(PATH_CONFIG / "config.json")
    macros = deepr.io.read_json(PATH_CONFIG / "macros.json")
    macros["paths"]["path_model"] = path_model
    macros["paths"]["path_dataset"] = path_dataset
    parsed = deepr.parse_config(config, macros)
    job = deepr.from_config(parsed)
    job.run()
