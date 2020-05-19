"""Test for example.configs"""

import deepr as dpr
import deepr.example


PATH_CONFIG = dpr.io.Path(deepr.example.__file__).parent / "configs"


def test_example_configs():
    """Test for example.configs"""
    config = dpr.io.read_json(PATH_CONFIG / "config.json")
    macros = dpr.io.read_json(PATH_CONFIG / "macros.json")
    parsed = dpr.parse_config(config, macros)
    dpr.from_config(parsed)
