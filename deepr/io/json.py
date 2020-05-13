"""Json IO"""

from typing import Union, Dict
import logging
import json

import _jsonnet

from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


def read_json(path: Union[str, Path]) -> Dict:
    """Read json or jsonnet file into dictionary"""
    if Path(path).suffix == ".jsonnet":
        LOGGER.info(f"Parsing jsonnet file '{path}'")
        json_str = _jsonnet.evaluate_file(str(path))
        data = json.loads(json_str)
    else:
        with Path(path).open() as file:
            data = json.load(file)
    return data


def write_json(data: Dict, path: Union[str, Path]):
    """Write data to path"""
    with Path(path).open("w") as file:
        json.dump(data, file, indent=4)
