"""Experimental utilities for config"""

import logging
import inspect
from typing import Dict, Any, List
from collections import Counter

from deepr.config.base import TYPE
from deepr.config.macros import MACRO, find_macro_params


LOGGER = logging.getLogger(__name__)


def to_config(obj):
    """Experimental utility to generate config of objects"""
    if isinstance(obj, list):
        return [to_config(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_config(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: to_config(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif obj is None:
        return obj
    else:

        def _get_params(cls):
            parameters = dict()
            try:
                for key, param in inspect.signature(cls).parameters.items():
                    if hasattr(obj, key):
                        parameters[key] = getattr(obj, key)
                    elif hasattr(obj, f"_{key}"):
                        parameters[key] = getattr(obj, f"_{key}")
                    if param.kind is param.VAR_KEYWORD:
                        parameters.update(_get_params(cls.mro()[1]))
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.warning(e)
            return parameters

        return {
            TYPE: obj.__class__.__module__ + "." + obj.__class__.__name__,
            **{key: to_config(value) for key, value in _get_params(obj.__class__).items()},
        }


def add_macro_params(config: Dict, macro: str, params: List[str]) -> Dict:
    """Add new macro parameters in config automatically.

    Parameters
    ----------
    config : Dict
        Config to modify
    macro : str
        Name of the new macro
    params : List[str]
        List of new parameter names

    Returns
    -------
    Dict
        A new config with new macro parameters

    Raises
    ------
    ValueError
        If one param has no reference in config after adding new params.
    """
    # Add new macro params to config
    old_counts = Counter(find_macro_params(config, macro))
    values = {key: f"{MACRO}{macro}:{key}" for key in params}
    config = replace_values(config, values)

    # Check that the macro params in config are consistent
    new_counts = Counter(find_macro_params(config, macro))
    for param in params:
        old_count = old_counts[param]
        new_count = new_counts[param]
        if new_count == 0:
            raise ValueError(f"Parameter '{param}': no reference in config.")
        if new_count - old_count == 0:
            LOGGER.warning(f"Parameter '{param}': NO NEW REFERENCE IN CONFIG.")
        if new_count - old_count == 1:
            LOGGER.info(f"Parameter '{param}': 1 new reference in config.")
        if new_count - old_count > 1:
            LOGGER.warning(f"Parameter '{param}': {new_count - old_count} NEW REFERENCES IN CONFIG.")

    return config


def replace_values(item, values: Dict[str, Any]):
    """Replace values for dictionary keys defined in values.

    WARNING: if a a key in item already has a value that is either a
    dict, tuple, or list, raise ValueError.

    Parameters
    ----------
    item : Any
        Config item
    values : Dict[str, Any]
        New values

    Returns
    -------
    item whose keys in values have a new value.

    Raises
    ------
    ValueError
        If one key is found whose value is a tuple, list or dict.
    """
    if isinstance(item, list):
        return [replace_values(it, values) for it in item]
    elif isinstance(item, tuple):
        return tuple(replace_values(it, values) for it in item)
    elif isinstance(item, dict):
        items = dict()
        for key, value in item.items():
            if key in values:
                if isinstance(value, (list, tuple, dict)):
                    raise ValueError(f"Cannot replace value for '{key}'. Type must be literal but got {type(value)}")
                LOGGER.info(f"Replacing value for '{key}': {value} -> {values[key]}")
                value = values[key]
            else:
                value = replace_values(value, values)
            items[key] = value
        return items
    else:
        return item


def find_values(item, keys: List[str]) -> Dict:
    """Find values for keys in item, if present.

    Parameters
    ----------
    item : Any
        Any item
    keys : List[str]
        Keys whose value to retrieve in item

    Returns
    -------
    Dict
        Mapping of key -> value for keys found in item.

    Raises
    ------
    ValueError
        If one key is found whose value is a tuple, list or dict.
    """
    if isinstance(item, (list, tuple)):
        values = {}
        for it in item:
            values.update(find_values(it, keys))
        return values
    elif isinstance(item, dict):
        values = dict()
        for key, value in item.items():
            if key in keys:
                if isinstance(value, (list, tuple, dict)):
                    raise ValueError(f"Cannot find value for '{key}'. Type must be literal but got {type(value)}")
                values[key] = value
            else:
                values.update(find_values(value, keys))
        return values
    else:
        return {}
