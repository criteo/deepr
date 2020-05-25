"""Evaluate objects from arbitrary nested dictionaries."""

import logging
from typing import Dict, Any

from deepr.config.macros import fill_macros, find_macro_params, assert_no_macros, macros_eval_order
from deepr.config.references import fill_references, default_references


LOGGER = logging.getLogger(__name__)

KEY_TYPE = "type"

KEY_EVAL_MODE = "eval"

KEY_POSITIONAL_ARG = "*"

EVAL_MODE_INSTANCE = "instance"

EVAL_MODE_SKIP = "skip"


def parse_config(config: Dict, macros: Dict = None) -> Dict:
    """Fill macro parameters and references in config from macros.

    Example
    -------
    >>> config = {"x": "$params:x", "y": 2}
    >>> macros = {"params": {"x": 1}}
    >>> parse_config(config, macros)
    {"x": 1, "y": 2}

    Parameters
    ----------
    config : Dict
        Config dictionary
    macros : Dict, optional
        Dictionary of macro parameters.

    Returns
    -------
    Dict
        Parsed Config, without macro parameters and references.

    Raises
    ------
    ValueError
        If some macro parameter in config not found in macros.
        If some references not found.
    """
    # Evaluate macros in order to account for inter macro params
    if macros is not None:
        macros_eval = dict()  # type: ignore
        for macro in macros_eval_order(macros):
            macro_eval = fill_macros(macros[macro], macros_eval)
            assert_no_macros(macro_eval)
            macros_eval[macro] = from_config(macro_eval)
    else:
        macros_eval = None  # type: ignore

    # Fill macros and references in config, evaluate and return
    config_no_macro = fill_macros(config, macros_eval)
    assert_no_macros(config_no_macro)
    references = default_references(config=config, macros=macros, macros_eval=macros_eval)
    parsed = fill_references(config_no_macro, references)

    # Look for macro params that were not used
    if macros_eval is not None:
        for macro, params in macros_eval.items():
            used = find_macro_params({"config": config, "macros": macros}, macro)
            for param in set(params) - set(used):
                LOGGER.warning(f"- MACRO PARAM NOT USED: macro = '{macro}', param = '{param}'")

    return parsed


def from_config(item: Any) -> Any:
    """Instantiate item from config.

    Raises
    ------
    ValueError
        If item is not a valid config (unexpected eval method)
    """
    if isinstance(item, dict):
        # Get eval_mode from item, default is EVAL_MODE_INSTANCE
        eval_mode = item.get(KEY_EVAL_MODE, EVAL_MODE_INSTANCE)
        item = {key: value for key, value in item.items() if key != KEY_EVAL_MODE}

        # Retrieve type, evaluate arguments and return instance
        if eval_mode == EVAL_MODE_INSTANCE:
            if KEY_TYPE in item:
                cls = _import(item[KEY_TYPE])
                args = item.get(KEY_POSITIONAL_ARG)
                kwargs = {key: value for key, value in item.items() if key not in {KEY_TYPE, KEY_POSITIONAL_ARG}}
                try:
                    return cls(*from_config(args), **from_config(kwargs)) if args else cls(**from_config(kwargs))
                except TypeError as e:
                    raise TypeError(f"Error instantiating {cls})") from e
            else:
                return {key: from_config(value) for key, value in item.items()}

        # Return item (no recursive evaluation)
        if eval_mode == EVAL_MODE_SKIP:
            return item

        # Fail if eval_mode is not recognized.
        raise ValueError(f"Unexpected eval_mode: '{eval_mode}' in item {item}")
    if isinstance(item, list):
        return [from_config(it) for it in item]
    if isinstance(item, tuple):
        return tuple(from_config(it) for it in item)
    return item


def _import(import_str: str):
    """Import class using import string"""
    parts = import_str.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
