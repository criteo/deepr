"""Evaluate objects from arbitrary nested dictionaries."""

import functools
import logging
from typing import Dict, Any

from deepr.config.macros import fill_macros, find_macro_params, assert_no_macros, macros_eval_order
from deepr.config.references import fill_references, default_references


LOGGER = logging.getLogger(__name__)

TYPE = "type"

EVAL = "eval"

CALL = "call"

PARTIAL = "partial"

POSITIONAL = "*"


def parse_config(config: Dict, macros: Dict = None) -> Dict:
    """Fill macro parameters and references in config from macros.

    Example
    -------
    >>> from deepr.config import parse_config
    >>> config = {"x": "$params:x", "y": 2}
    >>> macros = {"params": {"x": 1}}
    >>> parse_config(config, macros)
    {'x': 1, 'y': 2}

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
        mode = item.get(EVAL, CALL)
        params = {key: value for key, value in item.items() if key != EVAL}

        # Return evaluated class or function with provided arguments
        if mode == CALL:
            if TYPE in params:
                cls_or_fn = _import(params[TYPE])
                args = from_config(params.get(POSITIONAL))
                kwargs = {key: from_config(value) for key, value in params.items() if key not in {TYPE, POSITIONAL}}
                try:
                    return cls_or_fn(*args, **kwargs) if args else cls_or_fn(**kwargs)
                except (ValueError, TypeError) as e:
                    raise type(e)(f"Error while calling {cls_or_fn})") from e
            else:
                return {key: from_config(value) for key, value in params.items()}

        # Return partial class or function with provided arguments
        if mode == PARTIAL:
            if TYPE in params:
                cls_or_fn = _import(params[TYPE])
                args = from_config(params.get(POSITIONAL))
                kwargs = {key: from_config(value) for key, value in params.items() if key not in {TYPE, POSITIONAL}}
                return functools.partial(cls_or_fn, *args, **kwargs) if args else functools.partial(cls_or_fn, **kwargs)
            else:
                instantiated = from_config(params)

                def _partial(**kwargs):
                    return {**instantiated, **kwargs}

                return _partial

        # Return raw dictionary
        if mode is None:
            return params

        raise ValueError(f"Unexpected evaluation mode: '{mode}' in item {item}")

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
