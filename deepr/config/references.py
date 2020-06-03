"""Helpers for references"""

import logging
from typing import Dict, Any


LOGGER = logging.getLogger(__name__)

REF = "@"

REF_SELF = "@self"

REF_MACROS = "@macros"

REF_MACROS_EVAL = "@macros_eval"


def fill_references(item: Any, references: Dict[str, Any] = None) -> Any:
    """Fill all params that are references, fail if not found.

    Returns a new dictionary, tuple or list or item depending on item's
    type.

    Parameters that use the ref syntax "@reference" are replaced by the
    relevant entry from references (`references['@reference']`).

    If a reference is not found in `references`, raise `ValueError`

    Parameters
    ----------
    item : Any
        Any item, but typically a Dict
    references : Dict[str, Any], optional
        Mapping of names to reference objects

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        If some references are not found
    """
    if isreference(item):
        if references is None:
            raise ValueError(f"Found reference {item} but references is None.")
        if item not in references:
            raise ValueError(f"Found reference {item} not in references")
        return references[item]
    if isinstance(item, dict):
        return {key: fill_references(value, references) for key, value in item.items()}
    if isinstance(item, list):
        return [fill_references(it, references) for it in item]
    if isinstance(item, tuple):
        return tuple(fill_references(it, references) for it in item)
    return item


def isreference(item) -> bool:
    """True if item is a string that looks like '@reference'."""
    return isinstance(item, str) and item.startswith(REF)


def default_references(config: Dict, macros: Dict = None, macros_eval: Dict = None) -> Dict[str, Any]:
    """Create default references from config, macros and macros_eval.

    Evaluation mode for the default references is set to "skip" to avoid
    double evaluation of those nested references.
    """
    config = {**config, "eval": None} if config is not None else None
    macros = {**macros, "eval": None} if macros is not None else None
    macros_eval = {**macros_eval, "eval": None} if macros_eval is not None else None
    return {REF_SELF: config, REF_MACROS: macros, REF_MACROS_EVAL: macros_eval}
