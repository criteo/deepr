"""Helpers for macros"""

import logging
from typing import Dict, Any, List, Tuple


LOGGER = logging.getLogger(__name__)

MACRO = "$"


def fill_macros(item: Any, macros: Dict[str, Dict[str, Any]] = None) -> Any:
    """Create item whose macro params present in macros are filled.

    Returns a new dictionary, tuple or list or item depending on item's
    type. String params that use the macro syntax "$macro:param" are
    replaced by the relevant entry from macros (`macros[macro][param]`)
    ONLY IF FOUND.

    WARNING: No exception is raised if a macro value is not found in
    `macros`. Use :func:`~assert_no_macros` to check that a config contains
    no macro parameters.

    Parameters
    ----------
    item : Any
        Item to filled
    macros : Dict[str, Dict[str, Any]], optional
        Dictionary of macros

    Returns
    -------
    item
    """
    if not macros:
        return item
    if ismacro(item):
        macro, param = get_macro_and_param(item)
        return macros.get(macro, {}).get(param, item)
    if isinstance(item, dict):
        return {key: fill_macros(value, macros) for key, value in item.items()}
    if isinstance(item, list):
        return [fill_macros(it, macros) for it in item]
    if isinstance(item, tuple):
        return tuple(fill_macros(it, macros) for it in item)
    return item


def find_macro_params(item: Any, macro: str) -> List[str]:
    """Find macro params in item"""
    if ismacro(item):
        item_macro, param = get_macro_and_param(item)
        if item_macro == macro:
            return [param]
    if isinstance(item, dict):
        return find_macro_params(list(item.values()), macro)
    if isinstance(item, tuple):
        return find_macro_params(list(item), macro)
    if isinstance(item, list):
        found = []
        for it in item:
            found.extend(find_macro_params(it, macro))
        return found
    return []


def ismacro(item) -> bool:
    """True if item is a string that looks like '$macro:param'."""
    if isinstance(item, str) and item.startswith(MACRO):
        if len(item.split(":")) != 2:
            raise ValueError(f"Found unexpected macro {item}, format should be '$macro:name'")
        return True
    return False


def get_macro_and_param(item: str) -> Tuple[str, str]:
    """Return name of the macro and param for the item.

    Example
    -------
    >>> from deepr.config import get_macro_and_param
    >>> get_macro_and_param("$macro:param")
    ('macro', 'param')
    """
    macro, param = item[len(MACRO) :].split(":")
    return macro, param


def assert_no_macros(item: Any):
    """Raises a ValueError if item has macro parameters.

    Parameters
    ----------
    item : Any
        Item to be checked

    Raises
    ------
    ValueError
        If any parameter if a macro parameter.
    """
    if ismacro(item):
        raise ValueError(f"Item {item} is a macro value.")
    if isinstance(item, dict):
        for it in item.values():
            assert_no_macros(it)
    if isinstance(item, (list, tuple)):
        for it in item:
            assert_no_macros(it)


def macros_eval_order(macros: Dict = None) -> List[str]:
    """Resolve order of macros evaluation to account for inter macros.

    Parameters
    ----------
    macros : Dict, optional
        Dictionary of macros

    Returns
    -------
    List[str]
    """
    if macros is None:
        return []

    # Resolve dependencies between macros
    deps = dict()
    for name, params in macros.items():
        parents = set()
        for param in params.values():
            if ismacro(param):
                macro, _ = get_macro_and_param(param)
                parents.add(macro)
        deps[name] = parents

    # Resolve evaluation order using dependencies
    order = []  # type: List[str]

    def _add(macro: str, stack: Tuple[str, ...] = ()):
        if macro in order:
            return
        if macro in stack:
            cycle = " -> ".join(list(stack) + [macro])
            raise ValueError(f"Unable to resolve order of macro evaluation (cycle: {cycle}, dependencies: {deps})")
        if macro not in deps:
            raise ValueError(f"Missing macro: {macro}, mentioned by : {' -> '.join(stack)})")
        for parent in deps[macro]:
            _add(parent, (*stack, macro))
        order.append(macro)

    for macro in macros:
        _add(macro)

    # Macros should appear once and only once
    if len(order) != len(macros) or set(order) != set(macros):
        raise ValueError(f"Order {order} incoherent with macros {macros.keys()}")

    return order
