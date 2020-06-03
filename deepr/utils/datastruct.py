"""Generic Iter Utilities"""

from typing import Tuple, Dict
import types


def to_flat_tuple(items) -> Tuple:
    """Convert nested list, tuples and generators to a flat tuple.

    Flatten any nested structure of items. Will unpack lists, tuple and
    generators. Any other type will not be unpacked, meaning that you
    can safely use this function on other iterable types like strings
    or `tf.Tensor`. For example:

    .. code-block:: python

        to_flat_tuple(1)  # (1,)
        to_flat_tuple("hello")  # ("hello",)
        to_flat_tuple(tf.ones([2, 2]))  # (tf.ones([2, 2]),)
        to_flat_tuple((x for x in range(2)))  # (0, 1)
        to_flat_tuple((1, 2))  # (1, 2)
        to_flat_tuple(((0, 1), 2))  # (0, 1, 2)

    Parameters
    ----------
    items : Item, Tuple, List or Generator (nested)
        Items to transform to a flat tuple

    Returns
    -------
    Tuple
    """
    if isinstance(items, (tuple, list)):
        flat = []  # type: ignore
        for item in items:
            flat.extend(to_flat_tuple(item))
        return tuple(flat)
    elif isinstance(items, types.GeneratorType):
        return to_flat_tuple(list(items))
    else:
        return (items,)


def item_to_dict(items, keys) -> Dict:
    """Convert tuple or object to dictionary."""
    if isinstance(keys, str):
        return {keys: items}
    else:
        return dict(zip(keys, items))


def dict_to_item(data: Dict, keys):
    """Convert dictionary into object or tuple of objects."""
    if isinstance(keys, str):
        return data[keys]
    else:
        return tuple(data[key] for key in keys)
