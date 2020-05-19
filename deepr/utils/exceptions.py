"""Utilities for dealing with exceptions."""

import logging
import functools


LOGGER = logging.getLogger(__name__)


def handle_exceptions(fn):
    """Handle Exceptions Decorator."""
    # pylint: disable=broad-except,protected-access

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as ex:
            LOGGER.error(f"Exception in '{fn.__name__}': {ex}")
            return False

    _wrapper.__handle_exceptions = True

    return _wrapper
