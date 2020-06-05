"""Utilities for logging."""

import time
import logging
from typing import Iterable


LOGGER = logging.getLogger(__name__)


def progress(iterable: Iterable, secs: int = 60):
    """Log progress on Iterable.iterable

    >> from deepr.utils import progress
    >>> for idx in progress(range(2), secs=10):
    ...     print(idx)
    0
    1
    """

    def _wrapped():
        start = time.time()
        last = start
        idx = 0
        for idx, item in enumerate(iterable):
            # Log progress
            now = time.time()
            if now - last > secs:
                speed = (idx + 1) / (now - start)
                last = now
                LOGGER.info(f"- At {idx + 1}, item/sec = {speed}")
            yield item

        # Log end
        speed = (idx + 1) / (time.time() - start)
        LOGGER.info(f"Number of items {idx + 1}, item/sec = {speed}")

    if secs is None:
        return iterable
    else:
        return _wrapped()


def chunks(iterable: Iterable, chunk_size: int):
    """Split Iterable into Iterable chunks.

    >>> from deepr.utils import chunks
    >>> for chunk in chunks(range(5), chunk_size=2):
    ...     print("-")
    ...     for idx in chunk:
    ...         print(idx)
    -
    0
    1
    -
    2
    3
    -
    4
    """

    def _wrapped():
        it = iter(iterable)
        state = {"end": False}

        def _chunk():
            try:
                for _ in range(chunk_size):
                    yield next(it)
            except StopIteration:
                state["end"] = True

        while True:
            yield _chunk()
            if state["end"]:
                break

    return _wrapped()
