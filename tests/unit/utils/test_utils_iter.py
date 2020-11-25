"""Tests for utils.iter."""

import pytest

import deepr


def _gen():
    yield from range(5)


LIST = list(range(5))


class Custom:
    def __iter__(self):
        yield from range(5)


@pytest.mark.parametrize("iterable", [_gen(), LIST, Custom()])
def test_utils_progress(iterable):
    idx = 0
    for idx, item in enumerate(deepr.utils.progress(iterable)):
        assert item == idx
    assert idx == 4


@pytest.mark.parametrize("iterable", [_gen(), LIST, Custom()])
def test_utils_chunk(iterable, chunk_size: int = 2):
    chunk_id, idx = 0, 0
    for chunk_id, chunk in enumerate(deepr.utils.chunks(iterable, chunk_size)):
        for idx, item in enumerate(chunk):
            assert chunk_size * chunk_id + idx == item
    assert chunk_size * chunk_id + idx == 4
