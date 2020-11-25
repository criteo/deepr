# pylint: disable=redefined-outer-name
"""Test for utils.uuid"""

import pytest

import deepr


@pytest.fixture
def msb():
    return 2889684086332146052


@pytest.fixture
def lsb():
    return -7031166994073208036


@pytest.fixture
def uuid():
    return "281a385d-c6ed-4184-9e6c-46df21d0bb1c"


def test_msb_lsb_to_str(msb, lsb, uuid):
    assert deepr.utils.msb_lsb_to_str(msb, lsb) == uuid


def test_str_to_msb_lsb(msb, lsb, uuid):
    assert deepr.utils.str_to_msb_lsb(uuid) == (msb, lsb)
