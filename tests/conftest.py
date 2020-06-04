"""Setup tests."""

import logging

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--no-cleanup", action="store_true", help="Disable Auto Cleanup")


def pytest_runtest_setup(item):
    """Before test runs, reset Tensorflow graph.

    Parameters
    ----------
    item : test function
    """
    # pylint: disable=unused-argument
    LOGGER.info("Reset default graph")
    tf.reset_default_graph()


def pytest_runtest_teardown(item):
    """After test has run, remove tmpdir if fixture is given.

    Parameters
    ----------
    item : test function
    """
    if not item.config.getoption("--no-cleanup"):
        if "tmpdir" in item.funcargs:
            tmpdir = item.funcargs["tmpdir"]
            if tmpdir.check():
                LOGGER.info(f"Cleaning-up {str(tmpdir)}")
                tmpdir.remove()
