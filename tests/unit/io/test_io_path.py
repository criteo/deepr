"""Test for io.path"""

import pathlib
import pytest

import deepr


class MockFS:
    """Mock FileSystem"""

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        pass

    def open(self, name: str, mode: str):
        if mode not in {"rb", "wb"}:
            raise ValueError()
        if name.startswith("viewfs://"):
            prefix_length = len("viewfs://")
            name = name[prefix_length:]
        return pathlib.Path(name).open(mode)


@pytest.mark.parametrize(
    "args, string",
    [
        (("foo/bar",), "foo/bar"),
        (("foo", "bar"), "foo/bar"),
        (("foo", "bar"), "foo/bar"),
        (("viewfs://root", "foo/bar"), "viewfs://root/foo/bar"),
    ],
)
def test_io_path_string(args, string):
    """Test string and equality"""
    assert deepr.io.Path(*args) == string


@pytest.mark.parametrize(
    "args, parent", [(("foo", "bar"), "foo"), (("viewfs://root", "foo", "bar"), "viewfs://root/foo")]
)
def test_io_path_parent(args, parent):
    """Test parent method"""
    path = deepr.io.Path(*args)
    assert path.parent == parent


def test_io_path_local(tmpdir):
    """Test path write / read on local file system"""
    path = tmpdir.join("test.txt")
    with deepr.io.Path(path).open("w") as file:
        file.write("test")

    with deepr.io.Path(path).open() as file:
        assert file.read() == "test"


def test_io_path_filesystem(tmpdir):
    """Test path write / read on any file system"""
    path = str(tmpdir.join("test.txt"))
    with deepr.io.Path("viewfs://" + path).open("w", filesystem=MockFS()) as file:
        file.write("test")

    with deepr.io.Path(path).open() as file:
        assert file.read() == "test"
