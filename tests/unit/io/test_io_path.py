"""Test for io.path"""

import pytest
from unittest.mock import patch

import pyarrow.fs as pfs

import deepr


def strip_hdfs_prefix(f):
    """Remove the HDFS prefix from the first argument of function f"""

    def _strip_prefix(name):
        if name.startswith("viewfs://"):
            prefix_length = len("viewfs://")
            name = name[prefix_length:]
        return name

    def call_with_stripped(*args, **kwargs):
        if len(args) > 0 and type(args[0]) == str:
            return f(_strip_prefix(args[0]), *args[1:], **kwargs)
        return f(*args, **kwargs)

    return call_with_stripped


class MockFS(pfs.FileSystem):
    """Mock FileSystem"""

    def __init__(self):
        self.fs = pfs.LocalFileSystem()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        pass

    def __getattribute__(self, item):
        return strip_hdfs_prefix(object.__getattribute__(self, "fs").__getattribute__(item))


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


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_hdfs(tmpdir):
    """Test path write / read on hdfs file system"""
    path = str(tmpdir.join("test.txt"))
    with deepr.io.Path("viewfs://" + path).open("w") as file:
        file.write("test")

    with deepr.io.Path("viewfs://" + path).open() as file:
        assert file.read() == "test"


def test_io_path_exists_filesystem(file_exists, file_not_exist):
    """Test path exists on any filesystem"""
    assert deepr.io.Path(file_not_exist).exists() is False
    assert deepr.io.Path(file_exists).exists()


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_exists_hdfs(file_exists, file_not_exist):
    """Test path exists on hdfs"""
    assert deepr.io.Path("viewfs://" + file_not_exist).exists() is False
    assert deepr.io.Path("viewfs://" + file_exists).exists()


@pytest.mark.parametrize(
    "fixture_name, result",
    [
        ("dir_exists", True),
        ("dir_not_exist", False),
        ("file_exists", False),
        ("file_not_exist", False),
    ],
)
def test_io_path_is_dir_filesystem(fixture_name, result, request):
    """Test path is dir on any filesystem"""
    path = request.getfixturevalue(fixture_name)
    assert deepr.io.Path(path).is_dir(MockFS()) == result


@pytest.mark.parametrize(
    "fixture_name, result",
    [
        ("dir_exists", True),
        ("dir_not_exist", False),
        ("file_exists", False),
        ("file_not_exist", False),
    ],
)
@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_is_dir_hdfs(fixture_name, result, request):
    """Test path is dir on hdfs"""
    path = request.getfixturevalue(fixture_name)
    assert deepr.io.Path("viewfs://" + path).is_dir() == result


@pytest.mark.parametrize(
    "fixture_name, result",
    [
        ("dir_exists", False),
        ("dir_not_exist", False),
        ("file_exists", True),
        ("file_not_exist", False),
    ],
)
def test_io_path_is_file_filesystem(fixture_name, result, request):
    """Test path is file on any filesystem"""
    path = request.getfixturevalue(fixture_name)
    assert deepr.io.Path(path).is_file(MockFS()) == result


@pytest.mark.parametrize(
    "fixture_name, result",
    [
        ("dir_exists", False),
        ("dir_not_exist", False),
        ("file_exists", True),
        ("file_not_exist", False),
    ],
)
@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_is_file_hdfs(fixture_name, result, request):
    """Test path is file on hdfs"""
    path = request.getfixturevalue(fixture_name)
    assert deepr.io.Path("viewfs://" + path).is_file() == result


def test_io_path_mkdir_filesystem(tmpdir):
    """Test directory creation for any filesystem"""
    path = str(tmpdir.join("some/path"))
    deepr.io.Path(path).mkdir(parents=True, filesystem=MockFS())
    assert pfs.LocalFileSystem().get_file_info(path).type == pfs.FileType.Directory


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_mkdir_hdfs(tmpdir):
    """Test directory creation for hdfs"""
    path = str(tmpdir.join("some/path"))
    deepr.io.Path(path).mkdir(parents=True)
    assert pfs.LocalFileSystem().get_file_info(path).type == pfs.FileType.Directory


def test_io_path_delete_dir_filesystem(dir_exists):
    """Test directory deletion for any filesystem"""
    deepr.io.Path(dir_exists).delete_dir(filesystem=MockFS())
    assert pfs.LocalFileSystem().get_file_info(dir_exists).type == pfs.FileType.NotFound


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_delete_dir_hdfs(dir_exists):
    """Test directory deletion for hdfs"""
    deepr.io.Path(dir_exists).delete_dir()
    assert pfs.LocalFileSystem().get_file_info(dir_exists).type == pfs.FileType.NotFound


def test_io_path_delete_filesystem(file_exists):
    """Test file deletion for any filesystem"""
    deepr.io.Path(file_exists).delete(filesystem=MockFS())
    assert pfs.LocalFileSystem().get_file_info(file_exists).type == pfs.FileType.NotFound


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_delete_hdfs(file_exists):
    """Test file deletion for hdfs"""
    deepr.io.Path("viewfs://" + file_exists).delete()
    assert pfs.LocalFileSystem().get_file_info(file_exists).type == pfs.FileType.NotFound


def read_from_file(filename):
    f = pfs.LocalFileSystem().open_input_stream(filename)
    content = f.read().decode("utf-8")
    f.close()
    return content


def test_io_path_copy_file_filesystem(file_exists, tmpdir):
    """Test file copy for any filesystem"""
    dest = str(tmpdir.join("copied.txt"))
    deepr.io.Path(file_exists).copy_file(dest, filesystem=MockFS())
    assert read_from_file(dest) == read_from_file(file_exists)


@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_copy_file_hdfs(file_exists, tmpdir):
    """Test file copy for hdfs"""
    dest = str(tmpdir.join("copied.txt"))
    deepr.io.Path(file_exists).copy_file(dest)
    assert read_from_file(dest) == read_from_file(file_exists)


@pytest.mark.parametrize(
    "dir_contents",
    [("1.txt",), ("1.txt", "2.txt")],
)
def test_io_path_iterdir_filesystem(dir_contents, dir_exists):
    """Test directory listing for any filesystem"""
    expected = [deepr.io.Path(dir_exists, file) for file in dir_contents]
    for file in dir_contents:
        with deepr.io.Path(dir_exists, file).open(mode="w"):
            pass
    contents = deepr.io.Path(dir_exists).iterdir(filesystem=pfs.LocalFileSystem())
    assert sorted(contents, key=str) == sorted(expected, key=str)


@pytest.mark.parametrize(
    "dir_contents",
    [("1.txt",), ("1.txt", "2.txt")],
)
@patch("deepr.io.path.HDFSFileSystem", MockFS)
def test_io_path_iterdir_hdfs(dir_contents, dir_exists):
    """Test directory listing for hdfs"""
    expected = [deepr.io.Path(dir_exists, file) for file in dir_contents]
    for file in dir_contents:
        with deepr.io.Path(dir_exists, file).open(mode="w"):
            pass
    selector = pfs.FileSelector(dir_exists, recursive=False)
    with patch("pyarrow.fs.FileSelector") as mock:
        mock.return_value = selector
        contents = deepr.io.Path("viewfs://" + dir_exists).iterdir()
    assert sorted(contents, key=str) == sorted(expected, key=str)
