"""Common fixtures for deepr.io tests"""

import pytest

import pyarrow.fs as pfs


@pytest.fixture
def file_exists(tmpdir):
    path = str(tmpdir.join("exists.txt"))
    f = pfs.LocalFileSystem().open_output_stream(path)
    f.write("Hello world!".encode("utf-8"))
    f.close()
    return path


@pytest.fixture
def file_not_exist(tmpdir):
    path = str(tmpdir.join("not_exist.txt"))
    return path


@pytest.fixture
def dir_exists(tmpdir):
    path = str(tmpdir.join("dir_exists"))
    pfs.LocalFileSystem().create_dir(path)
    return path


@pytest.fixture
def dir_not_exist(tmpdir):
    path = str(tmpdir.join("dir_not_exist.txt"))
    return path
