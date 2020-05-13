"""Path Utilities"""

from contextlib import contextmanager
from typing import Union, Generator
import os
import pathlib
from urllib import parse

import tensorflow as tf
from pyarrow.filesystem import FileSystem

from deepr.io.hdfs import HDFSFileSystem, HDFSFile
from deepr.utils.datastruct import to_flat_tuple


class Path:
    """Equivalent of pathlib.Path for local and HDFS FileSystem

    Automatically opens and closes an HDFS connection if the path is
    an HDFS path.

    Allows you to work with local / HDFS files in an agnostic manner.

    DISCLAIMER : Might be replaced / changed to use tf.io.gfile.GFile.

    Example
    -------
    .. code-block:: python

        path = Path("viewfs://foo", "bar")
        with path.open("r") as file:
            for line in file:
                print(line)

    """

    def __init__(self, *args: Union[str, pathlib.Path, "Path"]):
        self.path = os.path.join(*[str(arg) for arg in to_flat_tuple(args)])

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"Path({str(self)})"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return str(self) == other
        else:
            return str(self) == str(other)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def parent(self):
        return Path("/".join(self.path.split("/")[:-1]))

    @property
    def is_hdfs(self) -> bool:
        scheme = parse.urlparse(str(self)).scheme
        return scheme in {"hdfs", "viewfs"}

    @property
    def is_local(self) -> bool:
        return not self.is_hdfs

    @property
    def suffix(self):
        return pathlib.Path(str(self)).suffix

    def exists(self, filesystem: FileSystem = None) -> bool:
        if filesystem is not None:
            return filesystem.exists(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.exists(str(self))
            else:
                return pathlib.Path(str(self)).exists()

    def is_dir(self, filesystem: FileSystem = None) -> bool:
        if filesystem is not None:
            return filesystem.isdir(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.isdir(str(self))
            else:
                return pathlib.Path(str(self)).is_dir()

    def is_file(self, filesystem: FileSystem = None) -> bool:
        if filesystem is not None:
            return filesystem.isfile(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.isfile(str(self))
            else:
                return pathlib.Path(str(self)).is_file()

    def mkdir(self, parents: bool = False, exist_ok: bool = False, filesystem: FileSystem = None):
        if filesystem is not None:
            filesystem.mkdir(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    hdfs.mkdir(str(self))
            else:
                pathlib.Path(str(self)).mkdir(parents=parents, exist_ok=exist_ok)

    def delete(self, filesystem: FileSystem = None):
        """Delete file from filesystem"""
        if not self.is_file(filesystem=filesystem):
            raise FileNotFoundError(str(self))
        if filesystem is not None:
            filesystem.delete(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    hdfs.delete(str(self))
            else:
                pathlib.Path(str(self)).unlink()

    def iterdir(self, filesystem: FileSystem = None) -> Generator["Path", None, None]:
        if filesystem is not None:
            return (Path(path) for path in list(filesystem.ls(str(self))))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return (Path(path) for path in list(hdfs.ls(str(self))))
            else:
                return (Path(str(path)) for path in pathlib.Path(str(self)).iterdir())

    def glob(self, pattern):
        if not self.is_hdfs:
            return [str(path) for path in pathlib.Path(str(self)).glob(pattern)]
        else:
            return (Path(path) for path in tf.io.gfile.glob(str(Path(self, pattern))))

    @contextmanager
    def open(self, mode: str = "r", encoding: str = None, filesystem: FileSystem = None):
        """Open File"""
        if filesystem is not None:
            with HDFSFile(filesystem=filesystem, path=str(self), mode=mode) as file:
                yield file
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    with HDFSFile(filesystem=hdfs, path=str(self), mode=mode) as file:
                        yield file
            else:
                with pathlib.Path(str(self)).open(mode=mode, encoding=encoding) as file:
                    yield file
