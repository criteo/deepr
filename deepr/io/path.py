"""Path Utilities"""

from contextlib import contextmanager
from typing import Union, Generator, Optional
import os
import pathlib
from urllib import parse
import shutil
import logging

import tensorflow as tf
from pyarrow.filesystem import FileSystem

from deepr.io.hdfs import HDFSFileSystem, HDFSFile
from deepr.utils.datastruct import to_flat_tuple


LOGGER = logging.getLogger(__name__)


class Path:
    """Equivalent of pathlib.Path for local and HDFS FileSystem

    Automatically opens and closes an HDFS connection if the path is
    an HDFS path.

    Allows you to work with local / HDFS files in an agnostic manner.

    Example
    -------
    .. code-block:: python

        path = Path("viewfs://foo", "bar") / "baz"
        path.parent.mkdir()
        with path.open("r") as file:
            for line in file:
                print(line)
        for path in path.glob("*"):
            print(path.is_file())

    """

    def __init__(self, *args: Union[str, pathlib.Path, "Path"]):
        self.path = os.path.join(*[str(arg) for arg in to_flat_tuple(args)])

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"Path({str(self)})"

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __truediv__(self, other) -> "Path":
        """Syntactic sugar for path definition."""
        return Path(self, other)

    @property
    def name(self) -> str:
        """Final path component."""
        return os.path.basename(self.path)

    @property
    def parent(self):
        """Path to the parent of the current path"""
        return Path("/".join(self.path.split("/")[:-1]))

    @property
    def is_hdfs(self) -> bool:
        """Return True if the path points to an HDFS location"""
        scheme = parse.urlparse(str(self)).scheme
        return scheme in {"hdfs", "viewfs"}

    @property
    def is_local(self) -> bool:
        """Return True if the path points to a local file or dir."""
        return not self.is_hdfs

    @property
    def suffix(self):
        """File extension of the file if any."""
        return pathlib.Path(str(self)).suffix

    def exists(self, filesystem: FileSystem = None) -> bool:
        """Return True if the path points to an existing file or dir."""
        if filesystem is not None:
            return filesystem.exists(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.exists(str(self))
            else:
                return pathlib.Path(str(self)).exists()

    def is_dir(self, filesystem: FileSystem = None) -> bool:
        """Return True if the path points to a regular directory."""
        if filesystem is not None:
            return filesystem.isdir(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.isdir(str(self))
            else:
                return pathlib.Path(str(self)).is_dir()

    def is_file(self, filesystem: FileSystem = None) -> bool:
        """Return True if the path points to a regular file."""
        if filesystem is not None:
            return filesystem.isfile(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return hdfs.isfile(str(self))
            else:
                return pathlib.Path(str(self)).is_file()

    def mkdir(self, parents: bool = False, exist_ok: bool = False, filesystem: FileSystem = None):
        """Create directory"""
        if self.is_dir(filesystem=filesystem):
            if exist_ok:
                return
            else:
                raise Exception(f"Directory {self} already exists.")
        if filesystem is not None:
            filesystem.mkdir(str(self))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    hdfs.mkdir(str(self))
            else:
                pathlib.Path(str(self)).mkdir(parents=parents, exist_ok=exist_ok)

    def delete_dir(self, filesystem: FileSystem = None):
        """Delete dir from filesystem"""
        if not self.is_dir(filesystem=filesystem):
            raise FileNotFoundError(str(self))
        if filesystem is not None:
            filesystem.rm(str(self), recursive=True)
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    hdfs.rm(str(self), recursive=True)
            else:
                shutil.rmtree(str(self))

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

    def copy_file(self, dest, filesystem: FileSystem = None):
        """Copy current file to dest (target directory must exist)."""
        LOGGER.info(f"Copying file {self} to {dest}")
        if not self.is_file(filesystem=filesystem):
            raise FileNotFoundError(str(self))
        if self.is_hdfs:
            tf.io.gfile.copy(str(self), str(dest), overwrite=True)
        else:
            shutil.copy(str(self), str(dest))

    def copy_dir(self, dest, recursive: bool = False, filesystem: FileSystem = None):
        """Copy current files and directories if recursive to dest."""
        LOGGER.info(f"Copying {self} to {dest}")
        if not self.is_dir(filesystem=filesystem):
            raise FileNotFoundError(str(self))
        Path(dest).mkdir(parents=True, exist_ok=True, filesystem=filesystem)
        for path in self.glob("*"):
            if path.is_file(filesystem):
                path.copy_file(Path(dest) / path.name, filesystem=filesystem)
            elif path.is_dir(filesystem):
                if recursive:
                    path.copy_dir(Path(dest) / path.name, recursive=recursive, filesystem=filesystem)
            else:
                raise Exception(f"Unable to copy {path}")

    def iterdir(self, filesystem: FileSystem = None) -> Generator["Path", None, None]:
        """Retrieve directory content."""
        if filesystem is not None:
            return (Path(path) for path in list(filesystem.ls(str(self))))
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    return (Path(path) for path in list(hdfs.ls(str(self))))
            else:
                return (Path(str(path)) for path in pathlib.Path(str(self)).iterdir())

    def glob(self, pattern) -> Generator["Path", None, None]:
        """Retrieve directory content matching pattern"""
        if not self.is_hdfs:
            return (Path(path) for path in pathlib.Path(str(self)).glob(pattern))
        else:
            return (Path(path) for path in tf.io.gfile.glob(str(Path(self, pattern))))

    @contextmanager
    def open(self, mode: str = "r", encoding: Optional[str] = "utf-8", filesystem: FileSystem = None):
        """Open file on both HDFS and Local File Systems.

        Example
        -------
        Use a context manager like so

        .. code-block:: python

            path = Path("viewfs://root/user/path/to/file.txt")
            with path.open("w") as file:
                file.write("Hello world!")

        """
        if "b" in mode:
            encoding = None  # mypy: ignore
        if filesystem is not None:
            with HDFSFile(filesystem=filesystem, path=str(self), mode=mode, encoding=encoding) as file:
                yield file
        else:
            if self.is_hdfs:
                with HDFSFileSystem() as hdfs:
                    with HDFSFile(filesystem=hdfs, path=str(self), mode=mode, encoding=encoding) as file:
                        yield file
            else:
                with pathlib.Path(str(self)).open(mode=mode, encoding=encoding) as file:
                    yield file
