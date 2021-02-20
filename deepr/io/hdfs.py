"""HDFS Utilities"""

from typing import Optional
import logging

from pyarrow.fs import FileSystem, HadoopFileSystem

LOGGER = logging.getLogger(__name__)


class HDFSFileSystem:
    """Context aware HDFSFileSystem using pyarrow.hdfs.

    Open and closes connection to HDFS thanks to a context manager

    >>> from deepr.io import HDFSFileSystem
    >>> with HDFSFileSystem() as fs:  # doctest: +SKIP
    ...     fs.open("path/to/file")  # doctest: +SKIP
    """

    def __init__(self, *args, host="viewfs://root", port=0, **kwargs):
        self._host = host
        self._port = port
        self._hdfs = None
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        self._hdfs = HadoopFileSystem(self._host, self._port, *self._args, **self._kwargs)
        return self._hdfs

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        self._hdfs = None

    def __getattr__(self, name):
        # Expose self._hdfs methods and attributes (mimic inheritance)
        return getattr(self._hdfs, name)


class HDFSFile:
    """FileSystemFile, support of "r", "w" modes, readlines and iter.

    Makes it easier to read or write file from any filesystem. For
    example, if you use HDFS you can do

    >>> from deepr.io import HDFSFileSystem
    >>> with HDFSFileSystem() as fs:
    ...     with HDFSFile(fs, "viewfs://root/user/foo.txt", "w") as file:  # doctest: +SKIP
    ...         file.write("Hello world!")  # doctest: +SKIP

    The use of context manager means that the connection to the
    filesystem is automatically opened / closed, as well as the file
    buffer.

    Attributes
    ----------
    filesystem : FileSystem
        FileSystem instance
    path : str
        Path to file
    mode : str, Optional
        Write / read mode. Supported: "r", "rb" (default), "w", "wb".
    """

    DEFAULT_BUFFER_SIZE = 2 ** 16  # 64K

    def __init__(self, filesystem: FileSystem, path: str, mode: str = "rb", encoding: Optional[str] = "utf-8"):
        self.filesystem = filesystem
        self.path = path
        self.mode = mode
        self.encoding = None if "b" in mode else encoding
        self._file = (
            filesystem.open_input_stream(path, compression="detect", buffer_size=HDFSFile.DEFAULT_BUFFER_SIZE)
            if "r" in mode
            else filesystem.open_output_stream(path, compression="detect", buffer_size=HDFSFile.DEFAULT_BUFFER_SIZE)
        )

    def __iter__(self):
        yield from self.readlines()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # pylint: disable=redefined-builtin
        self._file.close()

    def __getattr__(self, name):
        # Expose self._file methods and attributes (mimic inheritance)
        return getattr(self._file, name)

    def write(self, data, *args, **kwargs):
        # pylint: disable=unused-argument
        # TODO: remove arguments
        if self.mode == "w":
            self._file.write(data.encode(encoding=self.encoding))
        elif self.mode == "wb":
            self._file.write(data)
        else:
            raise ValueError(f"Mode {self.mode} unkown (must be 'w' or 'wb').")

    def read(self, *args, **kwargs):
        if self.mode == "r":
            return self._file.read(*args, **kwargs).decode(encoding=self.encoding)
        elif self.mode == "rb":
            return self._file.read(*args, **kwargs)
        else:
            raise ValueError(f"Mode {self.mode} unkown (must be 'r' or 'rb')")

    def readlines(self):
        return self.read().split("\n" if self.mode == "r" else b"\n")
