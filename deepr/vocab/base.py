"""Utilities for vocabularies"""

from typing import Iterable, List

from deepr.io.path import Path


def write(path: str, vocab: Iterable[str]):
    """Write vocabulary to file.

    Parameters
    ----------
    path : str
        Path to .txt file with one item per line
    vocab : Iterable[str]
        Iterable of lexemes (strings)
    """
    with Path(path).open("w") as file:
        file.write("\n".join(map(str, vocab)))


def read(path: str) -> List[str]:
    """Read vocabulary from file.

    Parameters
    ----------
    path : str
        Path to .txt file with one item per line
    """
    with Path(path).open() as file:
        return [line for line in file if line.strip()]


def size(path: str) -> int:
    """Return vocabulary size from mapping file.

    Parameters
    ----------
    path : str
        Path to .txt file with one item per line
    """
    return len(read(path))
