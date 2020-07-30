"""Utilities for vocabularies."""

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
        Lexemes should not have newline characters.
    """
    # Check that vocab is not a string
    if isinstance(vocab, str):
        msg = f"Expected iterable of strings, but got a string ({vocab})"
        raise TypeError(msg)

    # Check that no item in vocab has a newline character
    for item in vocab:
        if not isinstance(item, str):
            msg = f"Expected item of type str, but got {type(item)} for item {item}"
            raise TypeError(msg)
        if "\n" in item:
            msg = f"Found newline character in item {item} (forbidden)."
            raise ValueError(msg)

    # Write each item on a new line
    with Path(path).open("w") as file:
        file.write("\n".join(vocab))


def read(path: str) -> List[str]:
    """Read vocabulary from file.

    Parameters
    ----------
    path : str
        Path to .txt file with one item per line
    """
    with Path(path).open() as file:
        return [line.strip() for line in file if line.strip()]


def size(path: str) -> int:
    """Return vocabulary size from mapping file.

    Parameters
    ----------
    path : str
        Path to .txt file with one item per line
    """
    return len(read(path))
