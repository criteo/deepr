#!/usr/bin/env python3
"""Setup script"""

import setuptools
import re
from os import path


_METADATA = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', open("deepr/version.py").read()))

_INSTALL_REQUIRES = [
    "tf-yarn>=0.4.20",
    "cluster-pack>=0.0.7",
    "fire>=0.3",
    "graphyte>=1,<2",
    "jsonnet>=0.15",
    "mlflow",
    "numpy>=1.18,<2",
    "pandas>=1",
    "psutil>=5,<6",
    "pyarrow>=0.14",
    "skein>=0.8",
]

_EXTRAS_REQUIRE = {
    "cpu": ["tensorflow>=1.15,<2"],
    "gpu": ["tensorflow-gpu>=1.15,<2"],
}

_TEST_REQUIRE = ["pytest"]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "docs", "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="deepr",
    author=_METADATA["author"],
    version=_METADATA["version"],
    install_requires=_INSTALL_REQUIRES,
    extras_require=_EXTRAS_REQUIRE,
    tests_require=_TEST_REQUIRE,
    dependency_links=[],
    entry_points={"console_scripts": ["deepr = deepr.cli.main:main"]},
    data_files=[(".", ["requirements.txt", "requirements-gpu.txt", "docs/README.rst"])],
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
