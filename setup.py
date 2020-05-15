#!/usr/bin/env python3
"""Setup script"""

import setuptools
import os
import re


_METADATA = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', open("deepr/version.py").read()))


_GPU_SUFFIX = "_gpu" if "BUILD_GPU" in os.environ else ""


_INSTALL_REQUIRES = [
    f"tf-yarn{_GPU_SUFFIX}==0.4.15",
    f"tensorflow{_GPU_SUFFIX}==1.15.2",
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


_TEST_REQUIRE = ["pytest"]


setuptools.setup(
    name=f"deepr{_GPU_SUFFIX}",
    author=_METADATA["author"],
    version=_METADATA["version"],
    install_requires=_INSTALL_REQUIRES,
    tests_require=_TEST_REQUIRE,
    dependency_links=[],
    entry_points={"console_scripts": ["deepr = deepr.cli.main:main"]},
    data_files=[(".", ["requirements.txt"])],
    packages=setuptools.find_packages(),
)
