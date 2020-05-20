"""Setup script."""

from pathlib import Path
import re
import setuptools


if __name__ == "__main__":
    # Read metadata from version.py
    with Path("deepr/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))

    # Read description from README
    with Path(Path(__file__).parent, "docs", "README.rst").open(encoding="utf-8") as file:
        long_description = file.read()

    # Run setup
    setuptools.setup(
        name="deepr",
        author=metadata["author"],
        version=metadata["version"],
        install_requires=[
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
        ],
        extras_require={"cpu": ["tensorflow>=1.15,<2"], "gpu": ["tensorflow-gpu>=1.15,<2"]},
        tests_require=["pytest"],
        dependency_links=[],
        entry_points={"console_scripts": ["deepr = deepr.cli.main:main"]},
        data_files=[(".", ["requirements.txt", "requirements-gpu.txt", "docs/README.rst"])],
        packages=setuptools.find_packages(),
        description=long_description.split("\n")[0],
        long_description=long_description,
        long_description_content_type="text/x-rst",
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
