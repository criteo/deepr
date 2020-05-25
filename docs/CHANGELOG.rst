Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_.


[Unreleased]
------------

Added
~~~~~
- ``OptimizeSavedModel`` now supports multiple fetches
- new graph utils, ``import_graph_def``, ``get_feedable_tensors``, ``get_fetchable_tensors``

Changed
~~~~~~~
- ``example.jobs.Predict`` arguments (``path_model`` and ``graph_name`` instead of ``path_model_pb``, ``fetch`` instead of ``fetches`` for consistency with ``OptimizeSavedModel``).

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~


[1.1.0] - 2020-05-25
--------------------

Added
~~~~~
- Remove some kwargs for cleaner error stacks
- Make example more complex, add advanced notebook
- Track missing macro
- Update doc of logging tensor (change prefix to name)
- Add helper to debug class building from config

[1.0.0] - 2020-05-19
--------------------

Added
~~~~~
- Public Release

