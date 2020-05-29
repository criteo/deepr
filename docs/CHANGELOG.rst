Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_.

[Unreleased]
------------

Added
~~~~~
Doctest run in CI.

Changed
~~~~~~~
Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Doctests were fixed.
Add metric name sanitizer, especially needed to sanitize keras built variable names.

Security
~~~~~~~~

1.2.1 - 2020-05-27
------------------

Added
~~~~~
Changed
~~~~~~~
Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Avoid mkdir for HDFS path_model for permissions reasons

Security
~~~~~~~~


1.2.0 - 2020-05-26
------------------

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
- wrong arguments in ``YarnConfig`` for ``upload_zip_to_hdfs``.

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

