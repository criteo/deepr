Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_.


[Unreleased]
------------

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
Security
~~~~~~~~

[2.3.0] - 2020-06-10
--------------------

Added
~~~~~
- ``writers`` module, ``TFRecordWriter``
- ``ToExample`` prepro to convert a dataset to ``tf.Example``
- ``Field`` now has a ``to_feature(value)`` method
- ``iter`` utils: ``progress`` (logs progress every n seconds), ``chunks`` to return chunks from an iterable
- ``SaveDataset`` job to write a dataset to tfrecords.

Changed
~~~~~~~
- ``TFRecordSequenceExample`` renamed ``FromExample`` (but old name still available).
- ``Field`` method ``as_feature`` renamed ``feature_specs`` to avoid confusion with ``to_feature``.

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
- Removed ``Field`` methods (leading to incorrect uses): ``has_var_len``, ``as_feature``, ``has_fixed_len``

Fixed
~~~~~
- Incorrect ``shuffle`` argument use in ``TFRecordReader``

Security
~~~~~~~~

[2.2.0] - 2020-06-08
------------

Added
~~~~~
ExportXlaModelMetadata job is added : make it possible to export metadata for xla models
Changed
~~~~~~~
Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~

[2.1.1] - 2020-06-05
--------------------

Added
~~~~~
- Predictors also yield inputs when applied on a ``tf.data.Dataset``

Changed
~~~~~~~
Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~


[2.1.0] - 2020-06-04
--------------------

Added
~~~~~
- Add ``predictors``
- Add new example job ``PredictSavedModel``

Changed
~~~~~~~
- Example job ``Predict`` renamed into ``PredictProto``

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~


[2.0.0] - 2020-06-03
--------------------

Added
~~~~~
- Doctest run in CI.

Changed
~~~~~~~
- copy_dir job will now overwrite the target by default
- Nested support for ``prepros.Serial``
- Context manager ``TableContext`` for tables reuse
- Automatic table context creation in ``prepro.__call__``
- Prepro ``TableInitializer`` to run ``table_initializer_fn`` before ``map`` transforms
- Vocabulary utilities (``read``, ``write``, ``size``)
- Reverse lookup table function ``index_to_string_table_from_file`` and associated layer ``LookupIndexToString``
- Layer combinator ``ActiveMode`` to apply layer only on given modes
- Layer ``ToFloat``
- Config evaluation modes: ``skip`` -> ``None``, ``instance`` -> ``call``
- New evaluation mode for config dictionary ``partial``
- Remove ``__post_init__`` for ``YarnTrainer`` and ``YarnLauncher`` to avoid unexpected non-laziness

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
- Use of ``prepro`` and ``layer`` decorator on constructors
- Lazy behavior for ``prepro`` and ``layer`` decorator

Fixed
~~~~~
- Doctests were fixed.
- Add metric name sanitizer, especially needed to sanitize keras built variable names.
- Typo in ``example`` predict (feedable / fetchable)

Security
~~~~~~~~

[1.2.1] - 2020-05-27
--------------------

Added
~~~~~
- Utilities to ``metrics.base``: ``get_tensors``, ``keep_scalars``, ``get_scalars``
- New metrics for variables ``VariableValue`` that returns value / global norm of a variable

Changed
~~~~~~~
- ``LastValue`` metric does not store tensor values in a special metric variable.

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Avoid mkdir for HDFS path_model for permissions reasons

Security
~~~~~~~~


[1.2.0] - 2020-05-26
--------------------

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

