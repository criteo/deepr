Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_.


[2.13.0] - 2023-01-07
---------------------

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
- Fixed CI to make it keep passing on Python 3.6
- Fix mlflow compatibility with mlflow server 1.30.0
Security
~~~~~~~~


[2.12.0] - 2021-04-29
---------------------

Added
~~~~~
- function support for ``Pipeline``

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


[2.11.0] - 2020-12-11
---------------------

Added
~~~~~
- Add new name for ``layers.Sequential`` (now ``layers.DAG`` for Directed Acyclic Graph), kept old name for legacy reasons
- Add new argument options to ``layers.DenseIndex`` (reuse, trainable and initializer)

Changed
~~~~~~~
- Remove shortened imports ``import deepr as dpr`` as it is useless.

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
- Unnecessary logging in metrics tensors' names resolution

Fixed
~~~~~
- Incorrect numpy requirement (incompatible with Tensorflow requirements)

Security
~~~~~~~~

[2.10.0] - 2020-10-28
---------------------
Added
~~~~~
- Log yarn launcher app id with mlflow.


[2.9.1] - 2020-10-01
--------------------

Fixed
~~~~~
- move tf-yarn dependency to gpu / cpu additional packages.


[2.9.0] - 2020-10-01
--------------------

Added
~~~~~
- SVD jobs and configs to MovieLens
- VAE model to MovieLens
- NDCG@K and Recall@K metrics to MovieLens
- CSVReader to MovieLens
- AddWithWeight, DenseIndex, MultiLogLikelihoodCSS, MultiLogLikelihood layers
- Lambda layer to layers.base
- preds option to Trainer

Changed
~~~~~~~


[2.8.2] - 2020-09-07
--------------------

Fixed
~~~~~
- Convert tuple of FileSystems into a List for skein (not cast by cluster_pack).

[2.8.1] - 2020-09-03
--------------------

Fixed
~~~~~
- breaking change in grad norms in Optimizer (store by query name, not variable name)

[2.8.0] - 2020-09-03
--------------------

Added
~~~~~
- Accuracy and AccuracyAtK metrics
- EvaluateJob
- LSTM layer (using FusedOp)
- Recursive .glob() in TFRecordReader for nested directory structures
- Experimental Keras Trainer (converts tf.keras.Model into tf.estimator.Estimator)
- Add run_id parameter to MlFlow macro (to restart a run)
- Refactor to_example logic (make ``arrays_to_example`` importable to convert NumPy arrays to tf.Example)
- Add missing macros module import
- Add batch_shape to Field

Changed
~~~~~~~
- ``__iter__`` method of `Reader` (remove context manager to avoid issue with Keras)
- use skein_launcher instead of in-house implementation using skein.

Fixed
~~~~~
- Use skip_steps when computing gradient norms in Optimizer
- Path copy_file (to support local <> HDFS copies)


[2.7.0] - 2020-08-04
--------------------

Added
~~~~~
- Add encoding support to Path / HDFSFile
- Movielens example
- TripletPrecision layer

Changed
~~~~~~~
- Changed default initializer for embeddings in the embeddings layer

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~

[2.6.0] - 2020-07-02
--------------------

Added
~~~~~
- Add table support in ProtoPredict and ProtoExport


[2.5.1] - 2020-07-01
--------------------

Fixed
~~~~~
- wrong attribute in CopyDir


[2.5.0] - 2020-07-01
--------------------

Added
~~~~~
- Batched support for ``FromExample`` (use ``tf.train.parse_example`` instead of ``parse_single_example``)
- Support ``num_shards_embeddings=None`` in ``utils.save_checkpoint``
- Top1, Top1Max, BPRMax, NCE losses

Changed
~~~~~~~
- Various versions of requirements in ``requirements.txt``

Deprecated
~~~~~~~~~~
Removed
~~~~~~~
Fixed
~~~~~
Security
~~~~~~~~


[2.4.2] - 2020-06-19
--------------------

Added
~~~~~
- Allow user to ignore cpu packages shipped in the pex to Yarn
- URL formatter for MLFlow Macro
- More checks to ``vocab.write`` (check type, newlines)


[2.4.1] - 2020-06-15
--------------------

Fixed
~~~~~
- Fix incorrect inputs / outputs resolution in ``deepr.layers.Select``
- Fix edge cases of ``ToExample`` and ``FromExample`` (on scalars, tensors with dynamic shapes with ndims > 2)


[2.4.0] - 2020-06-10
--------------------

Added
~~~~~
- Utilities to ``metrics.base``: ``get_tensors``, ``keep_scalars``, ``get_scalars``
- New metrics for variables ``VariableValue`` that returns value / global norm of a variable

Changed
~~~~~~~
- ``LastValue`` metric does not store tensor values in a special metric variable.


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

