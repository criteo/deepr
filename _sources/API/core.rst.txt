DeepR
=====

.. _config_module:
Config
------

The configuration module makes it possible to configure any object.

.. currentmodule:: deepr.config

.. autosummary::
   :toctree: _autosummary

   assert_no_macros
   fill_macros
   fill_references
   from_config
   ismacro
   isreference
   parse_config


Exporter
--------

Exporters run at the end of training.

.. currentmodule:: deepr.exporters

.. autosummary::
   :toctree: _autosummary

   Exporter
   BestCheckpoint
   SaveVariables
   SavedModel

Hook
-----

Hooks are called regularly during training to send some information to another service.

.. currentmodule:: deepr.hooks

.. autosummary::
   :toctree: _autosummary

   EarlyStoppingHookFactory
   EstimatorHookFactory
   LoggingTensorHookFactory
   MaxResidentMemory
   ResidentMemory
   StepsPerSecHook
   SummarySaverHookFactory
   TensorHookFactory

Initializer
-----------

Initializers run before training.

.. currentmodule:: deepr.initializers

.. autosummary::
   :toctree: _autosummary

   CheckpointInitializer

Io
--

Io provides helpers to read/write from file systems.


.. currentmodule:: deepr.io

.. autosummary::
   :toctree: _autosummary

   HDFSFileSystem
   ParquetDataset
   Path
   read_json

Job
---

Jobs are the programs that will actually run, they are composable through the pipeline job, the yarn launcher and trainer job.

.. currentmodule:: deepr.jobs

.. autosummary::
   :toctree: _autosummary

   Job
   LogMetric
   MLFlowSaveConfigs
   MLFlowSaveInfo
   OptimizeSavedModel
   Pipeline
   Trainer
   YarnLauncher
   YarnTrainer

Layer
-----

Tensorflow logic is preferably defined in a `Layer` for re-usability and composability. It is the equivalent of `Keras`, `Trax`, etc. layers.
It takes as input / returns a dictionary of `tf.Tensor`. This means that the `__init__` method of a `Layer` must define which keys are used for inputs / outputs.

.. currentmodule:: deepr.layers

.. autosummary::
   :toctree: _autosummary

   Add
   Average
   BPR
   BooleanMask
   ClickRank
   Concat
   Dense
   DotProduct
   Embedding
   Equal
   Identity
   IsMinSize
   Layer
   LogicalAnd
   Lookup
   LookupFromFile
   LookupFromMapping
   MaskedBPR
   NotEqual
   Parallel
   Product
   Rename
   Select
   Sequential
   Slice
   SliceFirst
   SliceLast
   StringJoin
   Sum
   ToDense
   WeightedAverage

Macros
-------

Macros are subclasses of dictionaries that dynamically create params for configs.

.. currentmodule:: deepr.macros

.. autosummary::
   :toctree: _autosummary

   MLFlowInit


Metrics
-------

Metrics compute training and validation information during training.

.. currentmodule:: deepr.metrics

.. autosummary::
   :toctree: _autosummary

   DecayMean
   FiniteMean
   LastValue
   MaxValue
   Mean
   Metric
   StepCounter

Optimizer
---------

Optimizer is the way to optimize your graph.

.. currentmodule:: deepr.optimizers

.. autosummary::
   :toctree: _autosummary

   Optimizer
   TensorflowOptimizer

Prepro
------

The `Prepro` classes are utilities to transform `tf.data.Dataset`.

The most common way to define a `Prepro` is to wrap a `Layer` with a `MapLayer` or `FilterLayer` transform.

.. currentmodule:: deepr.prepros

.. autosummary::
   :toctree: _autosummary

   Batch
   Filter
   Map
   PaddedBatch
   Prefetch
   Prepro
   Repeat
   Serial
   Shuffle
   TFRecordSequenceExample
   Take



Reader
------

A `Reader` is the equivalent of `tensorflow_dataset` readers. Their `__init__` method defines all the parameters necessary to create a `tf.data.Dataset`.

.. currentmodule:: deepr.readers

.. autosummary::
   :toctree: _autosummary

   GeneratorReader
   Reader
   TFRecordReader

Utils
-----

Various functions

.. currentmodule:: deepr.utils

.. autosummary::
   :toctree: _autosummary

   Field
   GraphiteClient
   TensorType
   dict_to_item
   get_feedable_tensors
   get_fetchable_tensors
   handle_exceptions
   import_graph_def
   item_to_dict
   make_same_shape
   msb_lsb_to_str
   save_variables_in_ckpt
   str_to_msb_lsb
   to_flat_tuple
