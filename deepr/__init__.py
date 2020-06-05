# pylint: disable=unused-import,missing-docstring,wrong-import-position

import tensorflow as tf

import deepr.cli
import deepr.config
import deepr.exporters
import deepr.hooks
import deepr.initializers
import deepr.io
import deepr.jobs
import deepr.layers
import deepr.metrics
import deepr.optimizers
import deepr.predictors
import deepr.prepros
import deepr.readers
import deepr.utils
import deepr.vocab
import deepr.writers
from deepr.config import parse_config, from_config
from deepr.utils import Field
from deepr.version import __author__, __version__


# Make ModeKeys accessible via dpr.TRAIN, dpr.EVAL and dpr.PREDICT
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT
