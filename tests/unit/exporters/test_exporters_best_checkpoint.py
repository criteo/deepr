# pylint: disable=redefined-outer-name
"""Test for exporters.best_checkpoint"""

import dataclasses
from typing import List, Tuple
import re

import tensorflow as tf
import pytest

import deepr
from deepr.exporters.best_checkpoint import read_eval_metrics


@dataclasses.dataclass
class MockEstimator:
    """Mock Estimator for best checkpoint exporter test."""

    model_dir: str

    def __post_init__(self):
        deepr.io.Path(self.model_dir).mkdir()

    def set_step(self, step):
        with deepr.io.Path(self.model_dir, "checkpoint").open("w") as file:
            file.write(f'model_checkpoint_path: "model.ckpt-{step}"')

    def add_checkpoint(self, step):
        deepr.io.write_json({"global_step": step}, f"{self.model_dir}/model.ckpt-{step}.index")
        self.set_step(step)

    def eval_dir(self):
        return self.model_dir

    def get_variable_value(self, _):
        with deepr.io.Path(self.model_dir, "checkpoint").open() as file:
            return int(re.findall(r"-(\d+)", file.read())[0])


def write_summary(name: str, values: List[Tuple[int, float]], logdir):
    """Write summary to directory"""
    tf.reset_default_graph()
    var = tf.Variable(0, dtype=tf.float32)
    summary = tf.summary.scalar(name, var)
    with tf.Session() as sess, tf.summary.FileWriter(logdir) as writer:
        sess.run(var.initializer)
        for step, value in values:
            sess.run(tf.assign(var, value))
            writer.add_summary(sess.run(summary), step)


@pytest.fixture
def eval_dir(tmpdir):
    """Write summary in tmpdir"""
    logdir = str(tmpdir.join("eval_dir"))
    write_summary("loss", [(0, 1.0), (1, 0.5)], logdir)
    return logdir


def test_read_eval_metrics(eval_dir):
    """Test read eval metrics"""
    eval_metrics = read_eval_metrics(eval_dir)
    assert {0: {"loss": 1.0}, 1: {"loss": 0.5}} == eval_metrics


@pytest.mark.parametrize(
    "exporter, step",
    [
        (deepr.exporters.BestCheckpoint(metric="loss"), 20),
        (deepr.exporters.BestCheckpoint(metric="loss", mode="decrease"), 20),
        (deepr.exporters.BestCheckpoint(metric="loss", mode="increase"), 10),
        (deepr.exporters.BestCheckpoint(metric="accuracy", mode="increase"), KeyError),
    ],
)
def test_best_checkpoint(tmpdir, exporter, step):
    """Test BestCheckpoint Exporter"""
    model_dir = str(tmpdir.join("model_dir"))
    estimator = MockEstimator(model_dir)
    estimator.add_checkpoint(10)
    estimator.add_checkpoint(20)
    estimator.add_checkpoint(30)
    write_summary("loss", [(11, 1.0), (21, 0.2), (31, 0.3)], estimator.eval_dir())
    if isinstance(step, int):
        exporter(estimator)
        assert estimator.get_variable_value("step") == step
    else:
        with pytest.raises(step):
            exporter(estimator)
