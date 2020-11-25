# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Tests for optimize_saved_model"""

import logging

import tensorflow as tf

import deepr
import deepr.examples.multiply as multiply


logging.basicConfig(level=logging.INFO)


def test_jobs_optimize_saved_model(tmpdir):
    """Tests for optimize_saved_model"""
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    deepr.io.Path(path_model).mkdir(exist_ok=True, parents=True)
    deepr.io.Path(path_dataset).mkdir(exist_ok=True, parents=True)

    # Build Dataset
    build_job = multiply.jobs.Build(path_dataset=f"{path_dataset}/data.tfrecord", num_examples=1000)
    build_job.run()

    def _gen():
        for idx in range(2):
            yield {"x": idx}

    input_fn = deepr.readers.GeneratorReader(_gen, output_types={"x": tf.int32}, output_shapes={"x": ()})

    def pred_fn(tensors, mode):
        # pylint: disable=unused-argument
        x = tensors["x"]
        table = deepr.utils.table_from_mapping(name="table", mapping={0: 0.0, 1: 1.0})
        alpha = tf.get_variable(name="alpha", shape=())
        return {"y_pred": tf.identity(alpha * table.lookup(x), name="y_pred")}

    def loss_fn(tensors, mode):
        # pylint: disable=unused-argument
        return {"loss": tf.reduce_sum(tensors["y_pred"])}

    # Train model and export as SavedModel
    trainer_job = deepr.jobs.Trainer(
        path_model=path_model,
        pred_fn=pred_fn,
        loss_fn=loss_fn,
        prepro_fn=deepr.prepros.Batch(2),
        optimizer_fn=deepr.optimizers.TensorflowOptimizer(optimizer="Adam", learning_rate=0.1),
        train_input_fn=input_fn,
        eval_input_fn=input_fn,
        exporters=[
            deepr.exporters.SavedModel(
                path_saved_model=f"{path_model}/saved_model",
                fields=[deepr.utils.Field(name="x", shape=(), dtype="int32")],
            )
        ],
    )
    trainer_job.run()

    # Convert SavedModel to one proto file
    optimize_saved_model = deepr.jobs.OptimizeSavedModel(
        path_saved_model=f"{path_model}/saved_model",
        path_optimized_model=f"{path_model}/optimized_saved_model",
        graph_name="_model.pb",
        feeds=["inputs/x"],
        fetch="y_pred",
        new_names={"x": "inputs/x"},
    )
    optimize_saved_model.run()

    # Predict using proto
    def _infer_gen():
        for idx in range(2):
            yield {"inputs/x": idx}

    infer_input_fn = deepr.readers.GeneratorReader(
        _infer_gen, output_types={"inputs/x": tf.int32}, output_shapes={"inputs/x": ()}
    )

    predict_proto = multiply.jobs.PredictProto(
        path_model=f"{path_model}/optimized_saved_model",
        graph_name="_model.pb",
        input_fn=infer_input_fn,
        prepro_fn=deepr.prepros.Batch(2),
        feeds="inputs/x",
        fetches="y_pred",
    )
    predict_proto.run()
