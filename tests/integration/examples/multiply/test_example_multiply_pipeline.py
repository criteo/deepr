# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Test Example Pipeline."""

import tensorflow as tf

import deepr
import deepr.examples.multiply as multiply


def test_examples_multiply_pipeline(tmpdir):
    """Test Example Pipeline."""
    # Define paths
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    deepr.io.Path(path_model).mkdir(exist_ok=True, parents=True)
    deepr.io.Path(path_dataset).mkdir(exist_ok=True, parents=True)

    # Define jobs
    build_job = multiply.jobs.Build(path_dataset=f"{path_dataset}/data.tfrecord", num_examples=1000)
    train_spec = deepr.jobs.TrainSpec(max_steps=1000)
    eval_spec = deepr.jobs.EvalSpec(
        throttle_secs=10, start_delay_secs=10, steps=None  # None means "use all the validation set"
    )
    train_metrics = [
        deepr.metrics.StepCounter(name="num_steps"),
        deepr.metrics.DecayMean(tensors=["loss"], decay=0.98),
        deepr.metrics.VariableValue("alpha"),
    ]
    eval_metrics = [deepr.metrics.Mean(tensors=["loss"])]
    final_metrics = [deepr.metrics.Mean(tensors=["loss"])]
    train_hooks = [
        deepr.hooks.LoggingTensorHookFactory(
            name="training",
            functions={
                "memory_gb": deepr.hooks.ResidentMemory(unit="gb"),
                "max_memory_gb": deepr.hooks.MaxResidentMemory(unit="gb"),
            },
            every_n_iter=100,
            use_graphite=False,
            use_mlflow=False,
        ),
        deepr.hooks.SummarySaverHookFactory(save_steps=100),
        deepr.hooks.NumParamsHook(use_mlflow=False),
        deepr.hooks.LogVariablesInitHook(use_mlflow=False),
        deepr.hooks.StepsPerSecHook(
            name="training",
            batch_size=32,
            every_n_steps=100,
            skip_after_step=1000,
            use_mlflow=False,
            use_graphite=False,
        ),
        deepr.hooks.EarlyStoppingHookFactory(
            metric="loss",
            mode="decrease",
            max_steps_without_improvement=100,
            min_steps=500,
            run_every_steps=100,
            final_step=1000,
        ),
    ]
    eval_hooks = [deepr.hooks.LoggingTensorHookFactory(name="validation", at_end=True)]
    final_hooks = [deepr.hooks.LoggingTensorHookFactory(name="final_validation", at_end=True)]
    exporters = [
        deepr.exporters.BestCheckpoint(metric="loss"),
        deepr.exporters.SavedModel(
            path_saved_model=f"{path_model}/saved_model",
            fields=[deepr.utils.Field(name="x", shape=(), dtype="float32")],
        ),
    ]

    trainer_job = deepr.jobs.Trainer(
        path_model=path_model,
        pred_fn=multiply.layers.Multiply(inputs="x", outputs="y_pred"),
        loss_fn=multiply.layers.SquaredL2(inputs=("y", "y_pred"), outputs="loss"),
        optimizer_fn=deepr.optimizers.TensorflowOptimizer(optimizer="Adam", learning_rate=0.1),
        train_input_fn=deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        eval_input_fn=deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        prepro_fn=multiply.prepros.DefaultPrepro(batch_size=32, repeat_size=10),
        train_spec=train_spec,
        eval_spec=eval_spec,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        final_metrics=final_metrics,
        train_hooks=train_hooks,
        eval_hooks=eval_hooks,
        final_hooks=final_hooks,
        exporters=exporters,
    )

    cleanup_checkpoints = deepr.jobs.CleanupCheckpoints(path_model="model")
    optimize_saved_model = deepr.jobs.OptimizeSavedModel(
        path_saved_model=f"{path_model}/saved_model",
        path_optimized_model=f"{path_model}/optimized_saved_model",
        graph_name="_model.pb",
        feeds=["inputs/x"],
        fetch="y_pred",
        new_names={"x": "inputs/x"},
    )
    predict_proto = multiply.jobs.PredictProto(
        path_model=f"{path_model}/optimized_saved_model",
        graph_name="_model.pb",
        input_fn=deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        prepro_fn=multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="inputs/x"),
        feeds="inputs/x",
        fetches="y_pred",
    )
    predict_saved_model = multiply.jobs.PredictSavedModel(
        path_saved_model=f"{path_model}/saved_model",
        input_fn=deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        prepro_fn=multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x"),
        feeds="x",
        fetches="y_pred",
    )

    # Run pipeline
    pipeline = deepr.jobs.Pipeline(
        [build_job, trainer_job, cleanup_checkpoints, optimize_saved_model, predict_proto, predict_saved_model]
    )
    pipeline.run()

    # Test SavedModelPredictor (default)
    input_fn = deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x")
    predictor = deepr.predictors.SavedModelPredictor(
        path=deepr.predictors.get_latest_saved_model(f"{path_model}/saved_model")
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0

    # Test SavedModelPredictor (custom)
    input_fn = deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x")
    predictor = deepr.predictors.SavedModelPredictor(
        path=deepr.predictors.get_latest_saved_model(f"{path_model}/saved_model"), feeds="x", fetches="y_pred"
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0

    # Test ProtoPredictor
    input_fn = deepr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="inputs/x")
    predictor = deepr.predictors.ProtoPredictor(
        path=f"{path_model}/optimized_saved_model/_model.pb", feeds="inputs/x", fetches="y_pred"
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0
