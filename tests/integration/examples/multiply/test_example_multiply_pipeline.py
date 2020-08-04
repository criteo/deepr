# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Test Example Pipeline."""

import tensorflow as tf

import deepr as dpr
import deepr.examples.multiply as multiply


def test_examples_multiply_pipeline(tmpdir):
    """Test Example Pipeline."""
    # Define paths
    path_model = str(tmpdir.join("model"))
    path_dataset = str(tmpdir.join("dataset"))
    dpr.io.Path(path_model).mkdir(exist_ok=True, parents=True)
    dpr.io.Path(path_dataset).mkdir(exist_ok=True, parents=True)

    # Define jobs
    build_job = multiply.jobs.Build(path_dataset=f"{path_dataset}/data.tfrecord", num_examples=1000)
    train_spec = dpr.jobs.TrainSpec(max_steps=1000)
    eval_spec = dpr.jobs.EvalSpec(
        throttle_secs=10, start_delay_secs=10, steps=None  # None means "use all the validation set"
    )
    train_metrics = [
        dpr.metrics.StepCounter(name="num_steps"),
        dpr.metrics.DecayMean(tensors=["loss"], decay=0.98),
        dpr.metrics.VariableValue("alpha"),
    ]
    eval_metrics = [dpr.metrics.Mean(tensors=["loss"])]
    final_metrics = [dpr.metrics.Mean(tensors=["loss"])]
    train_hooks = [
        dpr.hooks.LoggingTensorHookFactory(
            name="training",
            functions={
                "memory_gb": dpr.hooks.ResidentMemory(unit="gb"),
                "max_memory_gb": dpr.hooks.MaxResidentMemory(unit="gb"),
            },
            every_n_iter=100,
            use_graphite=False,
            use_mlflow=False,
        ),
        dpr.hooks.SummarySaverHookFactory(save_steps=100),
        dpr.hooks.NumParamsHook(use_mlflow=False),
        dpr.hooks.LogVariablesInitHook(use_mlflow=False),
        dpr.hooks.StepsPerSecHook(
            name="training",
            batch_size=32,
            every_n_steps=100,
            skip_after_step=1000,
            use_mlflow=False,
            use_graphite=False,
        ),
        dpr.hooks.EarlyStoppingHookFactory(
            metric="loss",
            mode="decrease",
            max_steps_without_improvement=100,
            min_steps=500,
            run_every_steps=100,
            final_step=1000,
        ),
    ]
    eval_hooks = [dpr.hooks.LoggingTensorHookFactory(name="validation", at_end=True)]
    final_hooks = [dpr.hooks.LoggingTensorHookFactory(name="final_validation", at_end=True)]
    exporters = [
        dpr.exporters.BestCheckpoint(metric="loss"),
        dpr.exporters.SavedModel(
            path_saved_model=f"{path_model}/saved_model", fields=[dpr.utils.Field(name="x", shape=(), dtype="float32")]
        ),
    ]

    trainer_job = dpr.jobs.Trainer(
        path_model=path_model,
        pred_fn=multiply.layers.Multiply(inputs="x", outputs="y_pred"),
        loss_fn=multiply.layers.SquaredL2(inputs=("y", "y_pred"), outputs="loss"),
        optimizer_fn=dpr.optimizers.TensorflowOptimizer(optimizer="Adam", learning_rate=0.1),
        train_input_fn=dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        eval_input_fn=dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
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

    cleanup_checkpoints = dpr.jobs.CleanupCheckpoints(path_model="model")
    optimize_saved_model = dpr.jobs.OptimizeSavedModel(
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
        input_fn=dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        prepro_fn=multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="inputs/x"),
        feeds="inputs/x",
        fetches="y_pred",
    )
    predict_saved_model = multiply.jobs.PredictSavedModel(
        path_saved_model=f"{path_model}/saved_model",
        input_fn=dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord"),
        prepro_fn=multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x"),
        feeds="x",
        fetches="y_pred",
    )

    # Run pipeline
    pipeline = dpr.jobs.Pipeline(
        [build_job, trainer_job, cleanup_checkpoints, optimize_saved_model, predict_proto, predict_saved_model]
    )
    pipeline.run()

    # Test SavedModelPredictor (default)
    input_fn = dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x")
    predictor = dpr.predictors.SavedModelPredictor(
        path=dpr.predictors.get_latest_saved_model(f"{path_model}/saved_model")
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0

    # Test SavedModelPredictor (custom)
    input_fn = dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="x")
    predictor = dpr.predictors.SavedModelPredictor(
        path=dpr.predictors.get_latest_saved_model(f"{path_model}/saved_model"), feeds="x", fetches="y_pred"
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0

    # Test ProtoPredictor
    input_fn = dpr.readers.TFRecordReader(path=f"{path_dataset}/data.tfrecord")
    prepro_fn = multiply.prepros.InferencePrepro(batch_size=1, count=2, inputs="inputs/x")
    predictor = dpr.predictors.ProtoPredictor(
        path=f"{path_model}/optimized_saved_model/_model.pb", feeds="inputs/x", fetches="y_pred"
    )
    idx = 0
    for idx, preds in enumerate(predictor(lambda: prepro_fn(input_fn(), tf.estimator.ModeKeys.PREDICT))):
        assert isinstance(preds, dict)
        assert "y_pred" in preds
    assert idx > 0
