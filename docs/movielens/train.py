"""Train Transformer on MovieLens.

Usage
-----
python train.py /path/to/ml-20m/ratings.csv
"""

import logging

from fire import Fire

import tensorflow as tf
import deepr as dpr
from deepr.examples import movielens


def main(path_ratings: str):
    """Main entry point of the MovieLens example.

    Parameters
    ----------
    path : str
        Path to ML20 dataset ratings
        Link https://grouplens.org/datasets/movielens/20m/
    """
    path_root = "transformer_next_shuffle"
    path_model = path_root + "/model"
    path_data = path_root + "/data"
    path_variables = path_root + "/variables"
    path_predictions = path_root + "/predictions.parquet.snappy"
    path_saved_model = path_root + "/saved_model"
    path_mapping = path_data + "/mapping.txt"
    path_train = path_data + "/train.csv"
    path_eval = path_data + "/eval.csv"
    path_test = path_data + "/test.csv"
    dpr.io.Path(path_root).mkdir(exist_ok=True)
    dpr.io.Path(path_model).mkdir(exist_ok=True)
    dpr.io.Path(path_data).mkdir(exist_ok=True)
    max_steps = 100_000

    build = movielens.jobs.BuildCSV(
        path_ratings=path_ratings,
        path_mapping=path_mapping,
        path_train=path_train,
        path_eval=path_eval,
        path_test=path_test,
        min_rating=4,
        min_length=5,
        size_test=10_000,
        size_eval=10_000,
        seed=2020,
        next_movie=100_000,
    )
    build.run()
    vocab_size = dpr.vocab.size(path_mapping)

    transformer_model = movielens.layers.TransformerModel(
        vocab_size=vocab_size,
        dim=100,
        encoding_blocks=2,
        num_heads=8,
        dim_head=32,
        residual_connection=False,
        use_layer_normalization=True,
        use_feedforward=True,
        event_dropout_rate=0.4,
        ff_dropout_rate=0.5,
        ff_normalization=True,
        scale=False,
        use_positional_encoding=False,
        use_look_ahead_mask=False,
    )

    average_model = movielens.layers.AverageModel(vocab_size=vocab_size, dim=100)

    train = dpr.jobs.Trainer(
        path_model=path_model,
        pred_fn=transformer_model,
        loss_fn=movielens.layers.BPRLoss(vocab_size=vocab_size, dim=100),
        optimizer_fn=dpr.optimizers.TensorflowOptimizer("LazyAdam", 0.0001),
        train_input_fn=movielens.readers.CSVReader(
            path_csv=path_train,
            path_mapping=path_mapping,
            target_ratio=0.2,
            num_negatives=8,
            next_movie=100_000,
            num_shuffle=10,
        ),
        eval_input_fn=movielens.readers.CSVReader(
            path_csv=path_eval,
            path_mapping=path_mapping,
            target_ratio=0.2,
            num_negatives=8,
            next_movie=100_000,
            num_shuffle=1,
        ),
        prepro_fn=movielens.prepros.CSVPrepro(
            min_input_size=3,
            min_target_size=3,
            max_input_size=50,
            max_target_size=50,
            buffer_size=1024,
            batch_size=128,
            repeat_size=None,
            prefetch_size=1,
            num_parallel_calls=8,
        ),
        train_spec=dpr.jobs.TrainSpec(max_steps=max_steps),
        eval_spec=dpr.jobs.EvalSpec(steps=None, start_delay_secs=30, throttle_secs=30),
        final_spec=dpr.jobs.FinalSpec(steps=None),
        exporters=[
            dpr.exporters.BestCheckpoint(metric="triplet_precision", mode="increase"),
            dpr.exporters.SaveVariables(path_variables=path_variables, variable_names=["biases", "embeddings"]),
            dpr.exporters.SavedModel(
                path_saved_model=path_saved_model,
                fields=[
                    dpr.Field(name="inputPositives", shape=(None,), dtype=tf.int64),
                    dpr.Field(name="inputMask", shape=(None,), dtype=tf.bool),
                ],
            ),
        ],
        train_hooks=[
            dpr.hooks.LoggingTensorHookFactory(
                name="training",
                functions={
                    "memory_gb": dpr.hooks.ResidentMemory(unit="gb"),
                    "max_memory_gb": dpr.hooks.MaxResidentMemory(unit="gb"),
                },
                every_n_iter=300,
                use_graphite=False,
                use_mlflow=False,
            ),
            dpr.hooks.SummarySaverHookFactory(save_steps=300),
            dpr.hooks.NumParamsHook(use_mlflow=False),
            dpr.hooks.LogVariablesInitHook(use_mlflow=False),
            dpr.hooks.StepsPerSecHook(
                name="training",
                batch_size=128,
                every_n_steps=300,
                skip_after_step=max_steps,
                use_mlflow=False,
                use_graphite=False,
            ),
            dpr.hooks.EarlyStoppingHookFactory(
                metric="triplet_precision",
                mode="increase",
                max_steps_without_improvement=1000,
                min_steps=5_000,
                run_every_steps=300,
                final_step=max_steps,
            ),
        ],
        eval_hooks=[dpr.hooks.LoggingTensorHookFactory(name="validation", at_end=True)],
        final_hooks=[dpr.hooks.LoggingTensorHookFactory(name="final_validation", at_end=True)],
        train_metrics=[dpr.metrics.StepCounter(name="num_steps"), dpr.metrics.DecayMean(tensors=["loss"], decay=0.98)],
        eval_metrics=[dpr.metrics.Mean(tensors=["loss", "triplet_precision"])],
        final_metrics=[dpr.metrics.Mean(tensors=["loss", "triplet_precision"])],
        run_config=dpr.jobs.RunConfig(
            save_checkpoints_steps=300, save_summary_steps=300, keep_checkpoint_max=None, log_step_count_steps=300
        ),
        config_proto=dpr.jobs.ConfigProto(
            inter_op_parallelism_threads=8, intra_op_parallelism_threads=8, gpu_device_count=0, cpu_device_count=48,
        ),
    )
    predict = movielens.jobs.Predict(
        path_saved_model=path_saved_model,
        path_predictions=path_predictions,
        input_fn=movielens.readers.CSVReader(
            path_csv=path_test,
            path_mapping=path_mapping,
            target_ratio=0.2,
            num_negatives=8,
            next_movie=100_000,
            num_shuffle=1,
        ),
        prepro_fn=movielens.prepros.CSVPrepro(),
    )
    evaluate = [
        movielens.jobs.Evaluate(
            path_predictions=path_predictions,
            path_embeddings=path_variables + "/embeddings",
            path_biases=path_variables + "/biases",
            k=k,
        )
        for k in [10, 20, 50]
    ]
    pipeline = dpr.jobs.Pipeline([train, predict] + evaluate)
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
