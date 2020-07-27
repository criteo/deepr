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
    path_root = "transformer"
    path_model = path_root + "/model"
    path_data = path_root + "/data"
    path_variables = path_root + "/variables"
    path_predictions = path_root + "/predictions"
    path_saved_model = path_root + "/saved_model"
    path_train = path_data + "/train.tfrecord.gz"
    path_test = path_data + "/test.tfrecord.gz"
    dpr.io.Path(path_root).mkdir(exist_ok=True)
    dpr.io.Path(path_model).mkdir(exist_ok=True)
    dpr.io.Path(path_data).mkdir(exist_ok=True)

    build = movielens.jobs.Build(
        path_ratings=path_ratings,
        path_train=path_train,
        path_test=path_test,
        min_rating=4,
        min_length=5,
        test_ratio=0.2,
        num_negatives=8,
        target_ratio=0.2,
        sample_popularity=True,
        seed=2020)

    transformer_model = movielens.layers.TransformerModel(
        vocab_size=150_000,
        dim=1000,
        encoding_blocks=2,
        num_heads=8,
        dim_head=32,
        use_layer_normalization=True,
        event_dropout_rate=0.4,
        ff_dropout_rate=0.5,
        residual_connection=False,
        scale=False,
        use_positional_encoding=False,
        use_look_ahead_mask=False,
    )

    train = dpr.jobs.Trainer(
        path_model=path_model,
        pred_fn=transformer_model,
        loss_fn=movielens.layers.BPRLoss(vocab_size=150_000, dim=1000),
        optimizer_fn=dpr.optimizers.TensorflowOptimizer("LazyAdam", 0.0001),
        train_input_fn=dpr.readers.TFRecordReader(path_train),
        eval_input_fn=dpr.readers.TFRecordReader(path_test),
        prepro_fn=movielens.prepros.DefaultPrepro(
            batch_size=128,
            max_input_size=50,
        ),
        train_spec=dpr.jobs.TrainSpec(max_steps=10_000),
        eval_spec=dpr.jobs.EvalSpec(steps=100),
        final_spec=dpr.jobs.FinalSpec(steps=None),
        exporters=[
            dpr.exporters.SaveVariables(
                path_variables=path_variables,
                variable_names=["embeddings"]
            ),
            dpr.exporters.SavedModel(
                path_saved_model=path_saved_model,
                fields=[
                    dpr.Field(name="inputPositives", shape=[None], dtype=tf.int64),
                    dpr.Field(name="inputMask", shape=[None], dtype=tf.bool)
                ]
            ),
        ]
    )
    predict = movielens.jobs.Predict(
        path_saved_model=path_saved_model,
        path_predictions=path_predictions,
        input_fn=dpr.readers.TFRecordReader(path_test),
        prepro_fn=movielens.prepros.DefaultPrepro(),
    )
    evaluate = movielens.jobs.Evaluate(
        path_predictions=path_predictions,
        path_embeddings=path_variables + "/embeddings",
        k=20
    )
    pipeline = dpr.jobs.Pipeline([
        build,
        train,
        predict,
        evaluate,
    ])
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
