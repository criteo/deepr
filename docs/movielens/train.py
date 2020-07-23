"""Train Transformer on MovieLens."""

import logging

from fire import Fire

import tensorflow as tf
import deepr as dpr
from deepr import example


def main(path):
    """Main entry point of the MovieLens example."""
    path_ratings = path + "/ratings.csv"
    path_movies = path + "/movies.csv"
    job = example.jobs.BuildMovieLens(
        path_ratings=path_ratings,
        path_train="train.tfrecord.gz",
        path_test="test.tfrecord.gz",
        min_rating=4,
        min_length=5,
        test_ratio=0.2,
        num_negatives=8,
        target_ratio=0.2,
        sample_popularity=True,
        seed=2020)
    job.run()
    prepro_fn = example.prepros.MovieLensPrepro()
    train_input_fn = dpr.readers.TFRecordReader("train.tfrecord.gz")
    test_input_fn = dpr.readers.TFRecordReader("test.tfrecord.gz")

    vocab_size = 200_000

    pred_fn = dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives", outputs="inputEmbeddings", variable_name="embeddings", shape=[vocab_size, 100]
        ),
        dpr.layers.Transformer(dim=100),
    )

    loss_fn = dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        dpr.layers.Embedding(
            inputs="targetPositives",
            outputs="targetPositiveEmbeddings",
            variable_name="embeddings",
            shape=[vocab_size, 100],
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="targetNegatives",
            outputs="targetNegativeEmbeddings",
            variable_name="embeddings",
            shape=[vocab_size, 100],
            reuse=True,
        ),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetPositiveEmbeddings"), outputs="targetPositiveLogits"),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetNegativeEmbeddings"), outputs="targetNegativeLogits"),
        dpr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        dpr.layers.ExpandDims(inputs="targetMask", outputs="targetMask", axis=-1),
        dpr.layers.MaskedBPR(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"), outputs="loss"
        ),
    )

    job = dpr.jobs.Trainer(
        path_model="model",
        pred_fn=pred_fn,
        loss_fn=loss_fn,
        optimizer_fn=dpr.optimizers.TensorflowOptimizer("Adam", 0.00001),
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        prepro_fn=prepro_fn,
        train_spec=dpr.jobs.TrainSpec(max_steps=10),
        eval_spec=dpr.jobs.EvalSpec(steps=10),
        final_spec=dpr.jobs.FinalSpec(steps=10),
        exporters=[
            dpr.exporters.SaveVariables(
                path_variables="variables",
                variable_names=["embeddings"]
            ),
            dpr.exporters.SavedModel(
                path_saved_model="saved_model",
                fields=[
                    dpr.Field(name="inputPositives", shape=[None], dtype=tf.int64),
                    dpr.Field(name="inputMask", shape=[None], dtype=tf.bool)
                ]
            ),
        ]
    )
    job.run()
    job = example.jobs.PredictMovieLens(
        path_saved_model="saved_model",
        path_predictions="predictions",
        input_fn=test_input_fn,
        prepro_fn=prepro_fn,
    )
    job.run()
    job = example.jobs.EvalMovieLens(
        path_predictions="predictions",
        path_embeddings="variables/embeddings",
        k=20
    )
    job.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
