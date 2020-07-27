"""Train Transformer on MovieLens."""

import logging

from fire import Fire

import tensorflow as tf
import deepr as dpr
from deepr.examples import movielens


def main(path):
    """Main entry point of the MovieLens example."""
    build = movielens.jobs.Build(
        path_ratings=path + "/ratings.csv",
        path_train="train.tfrecord.gz",
        path_test="test.tfrecord.gz",
        min_rating=4,
        min_length=5,
        test_ratio=0.2,
        num_negatives=8,
        target_ratio=0.2,
        sample_popularity=True,
        seed=2020)
    train = dpr.jobs.Trainer(
        path_model="model",
        pred_fn=movielens.layers.TransformerModel(vocab_size=200_000, dim=100),
        loss_fn=movielens.layers.BPRLoss(vocab_size=200_000, dim=100),
        optimizer_fn=dpr.optimizers.TensorflowOptimizer("Adam", 0.00001),
        train_input_fn=dpr.readers.TFRecordReader("train.tfrecord.gz"),
        eval_input_fn=dpr.readers.TFRecordReader("test.tfrecord.gz"),
        prepro_fn=movielens.prepros.DefaultPrepro(),
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
    predict = movielens.jobs.Predict(
        path_saved_model="saved_model",
        path_predictions="predictions",
        input_fn=dpr.readers.TFRecordReader("test.tfrecord.gz"),
        prepro_fn=movielens.prepros.DefaultPrepro(),
    )
    evaluate = movielens.jobs.Evaluate(
        path_predictions="predictions",
        path_embeddings="variables/embeddings",
        k=20
    )
    pipeline = dpr.jobs.Pipeline([
        # build,
        train,
        predict,
        evaluate,
    ])
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
