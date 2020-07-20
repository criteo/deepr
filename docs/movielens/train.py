"""Train Transformer on MovieLens."""

import logging

from fire import Fire

import deepr as dpr
from deepr import example


def main(path_ratings: str):
    """Main entry point of the MovieLens example."""
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

    # Trainer
    prepro_fn = example.prepros.MovieLensPrepro()
    train_input_fn = dpr.readers.TFRecordReader()
    test_input_fn = dpr.readers.TFRecordReader()
    # TODO: need to embed inputs
    pred_fn = dpr.layers.TransformerModel(vocab_size=200_000, train_embeddings=True, dim=100)

    # TODO: need to embed target
    loss_fn = dpr.layers.MaskedBPR()
    optimizer_fn = dpr.optimizers.TensorflowOptimizer("Adam", 0.00001)
    job = dpr.jobs.Trainer(
        path_model="model",
        pred_fn=pred_fn,
        loss_fn=loss_fn,
        optimizer_fn=optimizer_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        prepro_fn=prepro_fn,
        # TODO: add SavedModelExporter
        # TODO: export movie embeddings
    )

    # Predict user embeddings
    # TODO: use PredictSavedModel (maybe move this job out of example)

    # Evaluate user embeddings
    job = example.jobs.EvalMovieLens()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
