"""Build MovieLens dataset as TFRecords."""

import logging
import random
from typing import List, Dict, Tuple, Callable
from functools import partial
from dataclasses import dataclass

import tensorflow as tf

import deepr as dpr
from deepr.examples.movielens.utils import fields

try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]


@dataclass
class BuildRecords(dpr.jobs.Job):
    """Build MovieLens dataset as TFRecords.

    It aggregates movie ratings by user and build timelines of movies.
    The users are split into train / validation / test sets. Each
    timeline is split in two sub-timelines: one input, one target. For
    each item in the target, n negatives are sampled.

    The resulting tfrecords have the following fields
    - "uid": ()
    - "inputPositives": [size_input]
    - "targetPositives": [size_target]
    - "targetNegatives": [size_target, num_negatives]
    """

    path_ratings: str
    path_mapping: str
    path_train: str
    path_eval: str
    path_test: str
    min_rating: int = 4
    min_length: int = 5
    num_negatives: int = 8
    target_ratio: float = 0.2
    size_test: int = 10_000
    size_eval: int = 10_000
    shuffle_timelines: bool = True
    seed: int = 2020

    def run(self):
        # Read timelines (sorted by timestamp)
        random.seed(self.seed)
        timelines = get_timelines(
            path_ratings=self.path_ratings, min_rating=self.min_rating, min_length=self.min_length
        )

        # Split into train, eval and test
        random.shuffle(timelines)
        timelines_test = timelines[: self.size_test]
        timelines_eval = timelines[self.size_test : self.size_test + self.size_eval]
        timelines_train = timelines[self.size_test + self.size_eval :]

        # Build vocabulary
        LOGGER.info("Building vocabulary of movieId")
        movies = set()
        for _, ids in timelines_train:
            movies.update(ids)
        vocab = sorted(movies)
        mapping = {movie: idx for idx, movie in enumerate(vocab)}
        dpr.io.Path(self.path_mapping).parent.mkdir(parents=True, exist_ok=True)
        dpr.vocab.write(self.path_mapping, [str(movie) for movie in vocab])
        LOGGER.info(f"Number of movies after filtration is: {len(vocab)}")

        # Write datasets
        for timelines, path in zip(
            [timelines_train, timelines_test, timelines_eval], [self.path_train, self.path_test, self.path_eval]
        ):
            dpr.io.Path(path).parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Writing {len(timelines)} timelines to {path}")
            LOGGER.info(f"shuffle_timelines = {self.shuffle_timelines}, num_negatives = {self.num_negatives}")
            write_records(
                partial(
                    records_generator,
                    timelines=timelines,
                    target_ratio=self.target_ratio,
                    num_negatives=self.num_negatives,
                    shuffle_timelines=self.shuffle_timelines,
                    mapping=mapping,
                ),
                path,
            )


def get_timelines(path_ratings: str, min_rating: float, min_length: int) -> List[Tuple[str, List[int]]]:
    """Build timelines from MovieLens Dataset.

    Apply the following filters
        keep movies with ratings > min_rating
        keep users with number of movies > min_length
    """
    # Open path_ratings from HDFS / Local FileSystem
    LOGGER.info(f"Reading ratings from {path_ratings}")
    with dpr.io.Path(path_ratings).open() as file:
        ratings_data = pd.read_csv(file)
    LOGGER.info(f"Number of timelines before filtration is {len(set(ratings_data.userId))}")
    LOGGER.info(f"Number of movies before filtration is {len(set(ratings_data.movieId))}")

    # Group and aggregate ratings per user
    LOGGER.info("Grouping ratings by user")
    ratings_data = ratings_data[ratings_data.rating >= min_rating]
    grouped_data = ratings_data.groupby("userId").agg(list).reset_index()
    grouped_data = grouped_data[grouped_data.rating.map(len) >= min_length]
    LOGGER.info(f"Number of timelines after filtration is {len(grouped_data)}")

    # Sort ratings by timestamp
    LOGGER.info("Building timelines (sort by timestamp).")
    timelines = []
    for _, row in dpr.utils.progress(grouped_data.iterrows(), secs=10):
        uid = str(row.userId)
        movies = [movie for _, movie in sorted(zip(row.timestamp, row.movieId))]
        timelines.append((uid, movies))
    return timelines


def write_records(gen: Callable, path: str):
    """Write records to path."""
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types={field.name: field.dtype for field in FIELDS},
        output_shapes={field.name: field.shape for field in FIELDS},
    )
    to_example = dpr.prepros.ToExample(fields=FIELDS)
    writer = dpr.writers.TFRecordWriter(path=path)
    writer.write(to_example(dataset))


def records_generator(
    timelines: List[Tuple[str, List[int]]],
    target_ratio: float,
    num_negatives: int,
    shuffle_timelines: bool,
    mapping: Dict[int, int],
):
    """Convert Timelines to list of Records with negative samples."""
    for uid, movies in dpr.utils.progress(timelines, secs=10):
        # Remap movies to index and shuffle
        indices = [mapping[movie] for movie in movies if movie in mapping]
        if shuffle_timelines:
            random.shuffle(indices)

        # Split into input and target
        split = int(len(indices) * (1 - target_ratio))
        input_positives = indices[:split]
        target_positives = indices[split:]

        # Sample negatives
        target_negatives = [random.sample(range(len(mapping)), num_negatives) for _ in range(len(target_positives))]

        yield {
            fields.UID.name: uid,
            fields.INPUT_POSITIVES.name: input_positives,
            fields.TARGET_POSITIVES.name: target_positives,
            fields.TARGET_NEGATIVES.name: target_negatives,
        }
