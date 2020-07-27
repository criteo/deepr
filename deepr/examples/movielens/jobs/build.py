"""Build dataset."""

import logging
import random
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import tensorflow as tf

import deepr as dpr
from deepr.examples.movielens.utils import fields


LOGGER = logging.getLogger(__name__)


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]


@dataclass
class Build(dpr.jobs.Job):
    """Build MovieLens dataset."""

    path_ratings: str
    path_train: str
    path_test: str
    min_rating: int = 4
    min_length: int = 5
    test_ratio: float = 0.2
    num_negatives: int = 8
    target_ratio: float = 0.2
    sample_popularity: bool = True
    seed: int = 2020

    def run(self):
        # Build records
        random.seed(self.seed)
        timelines = get_timelines(
            path_ratings=self.path_ratings, min_rating=self.min_rating, min_length=self.min_length
        )
        records = timelines_to_records(
            timelines=timelines,
            target_ratio=self.target_ratio,
            num_negatives=self.num_negatives,
            sample_popularity=self.sample_popularity,
        )

        # Shuffle and split
        random.shuffle(records)
        delimiter = int(len(records) * (1 - self.test_ratio))

        # Write train dataset
        LOGGER.info(f"Writing train dataset to {self.path_train}")
        train_dataset = tf.data.Dataset.from_generator(
            lambda: (record for record in records[:delimiter]),
            output_types={field.name: field.dtype for field in FIELDS},
            output_shapes={field.name: field.shape for field in FIELDS},
        )
        to_example = dpr.prepros.ToExample(fields=FIELDS)
        writer = dpr.writers.TFRecordWriter(path=self.path_train)
        writer.write(to_example(train_dataset))

        # Write test dataset
        LOGGER.info(f"Writing test dataset to {self.path_test}")
        test_dataset = tf.data.Dataset.from_generator(
            lambda: (record for record in records[delimiter:]),
            output_types={field.name: field.dtype for field in FIELDS},
            output_shapes={field.name: field.shape for field in FIELDS},
        )
        writer = dpr.writers.TFRecordWriter(path=self.path_test)
        writer.write(to_example(test_dataset))


def get_timelines(path_ratings: str, min_rating: float, min_length: int) -> List[Tuple[str, List[int]]]:
    """Build timelines from MovieLens Dataset.

    Apply the following filters
        keep movies with ratings > min_rating
        keep users with number of movies > min_length
    """
    # Open path_ratings from HDFS / Local FileSystem
    LOGGER.info(f"Reading ratings from {path_ratings}")
    with dpr.io.Path(path_ratings).open() as file:
        ratings_data = pd.read_csv(file, sep=",")

    # Group and aggregate ratings per user
    LOGGER.info("Grouping ratings by user")
    ratings_data = ratings_data[ratings_data.rating >= min_rating]
    grouped_data = ratings_data.groupby("userId").agg(list).reset_index()
    grouped_data = grouped_data[grouped_data.rating.map(len) >= min_length]

    # Sort ratings by timestamp
    LOGGER.info("Building timelines (sort by timestamp).")
    timelines = []
    for _, row in dpr.utils.progress(grouped_data.iterrows(), secs=10):
        uid = str(row.userId)
        movie_ids = [movie_id for _, movie_id in sorted(zip(row.timestamp, row.movieId))]
        timelines.append((uid, movie_ids))
    return timelines


def timelines_to_records(
    timelines: List[Tuple[str, List[int]]], target_ratio: float, num_negatives: int, sample_popularity: bool
) -> List[Dict]:
    """Convert Timelines to list of Records with negative samples."""
    # Split timelines by ratio
    LOGGER.info("Splitting timelines.")
    splitted_timelines = []
    for uid, movie_ids in dpr.utils.progress(timelines, secs=10):
        start_target_index = int(len(movie_ids) * (1 - target_ratio))
        input_positives = movie_ids[:start_target_index]
        target_positives = movie_ids[start_target_index:]
        splitted_timelines.append((uid, input_positives, target_positives))

    # Pre-sample with replacement enough negatives for all examples
    LOGGER.info("Pre-sampling negatives with replacement.")
    movie_counts = Counter()
    for _, movie_ids in timelines:
        movie_counts.update(movie_ids)
    movies, counts = zip(*movie_counts.items())
    negatives = random.choices(
        movies,
        counts if sample_popularity else None,
        k=sum(len(tgt) for _, _, tgt in splitted_timelines) * num_negatives,
    )

    # Slice negatives from negatives and build record
    LOGGER.info("Creating records with negatives.")
    records, offset = [], 0
    for uid, input_positives, target_positives in dpr.utils.progress(splitted_timelines, secs=10):
        # Extract slice from negatives
        target_negatives = [
            negatives[offset + idx : offset + idx + num_negatives]
            for idx in range(0, len(target_positives) * num_negatives, num_negatives)
        ]

        # Update slice offset for negatives, create new record
        offset += len(target_positives) * num_negatives
        record = {
            fields.UID.name: uid,
            fields.INPUT_POSITIVES.name: input_positives,
            fields.TARGET_POSITIVES.name: target_positives,
            fields.TARGET_NEGATIVES.name: target_negatives,
        }
        records.append(record)
    return records