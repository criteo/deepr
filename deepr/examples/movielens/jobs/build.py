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
    path_mapping: str
    path_train: str
    path_test: str
    min_rating: int = 4
    min_length: int = 5
    test_ratio: float = 0.2
    num_negatives: int = 8
    target_ratio: float = 0.2
    sample_popularity: bool = False
    seed: int = 2020

    def run(self):
        # Build records
        random.seed(self.seed)
        timelines, mapping = get_timelines(
            path_ratings=self.path_ratings, min_rating=self.min_rating, min_length=self.min_length
        )
        records = timelines_to_records(
            timelines=timelines,
            target_ratio=self.target_ratio,
            num_negatives=self.num_negatives,
            sample_popularity=self.sample_popularity,
        )

        # Write mapping
        reverse_mapping = {idx: movie for movie, idx in mapping.items()}
        vocab = [str(reverse_mapping[idx]) for idx in range(len(reverse_mapping))]
        dpr.vocab.write(self.path_mapping, vocab)

        # Shuffle and split
        random.shuffle(records)
        delimiter = int(len(records) * self.test_ratio)
        write_records(records[delimiter:], self.path_train)
        write_records(records[:delimiter], self.path_test)


def write_records(records: List[Dict], path: str):
    """Write records to path."""
    LOGGER.info(f"Writing {len(records)} records to {path}")
    dataset = tf.data.Dataset.from_generator(
        lambda: (record for record in records),
        output_types={field.name: field.dtype for field in FIELDS},
        output_shapes={field.name: field.shape for field in FIELDS},
    )
    to_example = dpr.prepros.ToExample(fields=FIELDS)
    writer = dpr.writers.TFRecordWriter(path=path)
    writer.write(to_example(dataset))


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
    LOGGER.info(f"Number of timelines before filtration is {len(set(ratings_data.userId))}")
    LOGGER.info(f"Number of movies before filtration is {len(set(ratings_data.movieId))}")

    # Group and aggregate ratings per user
    LOGGER.info("Grouping ratings by user")
    ratings_data = ratings_data[ratings_data.rating >= min_rating]
    grouped_data = ratings_data.groupby("userId").agg(list).reset_index()
    grouped_data = grouped_data[grouped_data.rating.map(len) >= min_length]
    LOGGER.info(f"Number of timelines after filtration is {len(grouped_data)}")

    # Build Mapping
    LOGGER.info("Building mapping movieId -> index")
    movies = set()
    for ids in grouped_data["movieId"]:
        movies.update(ids)
    mapping = {movie: idx for idx, movie in enumerate(sorted(movies))}
    LOGGER.info(f"Number of movies after filtration is: {len(mapping)}")

    # Sort ratings by timestamp
    LOGGER.info("Building timelines (sort by timestamp).")
    timelines = []
    for _, row in dpr.utils.progress(grouped_data.iterrows(), secs=10):
        uid = str(row.userId)
        movies = [mapping[movie] for _, movie in sorted(zip(row.timestamp, row.movieId))]
        timelines.append((uid, movies))
    return timelines, mapping


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
    LOGGER.info(f"Pre-sampling negatives with replacement, sample_popularity={sample_popularity}.")
    movie_counts = Counter()  # type: Dict[str, int]
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
            negatives[offset + idx * num_negatives : offset + (idx + 1) * num_negatives]
            for idx in range(len(target_positives))
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
