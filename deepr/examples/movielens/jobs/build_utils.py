"""Build MovieLens utils."""

import logging
from typing import List, Tuple

import deepr as dpr

try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


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
