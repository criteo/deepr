"""Build MovieLens dataset as CSV files."""

import logging
import random
from dataclasses import dataclass

import deepr as dpr
from deepr.examples.movielens.utils import fields

from .build_utils import get_timelines

try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]


@dataclass
class BuildCSV(dpr.jobs.Job):
    """Build MovieLens dataset as CSV files."""

    path_ratings: str
    path_mapping: str
    path_train: str
    path_eval: str
    path_test: str
    min_rating: int = 4
    min_length: int = 5
    size_test: int = 10_000
    size_eval: int = 10_000
    seed: int = 2020
    next_movie: int = 100_000

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

        # Build vocabulary on train
        movies = set()
        for _, ids in timelines_train:
            movies.update(ids)
        vocab = sorted(movies)

        # Add special id for a special <next> movie
        if self.next_movie is not None:
            if self.next_movie in vocab:
                raise ValueError(f"{self.next_movie} already in vocab (try {max(vocab) + 1} ?)")
            vocab = [self.next_movie] + vocab

        # Write vocab
        dpr.vocab.write(self.path_mapping, [str(movie) for movie in vocab])

        # Write timelines
        with dpr.io.Path(self.path_train).open("w") as file:
            df = pd.DataFrame(timelines_train, columns=["uid", "movieIds"])
            df.to_csv(file, index=False)
        with dpr.io.Path(self.path_eval).open("w") as file:
            df = pd.DataFrame(timelines_eval, columns=["uid", "movieIds"])
            df.to_csv(file, index=False)
        with dpr.io.Path(self.path_test).open("w") as file:
            df = pd.DataFrame(timelines_test, columns=["uid", "movieIds"])
            df.to_csv(file, index=False)
