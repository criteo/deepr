"""CSV Reader for MovieLens."""

import random
import json

import pandas as pd
import tensorflow as tf

import deepr as dpr

from deepr.examples.movielens.utils import fields


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]


class CSVReader(dpr.readers.Reader):
    """CSV Reader for MovieLens."""

    def __init__(
        self,
        path_csv: str,
        path_mapping: str,
        target_ratio: float = 0.2,
        num_negatives: int = 8,
        next_movie: int = None,
        num_shuffle: int = None,
    ):
        self.path_csv = path_csv
        self.path_mapping = path_mapping
        self.target_ratio = target_ratio
        self.num_negatives = num_negatives
        self.next_movie = next_movie
        self.num_shuffle = num_shuffle

    def as_dataset(self):
        # Read vocabulary and movies
        with dpr.io.Path(self.path_csv).open() as file:
            df = pd.read_csv(file)
            df.movieIds = df.movieIds.apply(json.loads)
        vocab = dpr.vocab.read(self.path_mapping)
        mapping = {int(movie): idx for idx, movie in enumerate(vocab)}
        sample_indices = [idx for movie, idx in mapping.items() if movie != self.next_movie]

        def _to_record(uid, indices):
            split = int(len(indices) * (1 - self.target_ratio))
            input_positives = indices[:split]
            target_positives = indices[split:]
            if self.next_movie:
                input_positives.append(mapping[self.next_movie])
            return {
                "uid": uid,
                "inputPositives": input_positives,
                "targetPositives": target_positives,
                "targetNegatives": [
                    random.sample(sample_indices, self.num_negatives) for _ in range(len(target_positives))
                ],
            }

        def _gen():
            for _, row in df.iterrows():
                indices = [mapping[int(movie)] for movie in row.movieIds if int(movie) in mapping]
                if self.num_shuffle is not None:
                    for _ in range(self.num_shuffle):
                        random.shuffle(indices)
                        yield _to_record(row.uid, indices)
                else:
                    yield _to_record(row.uid, indices)

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in FIELDS},
            output_shapes={field.name: field.shape for field in FIELDS},
        )
