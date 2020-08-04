"""Macro that generates new path_model based on date."""

import datetime


class Paths(dict):
    """Macro that generates new path_model based on date."""

    def __init__(self, path_ratings: str, path_root: str = None, run_name: str = None):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if path_root is None:
            path_root = f"viewfs://root/user/deepr/dev/movielens/models/{now}"
        if run_name is None:
            run_name = f"movielens.{now}"
        path_model = f"{path_root}/model"
        path_mapping = f"{path_root}/data/mapping.txt"
        path_train = f"{path_root}/data/train.tfrecord.gz"
        path_eval = f"{path_root}/data/eval.tfrecord.gz"
        path_test = f"{path_root}/data/test.tfrecord.gz"
        path_variables = f"{path_root}/variables"
        path_embeddings = f"{path_variables}/embeddings"
        path_biases = f"{path_variables}/biases"
        path_saved_model = f"{path_root}/saved_model"
        path_predictions = f"{path_root}/predictions.parquet.snappy"
        super().__init__(
            path_ratings=path_ratings,
            run_name=run_name,
            path_root=path_root,
            path_model=path_model,
            path_mapping=path_mapping,
            path_train=path_train,
            path_eval=path_eval,
            path_test=path_test,
            path_variables=path_variables,
            path_embeddings=path_embeddings,
            path_biases=path_biases,
            path_saved_model=path_saved_model,
            path_predictions=path_predictions,
        )
