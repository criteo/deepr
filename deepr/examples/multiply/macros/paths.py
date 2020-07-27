"""Macro that generates new path_model based on date."""

import datetime


class Paths(dict):
    """Macro that generates new path_model based on date."""

    def __init__(self, path_model: str = None, path_dataset: str = None):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if path_model is None:
            path_model = f"viewfs://root/user/deepr/dev/example/models/{now}"
        if path_dataset is None:
            path_dataset = f"viewfs://root/user/deepr/dev/example/models/{now}/data.tfrecord"
        path_variables = f"{path_model}/variables"
        path_saved_model = f"{path_model}/saved_model"
        path_optimized_model = f"{path_model}/optimized_model"
        super().__init__(
            path_model=path_model,
            path_dataset=path_dataset,
            path_variables=path_variables,
            path_saved_model=path_saved_model,
            path_optimized_model=path_optimized_model,
        )
