"""Hyper parameter tuning"""

import abc
from typing import Dict
from dataclasses import dataclass
import logging
from copy import deepcopy
import itertools
import time

import numpy as np

from deepr.config.base import parse_config, from_config
from deepr.config.macros import assert_no_macros
from deepr.jobs import base
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


class Sampler(abc.ABC):
    """Parameters Sampler"""

    def __iter__(self):
        raise NotImplementedError()


class GridSampler(Sampler):
    """Grid Sampler wrapping ParameterGrid from sklearn"""

    def __init__(self, param_grid, repeat: int = 1):
        self.param_grid = param_grid
        self.repeat = repeat

    def __iter__(self):
        for _ in range(self.repeat):
            params, values = zip(*sorted(self.param_grid.items()))
            for vals in itertools.product(*values):
                yield dict(zip(params, vals))


class ParamsSampler(Sampler):
    """Parameter Sampler"""

    def __init__(self, param_grid, n_iter: int = 10, repeat: int = 1, seed: int = None):
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.repeat = repeat
        self.seed = seed

    def __iter__(self):
        # Initialize random state and sort params for reproducibility
        rng = np.random.RandomState(self.seed)
        items = sorted(self.param_grid.items())

        # If all params values are lists, sample with no replacement
        if not any(hasattr(val, "rvs") for val in self.param_grid.values()):
            LOGGER.info("Sampling with no replacement (parameter grid only has lists of values)")
            grid = list(GridSampler(self.param_grid))
            sampled = [grid[idx] for idx in rng.randint(len(grid), size=min(len(grid), self.n_iter))]
            LOGGER.info(f"Sampled {len(sampled)} parameters from a total of {len(grid)}")
            for params in sampled:
                for _ in range(self.repeat):
                    yield params

        # If any param is a distribution, sample with replacement
        else:
            for _ in range(self.n_iter):
                params = dict()
                for param, val in items:
                    if hasattr(val, "rvs"):
                        params[param] = val.rvs(random_state=rng)
                    else:
                        params[param] = val[rng.randint(len(val))]
                for _ in range(self.repeat):
                    yield params


@dataclass
class ParamsTuner(base.Job):
    """Params tuner"""

    job: Dict
    macros: Dict
    sampler: Sampler

    def run(self):
        sampled = list(self.sampler)
        for idx, params in enumerate(sampled):
            LOGGER.info(f"Launching job with params: {params}")

            # Update macro params with sampled values
            macros = deepcopy(self.macros)
            macros["params"] = {**macros["params"], **params}
            assert_no_macros(macros["params"])

            # Parse config and run job
            parsed = parse_config(self.job, macros)
            job = from_config(parsed)
            if not isinstance(job, base.Job):
                raise TypeError(f"Expected type Job but got {type(job)}")
            job.run()
            mlflow.clear_run()

            # New parameters based on time need to be different
            if idx + 1 < len(sampled):
                LOGGER.info("Sleeping 2 seconds before next experiment\n")
                time.sleep(2)
