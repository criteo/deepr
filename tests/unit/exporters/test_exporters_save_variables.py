"""Test for exporters.SaveVariables"""

import numpy as np
import pandas as pd

import deepr


class MockEstimator:
    """Mock Estimator for SaveVariables"""

    def get_variable_value(self, name: str):
        if name == "x":
            return np.zeros([10, 10])
        elif name == "y":
            return np.ones([10])
        else:
            raise ValueError()


def test_exporters_save_variables(tmpdir):
    """Test SaveVariables exporter"""
    path_variables = str(tmpdir.join("variables"))
    exporter = deepr.exporters.SaveVariables(path_variables, ["x", "y"])
    exporter(MockEstimator())
    for variable in ["x", "y"]:
        with deepr.io.ParquetDataset(deepr.io.Path(path_variables, variable)).open() as ds:
            got = ds.read_pandas().to_pandas()
            expected = pd.DataFrame(MockEstimator().get_variable_value(variable))
            np.testing.assert_allclose(got, expected)
