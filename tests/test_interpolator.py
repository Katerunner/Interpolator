from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from dataframe_interpolator.interpolator import Interpolator


def create_example_dataset(shape=(1000, 10), missing=0.3):
    data = np.random.rand(shape[0], shape[1])
    data = pd.DataFrame(data)
    i_range, j_range = data.shape

    for i in range(i_range):
        for j in range(1, j_range):
            if np.random.rand() < missing:
                data.iloc[i, j] = np.nan

    data.iloc[:, 1] = np.nan
    data.iloc[:, -1] = data.iloc[:, -1].round()
    data.columns = [chr(i + 97) for i in range(shape[1])]
    return data


class TestInterpolator(TestCase):

    def general_test(self, data, data_result):
        self.assertEqual(data.shape, data_result.shape, msg="Shapes are different")
        self.assertEqual(data_result.iloc[:, 1].isna().sum(), 1000, msg="Empty column is not empty now")
        for j in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.assertEqual(data_result.iloc[:, j].isna().sum(), 0, msg="Missing values were not all interpolated")

    def test_1_general(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.fill_na(data.copy())
        self.general_test(data, data_result)

    def test_2_no_normalization(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        dip = Interpolator(LinearRegression(), normalize=False, n_iter=10, verbose=True)
        data_result = dip.fill_na(data.copy())
        self.general_test(data, data_result)

    def test_3_all_sklearn(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        regressors = [
            LinearRegression, Lasso, Ridge, ElasticNet,  # linear models
            DecisionTreeRegressor, ExtraTreeRegressor,  # trees
            RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor  # ensembles
        ]

        for regressor in regressors:
            dip = Interpolator(regressor(), normalize=False, n_iter=2, verbose=True)
            data_result = dip.fill_na(data.copy())
            self.general_test(data, data_result)

    def test_4_half_scenario(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.5)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.fill_na(data.copy())
        self.general_test(data, data_result)

    def test_5_bad_scenario(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.85)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.fill_na(data.copy())
        self.general_test(data, data_result)
