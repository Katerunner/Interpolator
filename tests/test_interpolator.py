import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from dataframe_interpolator.interpolator import Interpolator


def create_example_dataset(shape=(1000, 10), missing=0.3):
    assert shape[1] >= 3, "Must be more then 2 columns, 1-st is full, 2-nd is empty"

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

    def general_test(self, data, data_result, nan_columns_indexes: list = None):
        self.assertEqual(data.shape, data_result.shape, msg="Shapes are different")
        self.assertEqual(data_result.iloc[:, 1].isna().sum(), data.shape[0], msg="Empty column is not empty now")

        full_columns = list(range(data.shape[1]))
        full_columns.remove(1)

        if nan_columns_indexes is not None:
            for i in nan_columns_indexes:
                full_columns.remove(i)

        for j in full_columns:
            self.assertEqual(data_result.iloc[:, j].isna().sum(), 0, msg="Missing values were not all interpolated")

    def test_1_general(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

    def test_2_no_normalization(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        dip = Interpolator(LinearRegression(), normalize=False, n_iter=10, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

    @unittest.skipIf(True, "time consuming")
    def test_3_all_sklearn(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.3)
        regressors = [
            LinearRegression, Lasso, Ridge, ElasticNet,  # linear models
            DecisionTreeRegressor, ExtraTreeRegressor,  # trees
            RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor  # ensembles
        ]

        for regressor in regressors:
            dip = Interpolator(regressor(), normalize=False, n_iter=2, verbose=True)
            data_result = dip.process(data.copy())
            self.general_test(data, data_result)

    def test_4_half_scenario(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.5)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

    def test_5_bad_scenario(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.85)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

    def test_5_get_models_from_single_model_in_init(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.85)
        dip = Interpolator(LinearRegression(), n_iter=10, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

        models_list = dip.get_models()
        self.assertEqual(len(models_list), data.shape[1])

    def test_6_get_models_from_directly_specifying_in_method(self):
        data = create_example_dataset(shape=(1000, 3), missing=0.85)
        dip = Interpolator(n_iter=10, verbose=True)
        data_result = dip.process(data.copy(), [LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor()])
        self.general_test(data, data_result)

        models_list = dip.get_models()
        self.assertEqual(len(models_list), data.shape[1])
        self.assertIsInstance(models_list[0], LinearRegression)
        self.assertIsInstance(models_list[1], RandomForestRegressor)
        self.assertIsInstance(models_list[2], DecisionTreeRegressor)

    def test_7_min_present_values_to_consider_empty(self):
        min_present = 0.2
        data = create_example_dataset(shape=(1000, 10), missing=0.2)

        # Only 20% of the column is present
        data.iloc[:int((1 - min_present) * 1000), 5] = np.nan
        self.assertTrue(data.iloc[:, 5].isna().sum() >= int((1 - min_present) * 1000))

        # Column is considered empty if it has less then 30% of values
        dip = Interpolator(LinearRegression(), n_iter=10, min_present=0.3, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result, nan_columns_indexes=[5])
        self.assertTrue(data_result.iloc[:, 5].isna().sum() > 0)

    def test_8_consider_empty_if_less_then_two_values_present(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.2)

        # Only 2 records in the column are present
        data.iloc[:, 5] = np.nan
        data.iloc[:2, 5] = np.array([0.1, 0.2])
        self.assertEqual(data.iloc[:, 5].isna().sum(), 1000 - 2)

        # Column is considered empty if it has less then 2 values
        dip = Interpolator(LinearRegression(), n_iter=10, min_present=0.0, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result, nan_columns_indexes=[5])
        self.assertTrue(data_result.iloc[:, 5].isna().sum() > 0)

    def test_9_dont_consider_empty_if_three_values_present(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.2)
        # Now 3 records in the column are present
        data.iloc[:, 5] = np.nan
        data.iloc[:3, 5] = np.array([0.1, 0.2, 0.3])
        self.assertEqual(data.iloc[:, 5].isna().sum(), 1000 - 3)

        # Column is not considered empty if it has less then 2 values and min_present=0.0
        dip = Interpolator(LinearRegression(), n_iter=10, min_present=0.0, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)

    def test_11_test_normalization_with_same_value(self):
        data = create_example_dataset(shape=(1000, 10), missing=0.2)
        # Now 3 records in the column are present
        data[data.columns[0]] = 1
        data.iloc[2, 0] = np.nan

        # Column is not considered empty if it has less then 2 values and min_present=0.0
        dip = Interpolator(LinearRegression(), n_iter=10, min_present=0.0, verbose=True)
        data_result = dip.process(data.copy())
        self.general_test(data, data_result)
