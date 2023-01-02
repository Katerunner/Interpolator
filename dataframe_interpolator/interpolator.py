import copy

import numpy as np
import pandas as pd


class Interpolator:
    n_iter: int = None
    model = None
    verbose: bool = None
    normalize: bool = None
    normalize_algorithm: str = None
    df_min = None
    df_max = None
    df_avg = None
    df_std = None
    score_history = None
    models_list = None

    def __init__(self, model=None, normalize=True, normalize_algorithm='minmax', n_iter=20, verbose=True):
        self.model = model
        self.n_iter = n_iter
        self.verbose = verbose
        self.normalize = normalize
        self.normalize_algorithm = normalize_algorithm
        self.score_history = []

    def normalize_dataframe(self, df: pd.DataFrame):
        df_stage = df.copy()

        if self.normalize_algorithm == 'minmax':
            self.df_min = df_stage.min()
            self.df_max = df_stage.max()
            dfn = (df_stage - self.df_min) / (self.df_max - self.df_min)
        elif self.normalize_algorithm == 'standard':
            self.df_avg = df_stage.mean()
            self.df_std = df_stage.std()
            dfn = (df_stage - self.df_avg) / self.df_std
        else:
            raise ValueError(f"No such normalize algorithm as {self.normalize_algorithm} supported")
        return dfn

    def process(self, df: pd.DataFrame, models_list: list = None):
        dfi = self.normalize_dataframe(df) if self.normalize else df.copy()
        i_range, j_range = dfi.shape

        self.check_models_list(models_list, j_range)

        # Create models for each column
        self.models_list = [copy.deepcopy(self.model)] * j_range if models_list is None else models_list

        # Get missing values to fill
        tpass_indexes = []
        ppass_indexes = []
        inter_indexes = []

        for j in range(j_range):
            na_bool = dfi.iloc[:, j].isna()
            if na_bool.sum() == 0:
                ppass_indexes.append(j)
            elif na_bool.sum() == len(dfi.iloc[:, j]):
                tpass_indexes.append(j)

            inter_indexes.append(dfi.iloc[:, j].isna())

        if self.verbose:
            print("Iter\t", *range(j_range), "Score", end=" ")

        for iteration in range(self.n_iter):
            scores = []

            if self.verbose:
                print()
                print(iteration, end="\t ")

            for target_index in range(j_range):

                if self.verbose:
                    pl = len(str(target_index)) - 1

                    status_string = " " * pl + "."
                    status_string = " " * pl + "E" if target_index in tpass_indexes else status_string
                    status_string = " " * pl + "F" if target_index in ppass_indexes else status_string

                    print(status_string, end=" ")

                # If there are missing values in column
                if target_index not in (ppass_indexes + tpass_indexes):

                    explan_index = list(range(j_range))
                    explan_index.remove(target_index)

                    for train_pass_index in tpass_indexes:
                        explan_index.remove(train_pass_index)

                    y = dfi.iloc[:, target_index].copy()
                    X = dfi.iloc[:, explan_index].copy()
                    X.fillna(X.mean(), inplace=True)

                    y_train = y[~inter_indexes[target_index]]
                    X_train = X[~inter_indexes[target_index]]
                    X_inter = X[inter_indexes[target_index]]
                    y_inter = self.models_list[target_index].fit(X_train, y_train).predict(X_inter)
                    y[inter_indexes[target_index]] = y_inter
                    scores.append(self.models_list[target_index].score(X_train, y_train))
                    dfi.iloc[:, target_index] = y

            self.score_history.append(np.mean(scores))

            if self.verbose:
                print(np.mean(scores).round(4), end="")

        # Inverse normalization
        if self.normalize_algorithm == 'minmax':
            df_result = dfi * (self.df_max - self.df_min) + self.df_min if self.normalize else dfi
        elif self.normalize_algorithm == 'standard':
            df_result = dfi * self.df_std + self.df_avg
        else:
            df_result = dfi

        return df_result

    def get_models(self):
        return self.models_list

    def check_models_list(self, models_list, j_range):
        # Raise error if no model is specified
        if self.model is None and models_list is None:
            raise ValueError("No model was specified in __init__ or in process method")

        # Raise error if number of models does not correspond to number of columns
        if models_list is not None and len(models_list) != j_range:
            raise IndexError("Number of models must be equal to the number of columns")
