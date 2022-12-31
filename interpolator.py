import tqdm


class Interpolator:
    n_iter: int = None
    model = None
    verbose: bool = None
    normalize: bool = None
    normalize_algorithm: str = None
    df_min = None
    df_max = None

    def __init__(self, model, normalize=True, normalize_algorithm='minmax', n_iter=20, verbose=True):
        self.model = model
        self.n_iter = n_iter
        self.verbose = verbose
        self.normalize = normalize
        self.normalize_algorithm = normalize_algorithm

    def normalize_dataframe(self, df):
        df_stage = df.copy()

        if self.normalize_algorithm == 'minmax':
            self.df_min = df_stage.min()
            self.df_max = df_stage.max()
            dfn = (df_stage - self.df_min) / (self.df_max - self.df_min)
        elif self.normalize_algorithm == 'standard':
            dfn = (df_stage - df_stage.mean()) / df_stage.std()
        else:
            raise ValueError(f"No such normalize algorithm as {self.normalize_algorithm} supported")
        return dfn

    def fill_na(self, df):
        dfi = self.normalize_dataframe(df) if self.normalize else df.copy()
        i_range, j_range = dfi.shape

        # Get missing values to fill
        inter_indexes = []
        for j in range(j_range):
            inter_indexes.append(dfi.iloc[:, j].isna())

        for iteration in tqdm.trange(5, disable=not self.verbose):
            scores = []
            for target_index in range(j_range):
                explan_index = list(range(j_range))
                explan_index.remove(target_index)
                y = dfi.iloc[:, target_index].copy()

                X = dfi.iloc[:, explan_index].copy()
                X.fillna(X.mean(), inplace=True)

                y_train = y[~inter_indexes[target_index]]
                X_train = X[~inter_indexes[target_index]]
                X_inter = X[inter_indexes[target_index]]
                y_inter = self.model.fit(X_train, y_train).predict(X_inter)
                y[y.isna()] = y_inter
                scores.append(self.model.score(X_train, y_train))
                dfi.iloc[:, target_index] = y

        df_result = dfi * (self.df_max - self.df_min) + self.df_min if self.normalize else dfi
        return df_result
