---

# DataFrame Interpolator Tool

[comment]: <> (## dataframe_interpolator)

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/author_name/project_urlname/branch/main/graph/badge.svg?token=project_urlname_token_here&#41;]&#40;https://codecov.io/gh/author_name/project_urlname&#41;)

[comment]: <> ([![CI]&#40;https://github.com/author_name/project_urlname/actions/workflows/main.yml/badge.svg&#41;]&#40;https://github.com/author_name/project_urlname/actions/workflows/main.yml&#41;)

DataFrame Interpolator Tool is a python package that helps to solve the problem of missing data in pandas dataset. It
uses machine learning models from `scikit-learn` package to fill in missing data in dataframe.

The algorithm:

- Pick column by column as target data
- Rest columns are explanatory data
- NaNs in explanatory columns are interpolated by mean value of the column
- Non-NaN values from target columns are used for training
- Estimator is trained on explanatory data to predict known values of target column
- This estimator is used to predict unknown (missing or NaN) values of target column
- New (predicted) values are inserted into the column
- All the steps from the above applied to next column
- Iteration is when all columns in data set are being processed by the estimator
- Several iterations allow updating the previously predicted values, as the explanatory data was also updated in
  previous steps.

As a user, you need to provide the sklearn model or the similar syntax model with fit, score and predict methods, likely
regressor:

- RandomForestRegressor
- LinearRegression
- DecisionTreeRegressor
- etc

Script takes pandas `DataFrame` as an input. All values must be numerical, consider transforming your categorical data
to numerical labels or one-hot encodings.

## Install it from PyPI

Package is not yet released into PyPI, so the installation is performed through the GitHub.

```bash
pip install git+https://github.com/Katerunner/Interpolator
```

## Usage

Basic scenario:

```py
import numpy as np
import pandas as pd
from dataframe_interpolator.interpolator import Interpolator
# Load core model
from sklearn.ensemble import RandomForestRegressor

# Generate example data
data = np.random.rand(1000, 10)
data = pd.DataFrame(data)
i_range, j_range = data.shape

# With prob of 0.3 value is NaN
for i in range(i_range):
    for j in range(1, j_range):
        if np.random.rand() < 0.3:
            data.iloc[i, j] = np.nan

# With normalization
ip_model = Interpolator(model=RandomForestRegressor(),
                        normalize=True,
                        normalize_algorithm='minmax',
                        n_iter=10,
                        verbose=True)

df_result = ip_model.process(data.copy())
display(df_result)  # If in Jupyter
```

Interpolator also saves the models for each column. If the model was provided in `__init__`, then all the columns are
trained using the copy of the same model and `get_models()` method will return each trained model for each column.

In addition to this, the separate models for each column can be specified in `process()` method `models_list` parameter.
In this case the provided models will be trained corresponding to each column and return by `get_models()` method. The
model specified in `__init__` will be ignored in this case.

```python
ip_model.get_models()
```

You can also see the example of usage in the `example_usage.ipynb`.

