import importlib.util

import pytest

if importlib.util.find_spec("seaborn") is None:
    pytest.skip("Skipping tests that require seaborn", allow_module_level=True)
else:
    from pathlib import Path
    from typing import Any

    import matplotlib.pyplot as plt
    import numpy as np
    import pytest
    import seaborn as sns
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from pandas import DataFrame
    from scipy.stats import norm
    from sklearn.datasets import make_regression
    from xgboost import XGBRegressor

    from boost_loss.regression import L1Loss
    from boost_loss.regression.sklearn import VarianceEstimator

    @pytest.mark.parametrize(
        "estimator",
        [
            CatBoostRegressor(n_estimators=100),
            LGBMRegressor(),
            XGBRegressor(base_score=0.5),
        ],
    )
    def test_normal(estimator: Any) -> None:
        X, y = make_regression(n_samples=100, n_features=1, random_state=0)
        y = np.random.standard_normal(size=y.shape)
        ve = VarianceEstimator(estimator=estimator, loss=L1Loss(), ts=5)
        ve.fit(X, y)
        Y = ve.predict_raw(X)
        Y = DataFrame(Y.T, columns=[f"{t:g}/{norm.ppf(t):g}" for t in ve.ts_])
        sns.violinplot(data=Y)
        plt.title(
            f"Violin plot of predictions for estimator {estimator.__class__.__name__}"
        )
        path = (
            Path(__file__).parent
            / ".cache"
            / f"test_normal_{estimator.__class__.__name__}.png"
        )
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path)
        plt.close()
