from typing import Any, Literal, Sequence

import numpy as np
from joblib import Parallel
from sklearn.base import clone

from .._base import LossBase
from .._sklearn import apply_custom_loss
from .asymmetric import AsymmetricLoss


class VarianceEstimator:
    """Estimator that estimates the distribution by simply using multiple estimators
    with different `t`.
    Compared to [NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/) or
    [CatBoost's Uncertainty](https://catboost.ai/en/docs/references/uncertainty),
    this estimator is much slower and does not support "natural gradient",
    but does not require any assumption on the distribution.

    Note that NGBoost supports
    [any user-defineddistribution](https://stanfordmlgroup.github.io/ngboost/5-dev.html) # noqa
    but it has to be defined beforehand.

    NGBoost requires mean estimator and log standard deviation estimator
    to be trained simultaneously, which is very difficult to implement
    in sklearn / lightgbm / xgboost. (Need to start and stop fitting per iteration)
    Consider change `Base` parameter in NGBoost.
    (See https://github.com/stanfordmlgroup/ngboost/issues/250)
    """

    ts_: Sequence[float]

    def __init__(
        self,
        estimator: Any,
        loss: LossBase,
        *,
        ts: int | Sequence[float],
        n_jobs: int | None = 1,
        verbose: int = 0,
        m_type: Literal["mean", "median"] = "mean",
        var_type: Literal["var", "std", "range", "mae", "mse"] = "var",
    ) -> None:
        if not hasattr(estimator, "fit"):
            raise TypeError(f"{estimator} does not have fit method")
        if not hasattr(estimator, "predict"):
            raise TypeError(f"{estimator} does not have predict method")
        self.estimator = estimator
        self.loss = loss
        self.ts = ts
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> None:
        ts = self.ts
        if isinstance(ts, int):
            ts = np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts)
        self.ts_ = ts  # type: ignore
        estimators_ = [
            apply_custom_loss(clone(self.estimator), AsymmetricLoss(self.loss, t=t))
            for t in self.ts_
        ]
        parallel_result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            [estimator.fit(X, y) for estimator in estimators_]
        )
        if parallel_result is None:
            raise RuntimeError("joblib.Parallel returned None")
        self.estimators_ = estimators_

    def predict_raw(self, X: Any) -> np.ndarray:
        return np.array([estimator.predict(X) for estimator in self.estimators_])
