from typing import Any, Literal, Sequence

import numpy as np
from joblib import Parallel
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from .._base import LossBase
from .._sklearn import apply_custom_loss
from .asymmetric import AsymmetricLoss


class VarianceEstimator(BaseEstimator):
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
    m_type: Literal["mean", "median"]
    var_type: Literal["var", "std", "range", "mae", "mse"]

    def __init__(
        self,
        estimator: Any,
        loss: LossBase,
        *,
        ts: int | Sequence[float],
        n_jobs: int | None = 1,
        verbose: int = 0,
        random_state: int | None = None,
        m_type: Literal["mean", "median"] = "median",
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
        self.m_type = m_type
        self.var_type = var_type
        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

    def fit(self, X: Any, y: Any) -> None:
        ts = self.ts
        if isinstance(ts, int):
            ts = np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts)
        self.ts_ = ts  # type: ignore
        estimators_ = [
            apply_custom_loss(clone(self.estimator), AsymmetricLoss(self.loss, t=t))
            for t in self.ts_
        ]
        estimators_ = [
            estimator.set_params(random_state=self.random.randint(2**32))
            for estimator in estimators_
        ]
        parallel_result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            [estimator.fit(X, y) for estimator in estimators_]
        )
        if parallel_result is None:
            raise RuntimeError("joblib.Parallel returned None")
        self.estimators_ = estimators_

    def predict_raw(self, X: Any, **predict_params: Any) -> NDArray[Any]:
        return np.array(
            [estimator.predict(X, **predict_params) for estimator in self.estimators_]
        )

    def predict(
        self,
        X: Any,
        type_: Literal["mean", "median", "var", "std", "range", "mae", "mse"]
        | None = None,
        **predict_params: Any,
    ) -> NDArray[Any]:
        type_ = type_ or self.m_type
        if type_ == "mean":
            return self.predict_raw(X, **predict_params).mean(axis=0)
        elif type_ == "median":
            return np.median(self.predict_raw(X, **predict_params), axis=0)
        elif type_ == "var":
            return self.predict_raw(X, **predict_params).var(axis=0)
        elif type_ == "std":
            return self.predict_raw(X, **predict_params).std(axis=0)
        elif type_ == "range":
            return self.predict_raw(X, **predict_params).max(axis=0) - self.predict_raw(
                X, **predict_params
            ).min(axis=0)
        elif type_ == "mae":
            return np.abs(
                self.predict_raw(X, **predict_params)
                - self.predict_raw(X, **predict_params).mean(axis=0)
            ).mean(axis=0)
        elif type_ == "mse":
            return (
                (
                    self.predict_raw(X, **predict_params)
                    - self.predict_raw(X, **predict_params).mean(axis=0)
                )
                ** 2
            ).mean(axis=0)
        else:
            raise ValueError(f"Unknown type_: {type_}")

    def predict_var(
        self,
        X: Any,
        type_: Literal["var", "std", "range", "mae", "mse"] | None = None,
        **predict_params: Any,
    ) -> NDArray[Any]:
        type_ = type_ or self.var_type
        return self.predict(X, type_=type_, **predict_params)
