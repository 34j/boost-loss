from __future__ import annotations

from copy import copy
from typing import Any, Literal, Sequence

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import Self

from ..base import LossBase
from ..sklearn import apply_custom_loss
from .asymmetric import AsymmetricLoss


def _recursively_set_random_state(estimator: BaseEstimator, random_state: int) -> None:
    if hasattr(estimator, "random_state") and hasattr(estimator, "set_params"):
        estimator.set_params(random_state=random_state)
    for _, v in copy(estimator.get_params(deep=False)).items():
        if hasattr(v, "get_params"):
            _recursively_set_random_state(v, random_state)


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
        target_transformer: BaseEstimator | Any | None = None,
    ) -> None:
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

        Parameters
        ----------
        estimator : Any
            The base estimator to use for fitting the data.
        loss : LossBase
            The loss function to use for fitting the data.
            Generally, `loss` should not be `AsymmetricLoss`.
        ts : int | Sequence[float]
            The list of `t` to use for fitting the data or the number of `t` to use.
            If `ts` is an integer, `np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts)` is used.
        n_jobs : int | None, optional
            The number of jobs to run in parallel for `fit`. `None` means 1.
        verbose : int, optional
            The verbosity level.
        random_state : int | None, optional
            The random state to use for fitting the data. If `None`, the random state is not set.
            If not `None`, new random state generated from `random_state` is set to each estimator.
        m_type : Literal[&quot;mean&quot;, &quot;median&quot;], optional
            M-statistics type to return from `predict` by default, by default "median"
        var_type : Literal[&quot;var&quot;, &quot;std&quot;, &quot;range&quot;, &quot;mae&quot;, &quot;mse&quot;], optional
            Variance type to return from `predict` by default, by default "var"
        target_transformer : BaseEstimator | Any | None, optional
            The transformer to use for transforming the target, by default None
            If `None`, no `TransformedTargetRegressor` is used.

        Raises
        ------
        TypeError
            Raises if `estimator` does not have `fit` method or `predict` method.
        """
        if not hasattr(estimator, "fit"):
            raise TypeError(f"{estimator} does not have fit method")
        if not hasattr(estimator, "predict"):
            raise TypeError(f"{estimator} does not have predict method")
        self.estimator = estimator
        self.loss = loss
        self.ts = ts
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.m_type = m_type
        self.var_type = var_type
        self.target_transformer = target_transformer
        self.random = np.random.RandomState(random_state)

    def fit(self, X: Any, y: Any, **fit_params: Any) -> Self:
        """Fit each estimator with different `t`.

        Parameters
        ----------
        X : Any
            The training input samples.
        y : Any
            The target values.

        Returns
        -------
        Self
            Fitted estimator.

        Raises
        ------
        RuntimeError
            Raises if joblib fails to return the results.
        """
        ts = self.ts
        if isinstance(ts, int):
            ts = np.linspace(1 / (ts * 2), 1 - 1 / (ts * 2), ts)
        self.ts_ = ts  # type: ignore
        estimators_ = [
            apply_custom_loss(
                self.estimator,
                AsymmetricLoss(self.loss, t=t),
                target_transformer=self.target_transformer,
            )
            for t in self.ts_
        ]
        if self.random is not None:
            for estimator in estimators_:
                _recursively_set_random_state(
                    estimator, self.random.randint(0, np.iinfo(np.int32).max)
                )
        parallel_result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            [delayed(estimator.fit)(X, y, **fit_params) for estimator in estimators_]
        )
        if parallel_result is None:
            raise RuntimeError("joblib.Parallel returned None")
        self.estimators_ = parallel_result
        return self

    def predict_raw(self, X: Any, **predict_params: Any) -> NDArray[Any]:
        """Returns raw predictions of each estimator.

        Parameters
        ----------
        X : Any
            X
        **predict_params : Any
            The parameters to be passed to `predict` method of each estimator.

        Returns
        -------
        NDArray[Any]
            Raw predictions of each estimator with shape (n_estimators, n_samples)
        """
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
        """Returns predictions of the ensemble.

        Parameters
        ----------
        X : Any
            X
        type_ : Literal['mean', 'median', 'var', 'std', 'range', 'mae', 'mse'], optional
            Type of the prediction, by default None
            If None, self.m_type is used.
        **predict_params : Any
            The parameters to be passed to `predict` method of each estimator.

        Returns
        -------
        NDArray[Any]
            Predictions of the ensemble with shape (n_samples,)

        Raises
        ------
        ValueError
            When type_ is not supported.
        """
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
        """Returns variance of the ensemble.

        Parameters
        ----------
        X : Any
            X
        type_ : Literal['var', 'std', 'range', 'mae', 'mse'], optional
            Type of the variance, by default None
            If None, self.var_type is used.
        **predict_params : Any
            The parameters to be passed to `predict` method of each estimator.

        Returns
        -------
        NDArray[Any]
            Variance of the ensemble with shape (n_samples,)
        """
        type_ = type_ or self.var_type
        return self.predict(X, type_=type_, **predict_params)
