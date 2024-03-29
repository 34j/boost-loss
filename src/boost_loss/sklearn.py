from __future__ import annotations

import functools
import importlib.util
from copy import deepcopy
from typing import Any, Literal, TypeVar, overload

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from .base import LossBase

TEstimator = TypeVar("TEstimator", cb.CatBoost, lgb.LGBMModel, xgb.XGBModel)


@overload
def apply_custom_loss(
    estimator: TEstimator,
    loss: LossBase,
    *,
    copy: bool = ...,
    copy_loss: bool = ...,
    apply_objective: bool = ...,
    apply_eval_metric: bool = ...,
    target_transformer: None = ...,
    recursive: bool = ...,
    recursive_strict: bool = ...,
) -> TEstimator:
    ...


@overload
def apply_custom_loss(
    estimator: TEstimator,
    loss: LossBase,
    *,
    copy: bool = ...,
    copy_loss: bool = ...,
    apply_objective: bool = ...,
    apply_eval_metric: bool = ...,
    target_transformer: BaseEstimator = ...,
    recursive: bool = ...,
    recursive_strict: bool = ...,
) -> TransformedTargetRegressor:
    ...


def apply_custom_loss(
    estimator: TEstimator,
    loss: LossBase,
    *,
    copy: bool = True,
    copy_loss: bool = True,
    apply_objective: bool = True,
    apply_eval_metric: bool = True,
    target_transformer: BaseEstimator | Any | None = StandardScaler(),
    recursive: bool = True,
    recursive_strict: bool = False,
) -> TEstimator | TransformedTargetRegressor:
    """Apply custom loss to the estimator.

    Parameters
    ----------
    estimator : TEstimator
        CatBoost, LGBMModel, or XGBModel
    loss : LossBase
        The custom loss to apply
    copy : bool, optional
        Whether to copy the estimator using `sklearn.base.clone`, by default True
    copy_loss : bool, optional
        Whether to copy the loss using `copy.deepcopy`, by default True
    apply_objective : bool, optional
        Whether to apply the custom loss to the estimator's objective, by default True
    apply_eval_metric : bool, optional
        Whether to apply the custom loss to the estimator's eval_metric, by default True
    target_transformer : BaseEstimator | Any | None, optional
        The target transformer to use, by default StandardScaler()
        (This option exists because some loss functions require the target
        to be normalized (i.e. `LogCoshLoss`))
    recursive : bool, optional
        Whether to recursively search for estimators inside the estimator
        and apply the custom loss to all of them, by default True
    recursive_strict : bool, optional
        Whether to recursively search for estimators inside the estimator's
        attributes, lists, tuples, sets, and frozensets as well,
        by default False

    Returns
    -------
    TEstimator | TransformedTargetRegressor
        The estimator with the custom loss applied
    """
    if copy:
        estimator = clone(estimator)
    if copy_loss:
        loss = deepcopy(loss)
    if isinstance(estimator, cb.CatBoost):
        if apply_objective:
            estimator.set_params(loss_function=loss)
        if apply_eval_metric:
            estimator.set_params(eval_metric=loss)
    if isinstance(estimator, lgb.LGBMModel):
        if apply_objective:
            estimator.set_params(objective=loss)
        if apply_eval_metric:
            estimator_fit = estimator.fit

            @functools.wraps(estimator_fit)
            def fit(X: Any, y: Any, **fit_params: Any) -> Any:
                fit_params["eval_metric"] = loss.eval_metric_lgb
                return estimator_fit(X, y, **fit_params)

            setattr(estimator, "fit", fit)
    if isinstance(estimator, xgb.XGBModel):
        if apply_objective:
            estimator.set_params(objective=loss)
        if apply_eval_metric:
            estimator.set_params(eval_metric=loss.eval_metric_xgb_sklearn)

    if recursive:
        if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
            for key, value in estimator.get_params(deep=True).items():
                if hasattr(value, "fit"):
                    estimator.set_params(
                        **{
                            key: apply_custom_loss(
                                value,
                                loss,
                                copy=False,
                                copy_loss=copy_loss,
                                apply_objective=apply_objective,
                                apply_eval_metric=apply_eval_metric,
                                target_transformer=None,
                                recursive=False,
                                recursive_strict=False,
                            )
                        }
                    )
    if recursive_strict:
        if hasattr(estimator, "__dict__"):
            for _, value in estimator.__dict__.items():
                apply_custom_loss(
                    value,
                    loss,
                    copy=False,
                    copy_loss=copy_loss,
                    apply_objective=apply_objective,
                    apply_eval_metric=apply_eval_metric,
                    target_transformer=None,
                    recursive=True,
                    recursive_strict=True,
                )
        elif isinstance(estimator, (list, tuple, set, frozenset)):
            # https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/base.py#L66
            for value in estimator:
                apply_custom_loss(
                    value,
                    loss,
                    copy=False,
                    copy_loss=copy_loss,
                    apply_objective=apply_objective,
                    apply_eval_metric=apply_eval_metric,
                    target_transformer=None,
                    recursive=True,
                    recursive_strict=True,
                )

    if target_transformer is None:
        return estimator
    return TransformedTargetRegressor(estimator, transformer=clone(target_transformer))


if importlib.util.find_spec("ngboost") is not None:
    from ngboost import NGBoost
    from ngboost.distns import Normal
    from numpy.typing import NDArray

    def patch_ngboost(estimator: NGBoost) -> NGBoost:
        """Patch NGBoost to return only the mean prediction in `predict`
        and the variance in `predict_var` to be consistent with other models.
        The patch will not apply if the estimator is cloned using `sklearn.base.clone()`
        and requires re-patching.

        Parameters
        ----------
        estimator : NGBoost
            The NGBoost estimator to patch.

        Returns
        -------
        NGBoost
            The patched NGBoost estimator.
        """

        self = estimator

        def predict_var(X: Any, **predict_params: Any) -> NDArray[Any]:
            dist = self.pred_dist(X, **predict_params)
            if not isinstance(dist, Normal):
                raise NotImplementedError
            return dist.var

        setattr(estimator, "predict_var", predict_var)

        def predict_std(X: Any, **predict_params: Any) -> NDArray[Any]:
            dist = self.pred_dist(X, **predict_params)
            if not isinstance(dist, Normal):
                raise NotImplementedError
            return dist.scale

        setattr(estimator, "predict_std", predict_std)

        original_predict = estimator.predict

        def predict(
            X: Any,
            *,
            return_std: bool = False,
            **predict_params: Any,
        ) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
            if return_std:
                dist = self.pred_dist(X, **predict_params)
                if not isinstance(dist, Normal):
                    raise NotImplementedError
                return dist.mean, dist.scale
            else:
                return original_predict(X, **predict_params)

        setattr(estimator, "predict", predict)

        return estimator


def patch_catboost(estimator: cb.CatBoost) -> cb.CatBoost:
    """Patch CatBoost to return only the mean prediction in `predict`
    and the variance in `predict_var` to be consistent with other models.
    The patch will not apply if the estimator is cloned using `sklearn.base.clone()`
    and requires re-patching.

    Parameters
    ----------
    estimator : cb.CatBoost
        The CatBoost estimator to patch.

    Returns
    -------
    cb.CatBoost
        The patched CatBoost estimator.
    """
    original_predict = estimator.predict

    self = estimator

    def predict_var(
        X: Any,
        prediction_type: Literal["knowledge", "data", "total"] = "total",
        **predict_params: Any,
    ) -> NDArray[Any]:
        uncertainty = self.virtual_ensembles_predict(
            X, prediction_type="TotalUncertainty", **predict_params
        )

        loss_function = self.get_params()["loss_function"]
        knowledge_uncertainty = np.zeros_like(uncertainty[..., 0])
        data_uncertainty = np.zeros_like(uncertainty[..., 0])
        if loss_function == "RMSEWithUncertainty":
            knowledge_uncertainty = uncertainty[..., 1]
            data_uncertainty = uncertainty[..., 2]
        elif isinstance(estimator, cb.CatBoostClassifier):
            knowledge_uncertainty = uncertainty[..., 1]
        else:
            data_uncertainty = uncertainty[..., 0]
            knowledge_uncertainty = uncertainty[..., 1] - data_uncertainty

        if prediction_type == "knowledge":
            return knowledge_uncertainty
        elif prediction_type == "data":
            return data_uncertainty
        elif prediction_type == "total":
            return knowledge_uncertainty + data_uncertainty
        else:
            raise ValueError(
                "prediction_type must be one of ['knowledge', 'data', 'total'], "
                f"but got {prediction_type}"
            )

    setattr(estimator, "predict_var", predict_var)

    @functools.wraps(original_predict)
    def predict(
        data: Any,
        prediction_type: Literal[
            "Probability", "Class", "RawFormulaVal", "Exponent", "LogProbability"
        ] = "RawFormulaVal",
        ntree_start: int = 0,
        ntree_end: int = 0,
        thread_count: int = -1,
        verbose: bool | None = None,
        task_type: str = "CPU",
        return_std: bool = False,
    ) -> NDArray[Any]:
        prediction = original_predict(
            data,
            prediction_type,
            ntree_start,
            ntree_end,
            thread_count,
            verbose,
            task_type,
        )
        if prediction.ndim == 2:
            return prediction[:, 0]
        if return_std:
            # see virtual_ensembles_predict() for details
            return (
                prediction,
                predict_var(
                    data,
                    ntree_end=ntree_end,  # 0
                    thread_count=thread_count,  # -1
                    verbose=verbose,  # None
                )
                ** 0.5,
            )
        return prediction

    setattr(estimator, "predict", predict)

    return estimator


TAny = TypeVar("TAny")


def patch(
    estimator: TAny,
    *,
    copy: bool = True,
    recursive: bool = True,
    recursive_strict: bool = False,
) -> TAny:
    """Patch estimator if it is supported. (`patch_ngboost` and `patch_catboost`.)
    The patch will not apply if the estimator is cloned using `sklearn.base.clone()`
    and requires re-patching.

    Parameters
    ----------
    estimator : TAny
        The estimator to patch.
    copy : bool, optional
        Whether to copy the estimator before patching, by default True
    recursive : bool, optional
        Whether to recursively patch the estimator, by default True
    recursive_strict : bool, optional
        Whether to recursively patch the estimator's attributes,
        lists, tuples, sets, and frozensets as well, by default False

    Returns
    -------
    TAny
        The patched estimator.
    """
    if recursive_strict and not recursive:
        raise ValueError("recursive_strict requires recursive=True")
    if copy:
        estimator = clone(estimator)
    if importlib.util.find_spec("ngboost") is not None:
        if isinstance(estimator, NGBoost):
            return patch_ngboost(estimator)
    if isinstance(estimator, cb.CatBoost):
        return patch_catboost(estimator)

    if recursive and hasattr(estimator, "get_params"):
        for _, value in estimator.get_params(deep=True).items():
            patch(value, copy=False, recursive=False, recursive_strict=False)
    if recursive_strict:
        if hasattr(estimator, "__dict__"):
            for _, value in estimator.__dict__.items():
                patch(value, copy=False, recursive=True, recursive_strict=True)
        elif isinstance(estimator, (list, tuple, set, frozenset)):
            # https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/base.py#L66
            for value in estimator:
                patch(value, copy=False, recursive=True, recursive_strict=True)

    return estimator
