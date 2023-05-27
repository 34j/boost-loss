from typing import Any, TypeVar

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from ._base import LossBase

TEstimator = TypeVar("TEstimator", cb.CatBoost, lgb.LGBMModel, xgb.XGBModel)


def apply_custom_loss(
    estimator: TEstimator,
    loss: LossBase,
    *,
    copy: bool = True,
    target_transformer: BaseEstimator | Any | None = StandardScaler()
) -> TEstimator | TransformedTargetRegressor:
    if copy:
        estimator = clone(estimator)
    if isinstance(estimator, cb.CatBoost):
        estimator.set_params(loss_function=loss, eval_metric=loss)
    if isinstance(estimator, lgb.LGBMModel):
        estimator.set_params(objective=loss, metric=[loss.eval_metric_lgb])
    if isinstance(estimator, xgb.XGBModel):
        estimator.set_params(objective=loss, eval_metric=loss.eval_metric_xgb_sklearn)
    if target_transformer is None:
        return estimator
    return TransformedTargetRegressor(estimator, transformer=clone(target_transformer))
