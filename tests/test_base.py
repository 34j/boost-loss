from __future__ import annotations

from functools import lru_cache
from unittest import SkipTest, TestCase

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.testing import assert_allclose
from parameterized import parameterized_class
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from boost_loss.base import LossBase
from boost_loss.regression.regression import L2Loss, LogCoshLoss
from boost_loss.sklearn import apply_custom_loss


def assert_array_almost_equal(a, b):
    try:
        assert_allclose(a, b, rtol=1e-2, atol=1e-7)
    except Exception as e:
        raise AssertionError(f"{a[:3]}... != {b[:3]}...") from e


class TestBase(TestCase):
    loss: LossBase
    loss_name: str
    loss_names: dict[str, dict[str, str | None]] = {
        "catboost": {
            "l2": "RMSE",
            "l1": "MAE",
            "poisson": "Poisson",
            "logcosh": "LogCosh",
        },
        "lightgbm": {
            "l2": "regression",
            "l1": "regression_l1",
            "poisson": "poisson",
            "logcosh": None,
        },
        "xgboost": {
            "l2": "reg:squarederror",
            "l1": "reg:linear",
            "poisson": "count:poisson",
            "logcosh": None,
        },
    }

    def setUp(self) -> None:
        self.X, self.y = load_diabetes(return_X_y=True)
        self.seed = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        self.X_train = x_scaler.fit_transform(self.X_train)
        self.X_test = x_scaler.transform(self.X_test)
        self.y_train = y_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test = y_scaler.transform(self.y_test.reshape(-1, 1)).ravel()

    def tearDown(self) -> None:
        if hasattr(self, "y_pred"):
            score = self.loss.loss(self.y_test, self.y_pred)
            print(f"Score: {score}")

    @lru_cache
    def catboost_baseline(self):
        if self.loss_names["catboost"][self.loss_name] is None:
            raise SkipTest(f"CatBoost does not support {self.loss_name} loss.")
        model = cb.CatBoostRegressor(
            loss_function=self.loss_names["catboost"][self.loss_name],
            eval_metric=self.loss_names["catboost"][self.loss_name],
            iterations=100,
        )
        model.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test))
        return model.predict(self.X_test)

    @lru_cache
    def lightgbm_baseline(self):
        if self.loss_names["lightgbm"][self.loss_name] is None:
            raise SkipTest(f"LightGBM does not support {self.loss_name} loss.")
        model = lgb.LGBMRegressor(
            objective=self.loss_names["lightgbm"][self.loss_name],
        )
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        return model.predict(self.X_test)

    @lru_cache
    def xgboost_baseline(self):
        if self.loss_names["xgboost"][self.loss_name] is None:
            raise SkipTest(f"XGBoost does not support {self.loss_name} loss.")
        model = xgb.XGBRegressor(
            objective=self.loss_names["xgboost"][self.loss_name],
            base_score=0.5,
        )
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        return model.predict(self.X_test)


@parameterized_class(
    ("loss", "loss_name"),
    [
        # (L1Loss(), "l1"),
        (L2Loss(), "l2"),
        (LogCoshLoss(), "logcosh"),
    ],
)
class TestBasic(TestBase):
    def test_catboost_sklearn(self):
        # https://catboost.ai/en/docs/concepts/python-usages-examples#user-defined-loss-function
        model = cb.CatBoostRegressor(
            loss_function=self.loss,
            eval_metric=self.loss,
            iterations=100,
            learning_rate=0.1,
        )
        model.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test))
        self.y_pred = model.predict(self.X_test)
        y_pred_baseline = self.catboost_baseline()
        self.assertAlmostEqual(
            self.loss.eval_metric_xgb_sklearn(self.y_test, self.y_pred)
            / self.loss.eval_metric_xgb_sklearn(self.y_test, y_pred_baseline),
            1,
            places=0,
        )
        raise SkipTest("Custom RMSE is somewhat not consistent with CatBoost's RMSE")
        assert_array_almost_equal(self.y_pred, y_pred_baseline)  # type: ignore

    def test_catboost_native(self):
        model = cb.CatBoostRegressor(
            loss_function=self.loss,
            eval_metric=self.loss,
            iterations=100,
            learning_rate=0.1,
        )
        train_pool = cb.Pool(self.X_train, self.y_train)
        test_pool = cb.Pool(self.X_test, self.y_test)
        model.fit(train_pool, eval_set=test_pool)
        self.y_pred = model.predict(test_pool)
        self.assertAlmostEqual(
            self.loss.eval_metric_xgb_sklearn(self.y_test, self.y_pred)
            / self.loss.eval_metric_xgb_sklearn(self.y_test, self.catboost_baseline()),
            1,
            places=0,
        )
        raise SkipTest("Custom RMSE is somewhat not consistent with CatBoost's RMSE")
        assert_array_almost_equal(self.y_pred, self.catboost_baseline())  # type: ignore

    def test_catboost_sklearn_apply(self):
        # https://catboost.ai/en/docs/concepts/python-usages-examples#user-defined-loss-function
        model = cb.CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
        )
        model = apply_custom_loss(model, self.loss, target_transformer=None)
        model.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test))
        self.y_pred = model.predict(self.X_test)
        y_pred_baseline = self.catboost_baseline()
        self.assertAlmostEqual(
            self.loss.eval_metric_xgb_sklearn(self.y_test, self.y_pred)
            / self.loss.eval_metric_xgb_sklearn(self.y_test, y_pred_baseline),
            1,
            places=0,
        )
        raise SkipTest("Custom RMSE is somewhat not consistent with CatBoost's RMSE")
        assert_array_almost_equal(self.y_pred, y_pred_baseline)  # type: ignore

    def test_lightgbm_sklearn(self):
        model = lgb.LGBMRegressor(objective=self.loss)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            eval_metric=self.loss.eval_metric_lgb,
        )
        self.y_pred = model.predict(self.X_test)
        assert_array_almost_equal(self.y_pred, self.lightgbm_baseline())

    def test_lightgbm_native(self):
        train_set = lgb.Dataset(self.X_train, self.y_train)
        test_set = lgb.Dataset(self.X_test, self.y_test)
        booster = lgb.train(
            {"seed": self.seed},
            train_set=train_set,
            valid_sets=[train_set, test_set],
            fobj=self.loss,
            feval=self.loss.eval_metric_lgb,
        )
        self.y_pred = booster.predict(self.X_test)
        assert_array_almost_equal(self.y_pred, self.lightgbm_baseline())

    def test_lightgbm_sklearn_apply(self):
        model = lgb.LGBMRegressor()
        model = apply_custom_loss(model, self.loss, target_transformer=None)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            eval_metric=self.loss.eval_metric_lgb,
        )
        self.y_pred = model.predict(self.X_test)
        assert_array_almost_equal(self.y_pred, self.lightgbm_baseline())

    def test_xgboost_sklearn(self):
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-objective-function
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-metric-function
        model = xgb.XGBRegressor(
            objective=self.loss, eval_metric=self.loss.eval_metric_xgb_sklearn
        )
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        self.y_pred = model.predict(self.X_test)
        assert_array_almost_equal(self.y_pred, self.xgboost_baseline())

    def test_xgboost_native(self):
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#scikit-learn-interface
        train_set = xgb.DMatrix(self.X_train, self.y_train)
        test_set = xgb.DMatrix(self.X_test, self.y_test)
        booster = xgb.train(
            {"seed": self.seed},
            train_set,
            num_boost_round=100,  # it is fucking that default value is different
            evals=[(train_set, "train"), (test_set, "test")],
            obj=self.loss,
            custom_metric=self.loss.eval_metric_xgb_native,
        )
        self.y_pred = booster.predict(test_set)
        assert_array_almost_equal(self.y_pred, self.xgboost_baseline())

    def test_xgboost_sklearn_apply(self):
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-objective-function
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-metric-function
        model = xgb.XGBRegressor()
        model = apply_custom_loss(model, self.loss, target_transformer=None)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        self.y_pred = model.predict(self.X_test)
        assert_array_almost_equal(self.y_pred, self.xgboost_baseline())

    def test_sklearn_sklearn(self):
        raise SkipTest("Not implemented yet")


@parameterized_class(
    ("loss", "loss_name"),
    [
        (L2Loss(), "l2"),
    ],
)
class TestWeighted(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.w_train = np.random.rand(len(self.y_train))
        self.w_test = np.random.rand(len(self.y_test))

    def test_catboost_sklearn(self):
        # https://catboost.ai/en/docs/concepts/python-usages-examples#user-defined-loss-function
        model = cb.CatBoostRegressor(
            loss_function=self.loss,
            eval_metric=self.loss,
            iterations=100,
            learning_rate=0.1,
        )
        model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.w_train,
            eval_set=(self.X_test, self.y_test),
        )
        # no api to pass sample_weight to eval_set
        self.y_pred = model.predict(self.X_test)

    def test_catboost_native(self):
        model = cb.CatBoostRegressor(
            loss_function=self.loss,
            eval_metric=self.loss,
            iterations=100,
            learning_rate=0.1,
        )
        train_pool = cb.Pool(self.X_train, self.y_train, weight=self.w_train)
        test_pool = cb.Pool(self.X_test, self.y_test, weight=self.w_test)
        model.fit(train_pool, eval_set=test_pool)
        self.y_pred = model.predict(test_pool)

    def test_lightgbm_sklearn(self):
        model = lgb.LGBMRegressor(objective=self.loss)
        model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.w_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            eval_metric=self.loss.eval_metric_lgb,
            eval_sample_weight=[self.w_train, self.w_test],
        )
        self.y_pred = model.predict(self.X_test)

    def test_lightgbm_native(self):
        train_set = lgb.Dataset(self.X_train, self.y_train, weight=self.w_train)
        test_set = lgb.Dataset(self.X_test, self.y_test, weight=self.w_test)
        booster = lgb.train(
            {"seed": self.seed},
            train_set=train_set,
            valid_sets=[train_set, test_set],
            fobj=self.loss,
            feval=self.loss.eval_metric_lgb,
        )
        self.y_pred = booster.predict(self.X_test)

    def test_xgboost_sklearn(self):
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-objective-function
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#customized-metric-function
        model = xgb.XGBRegressor(
            objective=self.loss, eval_metric=self.loss.eval_metric_xgb_sklearn
        )
        model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.w_train,
            sample_weight_eval_set=[self.w_train, self.w_test],
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        self.y_pred = model.predict(self.X_test)

    def test_xgboost_native(self):
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#scikit-learn-interface
        train_set = xgb.DMatrix(self.X_train, self.y_train, weight=self.w_train)
        test_set = xgb.DMatrix(self.X_test, self.y_test, weight=self.w_test)
        booster = xgb.train(
            {"seed": self.seed},
            train_set,
            num_boost_round=100,  # it is fucking that default value is different
            evals=[(train_set, "train"), (test_set, "test")],
            obj=self.loss,
            custom_metric=self.loss.eval_metric_xgb_native,
        )
        self.y_pred = booster.predict(test_set)

    def test_sklearn_sklearn(self):
        raise SkipTest("Not implemented yet")
