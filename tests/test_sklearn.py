from unittest import TestCase

import pytest
from catboost import CatBoostClassifier, CatboostError, CatBoostRegressor
from parameterized import parameterized
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from boost_loss.sklearn import patch, patch_catboost


class TestPatchCatboost(TestCase):
    @parameterized.expand(["RMSE", "RMSEWithUncertainty"])
    def test_patch_catboost_reg(self, objective: str) -> None:
        estimator = patch_catboost(CatBoostRegressor(iterations=100, objective=objective))
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_pred_var = estimator.predict_var(X_test)
        assert y_test.shape == y_pred.shape
        assert y_test.shape == y_pred_var.shape

        assert not hasattr(CatBoostRegressor(), "predict_var")

        # assert method properly created for each object
        with pytest.raises(CatboostError):
            patch_catboost(CatBoostRegressor(iterations=100, objective=objective)).predict_var(
                X_test
            )

    def test_patch_catboost_clf(self) -> None:
        estimator = patch_catboost(CatBoostClassifier(iterations=100, loss_function="MultiClass"))
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_pred_var = estimator.predict_var(X_test)
        assert y_test.shape == y_pred.shape
        assert y_test.shape == y_pred_var.shape

        assert not hasattr(CatBoostClassifier(), "predict_var")

        with pytest.raises(CatboostError):
            patch_catboost(
                CatBoostClassifier(iterations=100, loss_function="MultiClass")
            ).predict_var(X_test)

    def test_patch(self) -> None:
        estimator = patch(
            Pipeline([
                (
                    "catboost",
                    CatBoostClassifier(iterations=100, loss_function="MultiClass"),
                )
            ])
        )
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_pred_var = estimator[-1].predict_var(X_test)
        assert y_test.shape == y_pred.shape
        assert y_test.shape == y_pred_var.shape


try:
    from ngboost import NGBRegressor

    from boost_loss.sklearn import patch_ngboost

    class TestPatchNGBoost(TestCase):
        def setUp(self) -> None:
            X, y = load_diabetes(return_X_y=True)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        def test_patch_ngboost(self) -> None:
            estimator = patch_ngboost(NGBRegressor(n_estimators=100))
            estimator.fit(self.X_train, self.y_train)
            y_pred = estimator.predict(self.X_test)
            y_pred_var = estimator.predict_var(self.X_test)
            assert self.y_test.shape == y_pred.shape
            assert self.y_test.shape == y_pred_var.shape

            assert not hasattr(NGBRegressor(), "predict_var")

            with pytest.raises(BaseException):
                patch_ngboost(NGBRegressor(n_estimators=100)).predict_var(self.X_test)

except ImportError:
    pass
