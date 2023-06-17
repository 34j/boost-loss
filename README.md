# Boost Loss

<p align="center">
  <a href="https://github.com/34j/boost-loss/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/boost-loss/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://boost-loss.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/boost-loss.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/boost-loss">
    <img src="https://img.shields.io/codecov/c/github/34j/boost-loss.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/boost-loss/">
    <img src="https://img.shields.io/pypi/v/boost-loss.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/boost-loss.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/boost-loss.svg?style=flat-square" alt="License">
</p>

Utilities for easy use of custom losses in CatBoost, LightGBM, XGBoost. This sounds very simple, but in reality it took a lot of work.

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install boost-loss
```

## Usage

### Basic Usage

```python
import numpy as np

from boost_loss import LossBase
from numpy.typing import NDArray


class L2Loss(LossBase):
    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (y_true - y_pred) ** 2 / 2

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray: # dL/dy_pred
        return - (y_true - y_pred)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray: # d^2L/dy_pred^2
        return np.ones_like(y_true)
```

```python
import lightgbm as lgb

from boost_loss import apply_custom_loss
from sklearn.datasets import load_boston


X, y = load_boston(return_X_y=True)
apply_custom_loss(lgb.LGBMRegressor(), L2Loss()).fit(X, y)
```

Built-in losses are available. [^bokbokbok]

```python
from boost_loss.regression import LogCoshLoss
```

### [`torch.autograd`](https://pytorch.org/docs/stable/autograd.html) Loss [^autograd]

```python
import torch

from boost_loss.torch import TorchLossBase


class L2LossTorch(TorchLossBase):
    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return (y_true - y_pred) ** 2 / 2
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/34j"><img src="https://avatars.githubusercontent.com/u/55338215?v=4?s=80" width="80px;" alt="34j"/><br /><sub><b>34j</b></sub></a><br /><a href="https://github.com/34j/boost-loss/commits?author=34j" title="Code">💻</a> <a href="#ideas-34j" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/34j/boost-loss/commits?author=34j" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

[^bokbokbok]: Inspired by [orchardbirds/bokbokbok](https://github.com/orchardbirds/bokbokbok)
[^autograd]: Inspired by [TomerRonen34/treeboost_autograd](https://github.com/TomerRonen34/treeboost_autograd)
