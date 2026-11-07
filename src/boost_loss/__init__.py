__version__ = "0.5.5"
import contextlib

from .base import LossBase
from .debug import DebugLoss, PrintLoss
from .resuming import ResumingLoss
from .sklearn import apply_custom_loss, patch_catboost

with contextlib.suppress(ImportError):
    from .sklearn import patch_ngboost
__all__ = [
    "DebugLoss",
    "LossBase",
    "PrintLoss",
    "ResumingLoss",
    "apply_custom_loss",
    "patch_catboost",
    "patch_ngboost",
]
