__version__ = "0.2.0"
from .base import LossBase
from .debug import DebugLoss, PrintLoss
from .resuming import ResumingLoss
from .sklearn import apply_custom_loss, patch_catboost

try:
    from .sklearn import patch_ngboost
except ImportError:
    pass
__all__ = [
    "LossBase",
    "DebugLoss",
    "PrintLoss",
    "ResumingLoss",
    "apply_custom_loss",
    "patch_catboost",
    "patch_ngboost",
]
