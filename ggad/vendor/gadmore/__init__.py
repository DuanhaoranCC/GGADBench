"""GAD-MoRE components used by the source-domain benchmark."""

from .model import GADMoRE
from .preprocess import mcfa, propagate_features

__all__ = ["GADMoRE", "mcfa", "propagate_features"]
