from .norm_max import NormMaxPool

from .pointwise_max import PointwiseMaxPool
from .pointwise_max import PointwiseMaxPoolAntialiased
from .pointwise_avg import PointwiseAvgPool, PointwiseAvgPool2D
from .pointwise_avg import PointwiseAvgPoolAntialiased, PointwiseAvgPoolAntialiased2D
from .pointwise_adaptive_avg import PointwiseAdaptiveAvgPool
from .pointwise_adaptive_max import PointwiseAdaptiveMaxPool

from .pointwise_avg_3d import PointwiseAvgPool3D
from .pointwise_avg_3d import PointwiseAvgPoolAntialiased3D


__all__ = [
    "NormMaxPool",
    "PointwiseMaxPool",
    "PointwiseMaxPoolAntialiased",
    "PointwiseAvgPool2D", "PointwiseAvgPool",
    "PointwiseAvgPoolAntialiased2D", "PointwiseAvgPoolAntialiased",
    "PointwiseAdaptiveAvgPool",
    "PointwiseAdaptiveMaxPool",
    ###### 3D ##################
    "PointwiseAvgPool3D",
    "PointwiseAvgPoolAntialiased3D",
]


