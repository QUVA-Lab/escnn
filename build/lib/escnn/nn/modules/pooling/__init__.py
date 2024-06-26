from .norm_max import NormMaxPool

from .pointwise_max import PointwiseMaxPool2D, PointwiseMaxPool
from .pointwise_max import PointwiseMaxPoolAntialiased2D, PointwiseMaxPoolAntialiased
from .pointwise_max import PointwiseMaxPool3D
from .pointwise_max import PointwiseMaxPoolAntialiased3D
from .pointwise_avg import PointwiseAvgPool, PointwiseAvgPool2D
from .pointwise_avg import PointwiseAvgPoolAntialiased, PointwiseAvgPoolAntialiased2D
from .pointwise_adaptive_avg import PointwiseAdaptiveAvgPool2D, PointwiseAdaptiveAvgPool
from .pointwise_adaptive_avg import PointwiseAdaptiveAvgPool3D
from .pointwise_adaptive_max import PointwiseAdaptiveMaxPool2D, PointwiseAdaptiveMaxPool
from .pointwise_adaptive_max import PointwiseAdaptiveMaxPool3D

from .pointwise_avg_3d import PointwiseAvgPool3D
from .pointwise_avg_3d import PointwiseAvgPoolAntialiased3D


__all__ = [
    "NormMaxPool",
    "PointwiseMaxPool2D", "PointwiseMaxPool",
    "PointwiseMaxPool3D",
    "PointwiseMaxPoolAntialiased2D", "PointwiseMaxPoolAntialiased",
    "PointwiseMaxPoolAntialiased3D",
    "PointwiseAvgPool2D", "PointwiseAvgPool",
    "PointwiseAvgPoolAntialiased2D", "PointwiseAvgPoolAntialiased",
    "PointwiseAdaptiveAvgPool2D", "PointwiseAdaptiveAvgPool",
    "PointwiseAdaptiveAvgPool3D",
    "PointwiseAdaptiveMaxPool2D", "PointwiseAdaptiveMaxPool",
    "PointwiseAdaptiveMaxPool3D",
    ###### 3D ##################
    "PointwiseAvgPool3D",
    "PointwiseAvgPoolAntialiased3D",
]


