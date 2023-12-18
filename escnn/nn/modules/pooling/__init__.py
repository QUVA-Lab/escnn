from .norm_max import NormMaxPool

from .pointwise_max import (
        PointwiseMaxPool2D, PointwiseMaxPool,
        PointwiseMaxPool3D,

        PointwiseAdaptiveMaxPool2D, PointwiseAdaptiveMaxPool,
        PointwiseAdaptiveMaxPool3D,

        PointwiseMaxPoolAntialiased2D, PointwiseMaxPoolAntialiased,
        PointwiseMaxPoolAntialiased3D,
)
from .pointwise_avg import (
        PointwiseAvgPool2D, PointwiseAvgPool,
        PointwiseAvgPool3D,

        PointwiseAdaptiveAvgPool2D, PointwiseAdaptiveAvgPool,
        PointwiseAdaptiveAvgPool3D,
        
        PointwiseAvgPoolAntialiased2D, PointwiseAvgPoolAntialiased,
        PointwiseAvgPoolAntialiased3D,
)

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


