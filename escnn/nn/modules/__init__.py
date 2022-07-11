from .equivariant_module import EquivariantModule

from .branching_module import BranchingModule
from .merge_module import MergeModule
from .multiple_module import MultipleModule

from .r2upsampling import R2Upsampling

from .conv import R3Conv
from .conv import R2Conv
from .conv import R2ConvTransposed
from .conv import R3ConvTransposed
from .conv import R3IcoConv
from .conv import R3IcoConvTransposed

try:
    from .pointconv import R2PointConv
    from .pointconv import R3PointConv
    _point_conv_modules = ['R2PointConv', 'R3PointConv']
except ImportError:
    _point_conv_modules = []


from .linear import Linear

from .nonlinearities import GatedNonLinearity1
from .nonlinearities import GatedNonLinearity2
from .nonlinearities import InducedGatedNonLinearity1
from .nonlinearities import NormNonLinearity
from .nonlinearities import InducedNormNonLinearity
from .nonlinearities import PointwiseNonLinearity
from .nonlinearities import ConcatenatedNonLinearity
from .nonlinearities import VectorFieldNonLinearity
from .nonlinearities import ReLU
from .nonlinearities import ELU
from .nonlinearities import FourierPointwise
from .nonlinearities import FourierELU
from .nonlinearities import QuotientFourierPointwise
from .nonlinearities import QuotientFourierELU
from .nonlinearities import TensorProductModule

from .reshuffle_module import ReshuffleModule

from .pooling import NormMaxPool
from .pooling import PointwiseMaxPool
from .pooling import PointwiseMaxPoolAntialiased
from .pooling import PointwiseAvgPool
from .pooling import PointwiseAvgPoolAntialiased
from .pooling import PointwiseAdaptiveAvgPool
from .pooling import PointwiseAdaptiveMaxPool
from .pooling import PointwiseAvgPool3D
from .pooling import PointwiseAvgPoolAntialiased3D

from .invariantmaps import GroupPooling
from .invariantmaps import MaxPoolChannels
from .invariantmaps import NormPool
from .invariantmaps import InducedNormPool

from .batchnormalization import InnerBatchNorm
from .batchnormalization import NormBatchNorm
from .batchnormalization import InducedNormBatchNorm
from .batchnormalization import GNormBatchNorm
from .batchnormalization import IIDBatchNorm1d, IIDBatchNorm2d, IIDBatchNorm3d

from .restriction_module import RestrictionModule
from .disentangle_module import DisentangleModule

from .dropout import FieldDropout
from .dropout import PointwiseDropout

from .sequential_module import SequentialModule
from .identity_module import IdentityModule

from .masking_module import MaskModule

__all__ = [
    "EquivariantModule",
    "BranchingModule",
    "MergeModule",
    "MultipleModule",
    "Linear",
] + _point_conv_modules + [
    "R3Conv",
    "R2Conv",
    "R2ConvTransposed",
    "R3ConvTransposed",
    "R3IcoConv",
    "R3IcoConvTransposed",
    "R2Upsampling",
    "GatedNonLinearity1",
    "GatedNonLinearity2",
    "InducedGatedNonLinearity1",
    "NormNonLinearity",
    "InducedNormNonLinearity",
    "PointwiseNonLinearity",
    "ConcatenatedNonLinearity",
    "VectorFieldNonLinearity",
    "ReLU",
    "ELU",
    "FourierPointwise",
    "FourierELU",
    "QuotientFourierPointwise",
    "QuotientFourierELU",
    "TensorProductModule",
    "ReshuffleModule",
    "NormMaxPool",
    "PointwiseMaxPool",
    "PointwiseMaxPoolAntialiased",
    "PointwiseAvgPool",
    "PointwiseAvgPoolAntialiased",
    "PointwiseAvgPool3D",
    "PointwiseAvgPoolAntialiased3D",
    "PointwiseAdaptiveAvgPool",
    "PointwiseAdaptiveMaxPool",
    "GroupPooling",
    "MaxPoolChannels",
    "NormPool",
    "InducedNormPool",
    "InnerBatchNorm",
    "NormBatchNorm",
    "InducedNormBatchNorm",
    "GNormBatchNorm",
    "IIDBatchNorm1d",
    "IIDBatchNorm2d",
    "IIDBatchNorm3d",
    "RestrictionModule",
    "DisentangleModule",
    "FieldDropout",
    "PointwiseDropout",
    "SequentialModule",
    "IdentityModule",
    "MaskModule",
]
