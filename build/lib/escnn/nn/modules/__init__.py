from .equivariant_module import EquivariantModule

from .branching_module import BranchingModule
from .merge_module import MergeModule
from .multiple_module import MultipleModule

from .rdupsampling import R2Upsampling, R3Upsampling

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
from .nonlinearities import GatedNonLinearityUniform
from .nonlinearities import InducedGatedNonLinearity1
from .nonlinearities import NormNonLinearity
from .nonlinearities import InducedNormNonLinearity
from .nonlinearities import PointwiseNonLinearity
from .nonlinearities import ConcatenatedNonLinearity
from .nonlinearities import VectorFieldNonLinearity
from .nonlinearities import ReLU
from .nonlinearities import ELU
from .nonlinearities import LeakyReLU
from .nonlinearities import FourierPointwise
from .nonlinearities import FourierELU
from .nonlinearities import QuotientFourierPointwise
from .nonlinearities import QuotientFourierELU
from .nonlinearities import TensorProductModule

from .reshuffle_module import ReshuffleModule

from .pooling import NormMaxPool
from .pooling import PointwiseMaxPool2D, PointwiseMaxPool
from .pooling import PointwiseMaxPoolAntialiased2D, PointwiseMaxPoolAntialiased
from .pooling import PointwiseMaxPool3D
from .pooling import PointwiseMaxPoolAntialiased3D
from .pooling import PointwiseAvgPool, PointwiseAvgPool2D
from .pooling import PointwiseAvgPoolAntialiased, PointwiseAvgPoolAntialiased2D
from .pooling import PointwiseAdaptiveAvgPool2D, PointwiseAdaptiveAvgPool
from .pooling import PointwiseAdaptiveAvgPool3D
from .pooling import PointwiseAdaptiveMaxPool2D, PointwiseAdaptiveMaxPool
from .pooling import PointwiseAdaptiveMaxPool3D
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

from .normalization import FieldNorm

from .restriction_module import RestrictionModule
from .disentangle_module import DisentangleModule

from .dropout import FieldDropout
from .dropout import PointwiseDropout

from .sequential_module import SequentialModule
from .identity_module import IdentityModule

from .masking_module import MaskModule

from .harmonic_polynomial_r3 import HarmonicPolynomialR3


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
    "R3Upsampling",
    "GatedNonLinearity1",
    "GatedNonLinearity2",
    "GatedNonLinearityUniform",
    "InducedGatedNonLinearity1",
    "NormNonLinearity",
    "InducedNormNonLinearity",
    "PointwiseNonLinearity",
    "ConcatenatedNonLinearity",
    "VectorFieldNonLinearity",
    "ReLU",
    "ELU",
    "LeakyReLU",
    "FourierPointwise",
    "FourierELU",
    "QuotientFourierPointwise",
    "QuotientFourierELU",
    "TensorProductModule",
    "ReshuffleModule",
    "NormMaxPool",
    "PointwiseMaxPool2D", "PointwiseMaxPool",
    "PointwiseMaxPool3D",
    "PointwiseMaxPoolAntialiased2D", "PointwiseMaxPoolAntialiased",
    "PointwiseMaxPoolAntialiased3D",
    "PointwiseAvgPool2D", "PointwiseAvgPool",
    "PointwiseAvgPoolAntialiased2D", "PointwiseAvgPoolAntialiased",
    "PointwiseAvgPool3D",
    "PointwiseAvgPoolAntialiased3D",
    "PointwiseAdaptiveAvgPool2D", "PointwiseAdaptiveAvgPool",
    "PointwiseAdaptiveAvgPool3D",
    "PointwiseAdaptiveMaxPool2D", "PointwiseAdaptiveMaxPool",
    "PointwiseAdaptiveMaxPool3D",
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
    "FieldNorm",
    "RestrictionModule",
    "DisentangleModule",
    "FieldDropout",
    "PointwiseDropout",
    "SequentialModule",
    "IdentityModule",
    "MaskModule",
    "HarmonicPolynomialR3",
]
