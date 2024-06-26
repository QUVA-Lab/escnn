
from .basis import EmptyBasisException, KernelBasis, AdjointBasis, UnionBasis

from .harmonic_polynomial_r3 import HarmonicPolynomialR3Generator

from .steerable_filters_basis import SteerableFiltersBasis, PointBasis
from .polar_basis import GaussianRadialProfile, SphericalShellsBasis, CircularShellsBasis
from .sparse_basis import SparseOrbitBasis, SparseOrbitBasisWithIcosahedralSymmetry

from .steerable_basis import SteerableKernelBasis, IrrepBasis

from .wignereckart_solver import WignerEckartBasis, RestrictedWignerEckartBasis

from .r2 import *
from .r3 import *


import escnn.group


def kernels_on_point(in_repr: escnn.group.Representation, out_repr: escnn.group.Representation) -> KernelBasis:
    r"""

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    basis = SteerableKernelBasis(
        PointBasis(group), in_repr, out_repr,
        WignerEckartBasis,
    )

    return basis


__all__ = [
    "EmptyBasisException",
    "KernelBasis",
    # General Bases
    "AdjointBasis",
    "UnionBasis",
    # Steerable Kernel Bases
    'SteerableKernelBasis',
    "IrrepBasis",
    "WignerEckartBasis",
    "RestrictedWignerEckartBasis",
    # Steerable Filters Bases
    'SteerableFiltersBasis',
    'PointBasis',
    "SphericalShellsBasis",
    "CircularShellsBasis",
    'GaussianRadialProfile',
    "SparseOrbitBasis",
    "SparseOrbitBasisWithIcosahedralSymmetry",
    # Other useful modules
    "HarmonicPolynomialR3Generator",
    # Generic group acting on R^0
    "kernels_on_point",
    # R2 bases
    "kernels_Flip_act_R2",
    "kernels_DN_act_R2",
    "kernels_O2_act_R2",
    "kernels_Trivial_act_R2",
    "kernels_CN_act_R2",
    "kernels_SO2_act_R2",
    "kernels_SO2_subgroup_act_R2",
    "kernels_O2_subgroup_act_R2",
    # R3 bases
    "kernels_O3_act_R3",
    "kernels_SO3_act_R3",
    "kernels_SO3_subgroup_act_R3",
    "kernels_O3_subgroup_act_R3",
    "kernels_Ico_act_R3",
    "kernels_Octa_act_R3",
    "kernels_Tetra_act_R3",
    "kernels_FullIco_act_R3",
    "kernels_FullOcta_act_R3",
    "kernels_FullTetra_act_R3",
    "kernels_Pyrito_act_R3",
    "kernels_SO2_act_R3",
    "kernels_CN_act_R3",
    "kernels_O2_conical_act_R3",
    "kernels_DN_conical_act_R3",
    "kernels_O2_dihedral_act_R3",
    "kernels_DN_dihedral_act_R3",
    "kernels_Inv_act_R3",
    "kernels_Trivial_act_R3",
    'kernels_aliased_Ico_act_R3_dodecahedron',
    'kernels_aliased_Ico_act_R3_icosidodecahedron',
    'kernels_aliased_Ico_act_R3_icosahedron',
]
