from escnn.kernels.basis import KernelBasis, AdjointBasis, UnionBasis, EmptyBasisException
from escnn.kernels.steerable_basis import SteerableKernelBasis
from escnn.kernels.polar_basis import GaussianRadialProfile, SphericalShellsBasis
from escnn.kernels.wignereckart_solver import WignerEckartBasis, RestrictedWignerEckartBasis
from escnn.kernels.sparse_basis import SparseSteerableBasis

from escnn.kernels.spaces import SphereO3, SphereSO3
from escnn.kernels.spaces import Dodecahedron, Icosahedron, Icosidodecahedron, PointRn

from escnn.group import *

import numpy as np

from typing import List, Union, Callable, Dict, Tuple

__all__ = [
    "kernels_SO3_act_R3",
    "kernels_O3_act_R3",
    ###################
    "kernels_SO3_subgroup_act_R3",
    "kernels_O3_subgroup_act_R3",
    ###################
    "kernels_Ico_act_R3",
    "kernels_Octa_act_R3",
    "kernels_Tetra_act_R3",
    "kernels_FullIco_act_R3",
    "kernels_FullOcta_act_R3",
    "kernels_FullTetra_act_R3",
    "kernels_Pyrito_act_R3",
    ###################
    "kernels_SO2_act_R3",
    "kernels_CN_act_R3",
    "kernels_O2_conical_act_R3",
    "kernels_DN_conical_act_R3",
    "kernels_O2_dihedral_act_R3",
    "kernels_DN_dihedral_act_R3",
    "kernels_full_cylinder_act_R3",
    "kernels_full_cylinder_discrete_act_R3",
    "kernels_cylinder_act_R3",
    "kernels_cylinder_discrete_act_R3",
    "kernels_Inv_act_R3",
    "kernels_Trivial_act_R3",
    ##################
    'kernels_aliased_Ico_act_R3_dodecahedron',
    'kernels_aliased_Ico_act_R3_icosidodecahedron',
    'kernels_aliased_Ico_act_R3_icosahedron',
]


def kernels_SO3_act_R3(in_repr: Representation, out_repr: Representation,
                       radii: List[float],
                       sigma: Union[List[float], float],
                       maximum_frequency: int = None,
                       adjoint: np.ndarray = None,
                       filter: Callable[[Dict], bool] = None
                       ) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to continuous rotations, modeled by the
    group :math:`SO(3)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.SO3`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        adjoint (~numpy.ndarray, optional): 3x3 orthogonal matrix defining a change of basis on the base space

    """
    assert in_repr.group == out_repr.group
    
    group = in_repr.group
    
    assert isinstance(group, SO3)

    if maximum_frequency is not None:
        harmonics = [(l,) for l in range(maximum_frequency+1)]
    else:
        harmonics = None
        
    angular_basis = SteerableKernelBasis(SphereSO3(), in_repr, out_repr, WignerEckartBasis, harmonics=harmonics)
    radial_profile = GaussianRadialProfile(radii, sigma)

    basis = SphericalShellsBasis(3, angular_basis, radial_profile, filter=filter)

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)
    
    return basis


def kernels_O3_act_R3(in_repr: Representation, out_repr: Representation,
                      radii: List[float],
                      sigma: Union[List[float], float],
                      maximum_frequency: int = None,
                      adjoint: np.ndarray = None,
                      filter: Callable[[Dict], bool] = None
                      ) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to reflections and continuous rotations, modeled by the
    group :math:`O(3)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.O3`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    Because :math:`O(3)` contains all rotations, the reflection element of the group can be associated to any reflection
    axis.

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        adjoint (~numpy.ndarray, optional): 3x3 orthogonal matrix defining a change of basis on the base space

    """
    assert in_repr.group == out_repr.group
    
    group = in_repr.group
    assert isinstance(group, O3)

    if maximum_frequency is not None:
        harmonics = [(k, l) for l in range(maximum_frequency+1) for k in range(2)]
    else:
        harmonics = None
        
    angular_basis = SteerableKernelBasis(SphereO3(), in_repr, out_repr, WignerEckartBasis, harmonics=harmonics)
    radial_profile = GaussianRadialProfile(radii, sigma)

    basis = SphericalShellsBasis(3, angular_basis, radial_profile, filter=filter)

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)
    
    return basis


###### Automatic subgroups kernel bases

def kernels_O3_subgroup_act_R3(in_repr: Representation, out_repr: Representation,
                               sg_id: Tuple,
                               radii: List[float],
                               sigma: Union[List[float], float],
                               maximum_frequency: int = 5,
                               adjoint: np.ndarray = None,
                               filter: Callable[[Dict], bool] = None
                               ) -> KernelBasis:
    
    o3 = o3_group(maximum_frequency)
    
    group, _, _ = o3.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group
    
    harmonics = [(k, l) for l in range(maximum_frequency+1) for k in range(2)]
    angular_basis = SteerableKernelBasis(SphereO3(), in_repr, out_repr, RestrictedWignerEckartBasis,
                                         sg_id=sg_id, harmonics=harmonics)
    radial_profile = GaussianRadialProfile(radii, sigma)
    
    basis = SphericalShellsBasis(3, angular_basis, radial_profile, filter=filter)
    
    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)
    
    return basis


def kernels_SO3_subgroup_act_R3(in_repr: Representation, out_repr: Representation,
                                sg_id: Tuple,
                                radii: List[float],
                                sigma: Union[List[float], float],
                                maximum_frequency: int = 5,
                                adjoint: np.ndarray = None,
                                filter: Callable[[Dict], bool] = None
                                ) -> KernelBasis:
    
    so3 = so3_group(maximum_frequency)
    
    group, _, _ = so3.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group
    
    harmonics = [(l,) for l in range(maximum_frequency+1)]
    angular_basis = SteerableKernelBasis(SphereSO3(), in_repr, out_repr, RestrictedWignerEckartBasis,
                                         sg_id=sg_id, harmonics=harmonics)
    radial_profile = GaussianRadialProfile(radii, sigma)
    
    basis = SphericalShellsBasis(3, angular_basis, radial_profile, filter=filter)
    
    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)
    
    return basis


###### Platonic Symmetries

def kernels_Ico_act_R3(in_repr: Representation, out_repr: Representation,
                       radii: List[float],
                       sigma: Union[List[float], float],
                       maximum_frequency: int = 5,
                       adjoint: np.ndarray = None,
                       filter: Callable[[Dict], bool] = None
                       ) -> KernelBasis:
    sg_id = 'ico'
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_Octa_act_R3(in_repr: Representation, out_repr: Representation,
                        radii: List[float],
                        sigma: Union[List[float], float],
                        maximum_frequency: int = 5,
                        adjoint: np.ndarray = None,
                        filter: Callable[[Dict], bool] = None
                        ) -> KernelBasis:
    sg_id = 'octa'
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_Tetra_act_R3(in_repr: Representation, out_repr: Representation,
                         radii: List[float],
                         sigma: Union[List[float], float],
                         maximum_frequency: int = 5,
                         adjoint: np.ndarray = None,
                         filter: Callable[[Dict], bool] = None
                         ) -> KernelBasis:
    sg_id = 'tetra'
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


###### Icosahedral Symmetry with Aliased samples

def kernels_aliased_Ico_act_R3_dodecahedron(in_repr: Representation, out_repr: Representation,
                                            radii: List[float],
                                            sigma: Union[List[float], float],
                                            adjoint: np.ndarray = None,
                                            ) -> KernelBasis:

    group = ico_group()

    assert in_repr.group == group
    assert out_repr.group == group

    dodecahedron = Dodecahedron()
    origin = PointRn(3, group)

    if isinstance(sigma, float):
        sigma = [sigma]*len(radii)
    assert len(sigma) == len(radii)

    basis_list = []
    for r, s in zip(radii, sigma):
        attributes = {'radius': r}
        try:
            if np.isclose(r, 0., rtol=1e-7, atol=1e-7):
                basis = SparseSteerableBasis(origin, in_repr, out_repr, sigma=s, attributes=attributes)
            else:
                change_of_basis = np.eye(3) * r
                basis = SparseSteerableBasis(dodecahedron, in_repr, out_repr, sigma=s, change_of_basis=change_of_basis, attributes=attributes)
            basis_list.append(basis)
        except EmptyBasisException:
            pass

    basis = UnionBasis(basis_list)

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)

    return basis


def kernels_aliased_Ico_act_R3_icosahedron(in_repr: Representation, out_repr: Representation,
                                           radii: List[float],
                                           sigma: Union[List[float], float],
                                           adjoint: np.ndarray = None,
                                           ) -> KernelBasis:

    group = ico_group()

    assert in_repr.group == group
    assert out_repr.group == group

    icosahedron = Icosahedron()
    origin = PointRn(3, group)

    if isinstance(sigma, float):
        sigma = [sigma]*len(radii)

    assert len(sigma) == len(radii)

    basis_list = []
    for r, s in zip(radii, sigma):
        attributes = {'radius': r}
        try:
            if np.isclose(r, 0., rtol=1e-7, atol=1e-7):
                basis = SparseSteerableBasis(origin, in_repr, out_repr, sigma=s, attributes=attributes)
            else:
                change_of_basis = np.eye(3) * r
                basis = SparseSteerableBasis(icosahedron, in_repr, out_repr, sigma=s, change_of_basis=change_of_basis, attributes=attributes)
            basis_list.append(basis)
        except EmptyBasisException:
            pass

    basis = UnionBasis(basis_list)

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)

    return basis


def kernels_aliased_Ico_act_R3_icosidodecahedron(in_repr: Representation, out_repr: Representation,
                                                 radii: List[float],
                                                 sigma: Union[List[float], float],
                                                 adjoint: np.ndarray = None,
                                                 ) -> KernelBasis:

    group = ico_group()

    assert in_repr.group == group
    assert out_repr.group == group

    icosahedron = Icosidodecahedron()
    origin = PointRn(3, group)

    if isinstance(sigma, float):
        sigma = [sigma]*len(radii)

    assert len(sigma) == len(radii)

    basis_list = []
    for r, s in zip(radii, sigma):
        attributes = {'radius': r}
        try:
            if np.isclose(r, 0., rtol=1e-7, atol=1e-7):
                basis = SparseSteerableBasis(origin, in_repr, out_repr, sigma=s, attributes=attributes)
            else:
                change_of_basis = np.eye(3) * r
                basis = SparseSteerableBasis(icosahedron, in_repr, out_repr, sigma=s, change_of_basis=change_of_basis, attributes=attributes)
            basis_list.append(basis)
        except EmptyBasisException:
            pass

    basis = UnionBasis(basis_list)

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)

    return basis


###### Full Platonic Symmetries


def kernels_FullIco_act_R3(in_repr: Representation, out_repr: Representation,
                           radii: List[float],
                           sigma: Union[List[float], float],
                           maximum_frequency: int = 5,
                           adjoint: np.ndarray = None,
                           filter: Callable[[Dict], bool] = None
                           ) -> KernelBasis:
    # I_h = I x C_2
    
    sg_id = (True, 'ico')
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_FullOcta_act_R3(in_repr: Representation, out_repr: Representation,
                            radii: List[float],
                            sigma: Union[List[float], float],
                            maximum_frequency: int = 5,
                            adjoint: np.ndarray = None,
                            filter: Callable[[Dict], bool] = None
                            ) -> KernelBasis:
    # O_h = O x C_2
    
    sg_id = (True, 'octa')
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_FullTetra_act_R3(in_repr: Representation, out_repr: Representation,
                             radii: List[float],
                             sigma: Union[List[float], float],
                             maximum_frequency: int = 5,
                             adjoint: np.ndarray = None,
                             filter: Callable[[Dict], bool] = None
                             ) -> KernelBasis:
    # group T_d
    # n.b. this is different from T x C_2
    
    sg_id = 'fulltetra'
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_Pyrito_act_R3(in_repr: Representation, out_repr: Representation,
                          radii: List[float],
                          sigma: Union[List[float], float],
                          maximum_frequency: int = 5,
                          adjoint: np.ndarray = None,
                          filter: Callable[[Dict], bool] = None
                          ) -> KernelBasis:
    # T_h = T x C_2
    # n.b. not a symmetry of the tetrahedron
    
    sg_id = (True, 'tetra')
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


###### Planar symmetries


def kernels_SO2_act_R3(in_repr: Representation, out_repr: Representation,
                       radii: List[float],
                       sigma: Union[List[float], float],
                       maximum_frequency: int = 5,
                       adjoint: np.ndarray = None,
                       filter: Callable[[Dict], bool] = None
                       ) -> KernelBasis:
    sg_id = (False, -1)
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_CN_act_R3(in_repr: Representation, out_repr: Representation,
                      radii: List[float],
                      sigma: Union[List[float], float],
                      maximum_frequency: int = 5,
                      max_offset: int = None,
                      adjoint: np.ndarray = None,
                      filter: Callable[[Dict], bool] = None
                      ) -> KernelBasis:
    assert in_repr.group == out_repr.group
    group = in_repr.group
    assert isinstance(group, CyclicGroup)

    # TODO implement max_offset as filter to apply on top later
    
    sg_id = (False, group.order())
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_O2_conical_act_R3(in_repr: Representation, out_repr: Representation,
                              radii: List[float],
                              sigma: Union[List[float], float],
                              maximum_frequency: int = 5,
                              adjoint: np.ndarray = None,
                              filter: Callable[[Dict], bool] = None
                              ) -> KernelBasis:
    sg_id = ('cone', -1)
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_DN_conical_act_R3(in_repr: Representation, out_repr: Representation,
                              radii: List[float],
                              sigma: Union[List[float], float],
                              # heights: List[float],
                              maximum_frequency: int = 5,
                              max_offset: int = None,
                              adjoint: np.ndarray = None,
                              filter: Callable[[Dict], bool] = None
                              ) -> KernelBasis:
    assert in_repr.group == out_repr.group
    group = in_repr.group
    assert isinstance(group, DihedralGroup)
    
    # TODO implement max_offset as filter to apply on top later

    sg_id = ('cone', group.rotation_order)
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_O2_dihedral_act_R3(in_repr: Representation, out_repr: Representation,
                               radii: List[float],
                               sigma: Union[List[float], float],
                               # heights: List[float],
                               maximum_frequency: int = 5,
                               adjoint: np.ndarray = None,
                               filter: Callable[[Dict], bool] = None
                               ) -> KernelBasis:

    sg_id = (True, -1)
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_DN_dihedral_act_R3(in_repr: Representation, out_repr: Representation,
                               radii: List[float],
                               sigma: Union[List[float], float],
                               # heights: List[float],
                               maximum_frequency: int = 5,
                               max_offset: int = None,
                               adjoint: np.ndarray = None,
                               filter: Callable[[Dict], bool] = None
                               ) -> KernelBasis:
    assert in_repr.group == out_repr.group
    
    # assert all(h >= 0. for h in heights)
    # TODO implement max_offset as filter to apply on top later

    group = in_repr.group
    
    assert isinstance(group, DihedralGroup)

    sg_id = (True, group.rotation_order)
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_full_cylinder_act_R3(in_repr: Representation, out_repr: Representation,
                                 radii: List[float],
                                 sigma: Union[List[float], float],
                                 maximum_frequency: int = 5,
                                 adjoint: np.ndarray = None,
                                 filter: Callable[[Dict], bool] = None
                                ) -> KernelBasis:
    sg_id = True, True, -1
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_full_cylinder_discrete_act_R3(in_repr: Representation, out_repr: Representation,
                                          radii: List[float],
                                          sigma: Union[List[float], float],
                                          # heights: List[float],
                                          maximum_frequency: int = 5,
                                          max_offset: int = None,
                                          adjoint: np.ndarray = None,
                                          filter: Callable[[Dict], bool] = None
                                          ) -> KernelBasis:
    group = in_repr.group
    
    # TODO implement max_offset as filter to apply on top later
    
    sg_id = True, True, group.order()//4
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_cylinder_act_R3(in_repr: Representation, out_repr: Representation,
                            radii: List[float],
                            sigma: Union[List[float], float],
                            maximum_frequency: int = 5,
                            adjoint: np.ndarray = None,
                            filter: Callable[[Dict], bool] = None
                           ) -> KernelBasis:
    sg_id = True, False, -1
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_cylinder_discrete_act_R3(in_repr: Representation, out_repr: Representation,
                                     radii: List[float],
                                     sigma: Union[List[float], float],
                                     # heights: List[float],
                                     maximum_frequency: int = 5,
                                     max_offset: int = None,
                                     adjoint: np.ndarray = None,
                                     filter: Callable[[Dict], bool] = None
                                    ) -> KernelBasis:
    group = in_repr.group
    
    # TODO implement max_offset as filter to apply on top later
    
    sg_id = True, False, group.order() // 2
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint, filter=filter
    )


def kernels_Inv_act_R3(in_repr: Representation, out_repr: Representation,
                       radii: List[float],
                       sigma: Union[List[float], float],
                       maximum_frequency: int = 5,
                       filter: Callable[[Dict], bool] = None
                       ) -> KernelBasis:
    sg_id = (True, False, 1)
    return kernels_O3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency, filter=filter
    )


def kernels_Trivial_act_R3(in_repr: Representation, out_repr: Representation,
                           radii: List[float],
                           sigma: Union[List[float], float],
                           maximum_frequency: int = 5,
                           filter: Callable[[Dict], bool] = None
                           ) -> KernelBasis:
    sg_id = so3_group().subgroup_trivial_id
    return kernels_SO3_subgroup_act_R3(
        in_repr, out_repr, sg_id,
        radii=radii, sigma=sigma,
        maximum_frequency=maximum_frequency, filter=filter
    )

