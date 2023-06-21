
from escnn.group import direct_product

from .cyclicgroup import CyclicGroup
from .dihedralgroup import DihedralGroup
from .so2group import SO2
from .o2group import O2
from .so3group import SO3
from .o3group import O3
from .ico import Icosahedral
from .octa import Octahedral


__all__ = [
    "trivial_group",
    "cyclic_group",
    "dihedral_group",
    "so2_group",
    "o2_group",
    "so3_group",
    "o3_group",
    "klein4_group",
    "ico_group",
    "octa_group",
    "full_ico_group",
    "full_octa_group",
    "full_cylinder_group",
    "cylinder_group",
    "full_cylinder_discrete_group",
    "cylinder_discrete_group",
]


def trivial_group():
    r"""
    
    Builds the trivial group :math:`C_1` which contains only the identity element :math:`e`.
    
    You should use this factory function to build an instance of the trivial group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built,
    this unique instance is updated with the new representations and, therefore, all its references will see the new
    representations.
    
    Returns:
        the trivial group

    """
    return CyclicGroup._generator(1)


def cyclic_group(N: int):
    r"""

    Builds a cyclic group :math:`C_N`of order ``N``, i.e. the group of ``N`` discrete planar rotations.
    
    You should use this factory function to build an instance of :class:`escnn.group.CyclicGroup`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~escnn.group.Group.quotient_representation`), this unique instance is updated with
    the new representations and, therefore, all its references will see the new representations.

    Args:
        N (int): number of discrete rotations in the group

    Returns:
        the cyclic group of order ``N``

    """
    return CyclicGroup._generator(N)


def dihedral_group(N: int):
    r"""

    Builds a dihedral group :math:`D_{2N}`of order ``2N``, i.e. the group of ``N`` discrete planar rotations
    and reflections.
    
    You should use this factory function to build an instance of :class:`escnn.group.DihedralGroup`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~escnn.group.Group.quotient_representation`), this unique instance is updated with
    the new representations and, therefore, all its references will see the new representations.

    Args:
        N (int): number of discrete rotations in the group
        
    Returns:
        the dihedral group of order ``2N``

    """
    return DihedralGroup._generator(N)


def so2_group(maximum_frequency: int = 3):
    r"""

    Builds the group :math:`SO(2)`, i.e. the group of continuous planar rotations.
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`escnn.group.SO2.irrep` (see the method's documentation).
    
    You should use this factory function to build an instance of :class:`escnn.group.SO2`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`escnn.group.SO2.irrep`), this unique instance is updated with the new representations
    and, therefore, all its references will see the new representations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`SO(2)`

    """
    return SO2._generator(maximum_frequency)


def o2_group(maximum_frequency: int = 3):
    r"""

    Builds the group :math:`O(2)`, i.e. the group of continuous planar rotations and reflections.
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`escnn.group.O2.irrep` (see the method's documentation).
    
    You should use this factory function to build an instance of :class:`escnn.group.O2`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`escnn.group.O2.irrep`), this unique instance is updated with the new representations
    and, therefore, all its references will see the new representations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`O(2)`

    """
    return O2._generator(maximum_frequency)


def so3_group(maximum_frequency: int = 2):
    r"""

    Builds the group :math:`SO(3)`, i.e. the group of continuous 3D rotations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`SO(3)`

    """
    return SO3._generator(maximum_frequency)


def o3_group(maximum_frequency: int = 2):
    r"""

    Builds the group :math:`O(3)`, i.e. the group of continuous 3D rotations and reflections.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`O(3)`

    """
    return O3._generator(maximum_frequency)


def klein4_group():
    r"""

    Builds the group :math:`K_4 = C_2 \times C_2`. This group is most commonly associated with the symmetries of a
    rectangle. I.e. a group generated by two perpendicular reflection and a rotation of 180 degrees.
    The returned group instance has an additional 2D representation named ``"rectangle"``.

    Returns:
        the group :math:`K_4`
    """
    c2 = cyclic_group(2)
    K4 = direct_product(c2, c2, name='Klein4')
    # Klein4 group is mostly associated with the symmetries of the rectangle (2 perpendicular reflections (gx, gy) and a
    # 180 deg rotation (gx · gy) in 2D space. We add this representation, constructed from the irreps of the group.
    # rep2d_rectangle(e) = [[1, 0], [0, 1]], rep2d_rectangle(gx) = [[-1, 0], [0, 1]],
    # rep2d_rectangle(gy) = [[1, 0], [0, -1]] rep2d_rectangle(gx · gy) = [[-1, 0], [0, -1]]
    rep2d_rectangle = K4.irrep((1,), (0,)) + K4.irrep((0,), (1,))
    K4.representations.update(rectangle=rep2d_rectangle)
    return K4


def ico_group():
    r"""

    Builds the group :math:`I`, i.e. the group of symmetries of the icosahedron

    Returns:
        the group :math:`I`

    """
    return Icosahedral._generator()


def full_ico_group():
    r"""

    Builds the group :math:`I_h = C_2 \times I`, i.e. the group of all symmetries of the icosahedron

    Returns:
        the group :math:`I`

    """
    ico = ico_group()
    c2 = cyclic_group(2)
    return direct_product(c2, ico, name='FullIcosahedral')


def octa_group():
    r"""

    Builds the group :math:`O`, i.e. the group of symmetries of the cube or octahedron

    Returns:
        the group :math:`O`

    """
    return Octahedral._generator()


def full_octa_group():
    r"""

    Builds the group :math:`O_h = C_2 \times O`, i.e. the group of all symmetries of the cube or octahedron

    Returns:
        the group :math:`O_h`

    """
    octa = octa_group()
    c2 = cyclic_group(2)
    return direct_product(c2, octa, name='FullOctahedral')


def full_cylinder_group(maximum_frequency: int = 3):
    r"""

    Builds the group :math:`C_2 \times O(2)`, i.e. the group of symmetries of a cylinder.
    It contains continuous planar rotations and planar reflections (the :math:`O(2)` dihedral symmetry subgroup)
    and 3D inversions (the :math:`C_2` subgroups).
    The group also includes the subgroup of mirroring with respect to a plane passing throgh the axis of the cylinder.
    
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`~escnn.group.DirectProductGroup.irrep` (see the method's documentation).

    You should use this factory function to build an instance of of this group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~escnn.group.O2.irrep` of an instance of :class:`~escnn.group.O2`), this unique
    instance is updated with the new representations and, therefore, all its references will see the new representations.

    Args:
        maximum_frequency (int): maximum frequency of the :math:`O(2)` irreps

    Returns:
        the group :math:`C_2 \times O(2)`

    """
    o2 = o2_group(maximum_frequency)
    c2 = cyclic_group(2)
    
    return direct_product(c2, o2, name='FullCylindrical')


def cylinder_group(maximum_frequency: int = 3):
    r"""

    Builds the group :math:`C_2 \times SO(2)`, which contains continuous planar rotations (the :math:`SO(2)`
    symmetry subgroup) and 3D inversions (the :math:`C_2` subgroups).

    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`~escnn.group.DirectProductGroup.irrep` (see the method's documentation).

    You should use this factory function to build an instance of of this group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~escnn.group.SO2.irrep` of an instance of :class:`~escnn.group.SO2`), this unique
    instance is updated with the new representations and, therefore, all its references will see the new representations.

    Args:
        maximum_frequency (int): maximum frequency of the :math:`SO(2)` irreps

    Returns:
        the group :math:`C_2 \times O(2)`

    """
    so2 = so2_group(maximum_frequency)
    c2 = cyclic_group(2)
    
    return direct_product(c2, so2, name='Cylindrical')


def full_cylinder_discrete_group(n: int):
    r"""

    Builds the group :math:`C_2 \times D_n`, i.e. a group of discrete symmetries of a cylinder.
    It contains `n` discrete planar rotations and planar reflections (the :math:`D_n` dihedral symmetry subgroup)
    and 3D inversions (the :math:`C_2` subgroups).
    The group also includes the subgroup of mirroring with respect to `n` planes passing through the axis of the
    cylinder.

    You should use this factory function to build an instance of of this group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.

    Args:
        n (int): number of discrete rotations

    Returns:
        the group :math:`C_2 \times D_n`

    """
    dn = dihedral_group(n)
    c2 = cyclic_group(2)
    
    return direct_product(c2, dn, name='FullCylindricalDiscrete')


def cylinder_discrete_group(n: int):
    r"""

    Builds the group :math:`C_2 \times C_n`, which contains `n` discrete planar rotations (the :math:`C_n`
    symmetry subgroup) and 3D inversions (the :math:`C_2` subgroups).

    You should use this factory function to build an instance of of this group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.

    Args:
        n (int): number of discrete rotations

    Returns:
        the group :math:`C_2 \times C_n`

    """
    if n == 2:
        return klein4_group()

    cn = cyclic_group(n)
    c2 = cyclic_group(2)
    
    return direct_product(c2, cn, name='CylindricalDiscrete')

