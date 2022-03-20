
import os
__cache_path__ = os.path.join(os.path.dirname(__file__), '_cache')


from .utils import psi, chi, psichi

from .group import Group, GroupElement

from .representation import Representation
from .irrep import IrreducibleRepresentation
from .representation import directsum
from .representation import disentangle
from .representation import change_basis
from .representation import homomorphism_space

from .homspace import HomSpace

from .directproduct import DirectProductGroup, direct_product
from .doublegroup import DoubleGroup, double_group

from .groups.factory import *
from .groups.cyclicgroup import CyclicGroup
from .groups.dihedralgroup import DihedralGroup
from .groups.so2group import SO2
from .groups.o2group import O2
from .groups.so3group import SO3
from .groups.o3group import O3
from .groups.ico import Icosahedral
from .groups.octa import Octahedral


from . import _numerical
from . import _clebsh_gordan
from ._clebsh_gordan import InsufficientIrrepsException, UnderconstrainedCGSystem



__all__ = [
    "Group",
    "GroupElement",
    #
    'DirectProductGroup',
    'direct_product',
    #
    'DoubleGroup',
    'double_group',
    # Groups
    "CyclicGroup",
    "DihedralGroup",
    "SO2",
    "O2",
    "SO3",
    "O3",
    "Icosahedral",
    "Octahedral",
    ############
    "trivial_group",
    "cyclic_group",
    "dihedral_group",
    "so2_group",
    "o2_group",
    "so3_group",
    "o3_group",
    "ico_group",
    "octa_group",
    "full_ico_group",
    "full_octa_group",
    "full_cylinder_group",
    "cylinder_group",
    "full_cylinder_discrete_group",
    "cylinder_discrete_group",
    # Representations
    "Representation",
    "IrreducibleRepresentation",
    "directsum",
    "disentangle",
    "change_basis",
    "homomorphism_space",
    #########
    "HomSpace",
    # Exceptions raised by numerical methods
    "InsufficientIrrepsException",
    "UnderconstrainedCGSystem",
    # utils
    "psi",
    "chi",
    "psichi",
    #
    "__cache_path__",
]


groups_dict = {
    CyclicGroup.__name__: CyclicGroup,
    DihedralGroup.__name__: DihedralGroup,
    SO2.__name__: SO2,
    O2.__name__: O2,
    SO3.__name__: SO3,
    O3.__name__: O3,
    Icosahedral.__name__: Icosahedral,
    Octahedral.__name__: Octahedral,
    DirectProductGroup.__name__: DirectProductGroup,
    DoubleGroup.__name__: DoubleGroup,
}