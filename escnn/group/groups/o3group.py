from __future__ import annotations

import escnn.group
from escnn.group import Group, GroupElement

from .so3_utils import *
from .so3_utils import IDENTITY as IDENTITY_SO3
from .so3group import _clebsh_gordan_tensor_so3

from escnn.group import IrreducibleRepresentation
from escnn.group import Representation
from escnn.group import directsum
from escnn.group import change_basis
from escnn.group.irrep import restrict_irrep

from .utils import *

import numpy as np

from typing import Tuple, Callable, Iterable, List, Dict, Any, Union


__all__ = ["O3"]


class O3(Group):
    
    PARAM = PARAMETRIZATION

    PARAMETRIZATIONS = PARAMETRIZATIONS

    def __init__(self, maximum_frequency: int = 3):
        r"""
        Build an instance of the orthogonal group :math:`O(3)` which contains reflections and continuous 3D rotations.
        
        Subgroup structure:
        
        
        +-----------------------------------+-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |    ``id[0]``                      |    ``id[1]``                      |    ``id[2]``                      |    subgroup                                                                                                                                                                                                   |
        +===================================+===================================+===================================+===============================================================================================================================================================================================================+
        |    `False`                        |        'so3'                      |                                   |   :math:`SO(3)` subgroup (equivalent to just `id = "so3"`)                                                                                                                                                    |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'ico'                      |                                   |   Icosahedral :math:`I` subgroup                                                                                                                                                                              |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'octa'                     |                                   |   Octahedral :math:`O` subgroup                                                                                                                                                                               |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'tetra'                    |                                   |   Tetrahedral :math:`T` subgroup                                                                                                                                                                              |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        False                      |    -1                             |   :math:`SO(2)` subgroup of planar rotations around Z axis                                                                                                                                                    |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        False                      |     N                             |   :math:`C_N` of N discrete planar rotations around Z axis                                                                                                                                                    |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |    -1                             |   *dihedral* :math:`O(2)` subgroup of planar rotations around Z axis and out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1]/2`` (X axis by default)                             |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |     N>1                           |   *dihedral* :math:`D_N` subgroup of N discrete planar rotations around Z axis and out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1]/2`` (X axis by default)                   |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |     1                             |   equivalent to ``(False, False, 2, adj)``                                                                                                                                                                    |
        +-----------------------------------+-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |    'fulltetra'                    |                                   |                                   |   :math:`T_d \cong O` full isometry group tetrahedron                                                                                                                                                         |
        +-----------------------------------+-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |    `True`                         |        'so3'                      |                                   |   :math:`O(3)` itself (equivalent to just `id = "o3"`)                                                                                                                                                        |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'ico'                      |                                   |   Icosahedral :math:`I_h \cong I \times C_2` subgroup                                                                                                                                                         |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'octa'                     |                                   |   Octahedral :math:`O_h \cong O \times C_2` subgroup                                                                                                                                                          |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        'tetra'                    |                                   |   Tetrahedral :math:`T_h \cong T \times C_2` subgroup                                                                                                                                                         |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        False                      |    -1                             |   :math:`SO(2) \times C_2` subgroup of planar rotations around Z axis and inversions                                                                                                                          |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        False                      |     N>1                           |   :math:`C_N \times C_2` of N discrete planar rotations around Z axis and inversions                                                                                                                          |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        False                      |     1                             |   :math:`C_2` subgroup inversions                                                                                                                                                                             |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |    -1                             |   :math:`O(2) \times C_2` subgroup of planar rotations around Z axis, out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1]/2`` (X axis by default) and inversions                 |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |     N>1                           |   :math:`D_N \times C_2` subgroup of N discrete planar rotations around Z axis, out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1]/2`` (X axis by default) and inversions       |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |        True (or float)            |     1                             |   equivalent to ``(True, False, 2, adj)``                                                                                                                                                                     |
        +-----------------------------------+-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |   'cone'                          |        -1                         |                                   |   *conic* subgroup :math:`O(2)`                                                                                                                                                                               |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |         N>1                       |                                   |   *conic* subgroup of N discrete rotations :math:`D_N`                                                                                                                                                        |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |         1                         |                                   |   subgroup :math:`C_2` of mirroring with respect to a plane passing through the X axis                                                                                                                        |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |         float                     |    -1                             |   *conic* subgroup :math:`O(2)` with reflection along the ``id[1]/2`` axis in the XY plane. Equivalent to ('cone', -1, adj)                                                                                   |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |         float                     |     N>1                           |   *conic* subgroup of N discrete rotations :math:`D_N` along the ``id[1]/2`` axis in the XY plane. Equivalent to ('cone', N, adj)                                                                             |
        +                                   +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |                                   |         float                     |     1                             |   subgroup :math:`C_2` of mirroring with respect to the plane passing through the ``id[1]/2`` axis in the XY plane. Equivalent to ('cone', 1, adj)                                                            |
        +-----------------------------------+-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        
        .. warning ::
            For some subgroups, ```id[1]``` expects either a boolean or a floating point value.
            In this case, ```id[1] = 0``` might be mistakenly interpreted as ```False``` rather than ```0.0```, so one
            should pay attention to use ```0.``` rather than ```0```.
        
        
        Args:
            maximum_frequency (int, optional): the maximum frequency to consider when building the irreps of the group
        
        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(O3, self).__init__("O(3)", True, False)

        self._maximum_frequency = maximum_frequency
        
        self._identity = self.element((0, IDENTITY_SO3))
        self._inversion = self.element((1, IDENTITY_SO3))
        
        self._build_representations()

    @property
    def generators(self) -> List[GroupElement]:
        raise ValueError(f'{self.name} is a continuous groups and '
                         f'some of its generators are infinitesimal. '
                         f'This is not currently supported')

    @property
    def identity(self) -> GroupElement:
        return self._identity

    @property
    def inversion(self) -> GroupElement:
        r"""
        The inversion element of :math:`\O3`.
        
        """
        return self._inversion

    @property
    def elements(self) -> List[GroupElement]:
        return None
     
    # @property
    # def elements_names(self) -> List[str]:
    #     return None

    @property
    def _keys(self) -> Dict[str, Any]:
        return dict()

    @property
    def subgroup_trivial_id(self):
        return (False, False, 1)

    @property
    def subgroup_self_id(self):
        return 'o3'

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################

    def _inverse(self, element: Tuple, param: str = PARAM) -> Tuple:
        r"""
        Return the inverse element of the input element.
        
        Args:
            element (tuple): a group element

        Returns:
            its inverse
            
        """
        
        return element[0], _invert(element[1], param)
        
    def _combine(self, e1: Tuple, e2: Tuple,
                param: str = PARAM,
                param1: str = None,
                param2: str = None
        ) -> Tuple:
        r"""
        Return the combination of the two input elements.
        Args:
            e1 (tuple): a group element
            e2 (tuple): another element

        Returns:
            their combination
        """
        return (e1[0] + e2[0]) % 2, _combine(e1[1], e2[1], param=param, param1=param1, param2=param2)

    def _equal(self, e1: Tuple, e2: Tuple,
              param: str = PARAM,
              param1: str = None,
              param2: str = None
        ) -> bool:
        r"""

        Check if the two input values corresponds to the same element.

        See :meth:`escnn.group.SO3.equal` for more details.

        Args:
            e1 (tuple): an element
            e2 (tuple): another element
            
        Returns:
            whether they are the same element

        """
        return (e1[0] == e2[0]) and _equal(e1[1], e2[1], param=param, param1=param1, param2=param2)

    def _hash_element(self, element: Tuple, param: str = PARAM):
        rot_hash = _hash(element[1], param)
        return hash(element[0]*100) + rot_hash

    def _repr_element(self, element: Tuple, param: str = PARAM):
    
        repr = _repr(element[1], param)
        repr = repr.split('\n')

        s = '+' if not element[0] else '-'
        repr[0] =    '     ' + repr[0]
        repr[1] = f'({s}), ' + repr[1]
        repr[2] =    '     ' + repr[2]
        return '\n'.join(repr)

    def _is_element(self, element: Tuple, param: str = PARAM, verbose: bool = False) -> bool:
        
        if not isinstance(element, Tuple) or len(element) != 2:
            if verbose:
                print(f'Expected a tuple of 2 elements, but {element} found')
            return False
        
        if element[0] not in {0, 1}:
            if verbose:
                print(f'The first value of an O(3) element needs to be an integer in {{0, 1}}, but {element[0]} found ')
            return False

        if not _check_param(element[1], param):
            if verbose:
                print(f'The second value of an O(3) element needs to be a rotation, but {element[1]} found ')
            return False
    
        return True

    def _change_param(self, element: Tuple, p_from, p_to) -> Tuple:
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS
        
        if p_from == p_to:
            return element
        
        return element[0], _change_param(element[1], p_from, p_to)

    ###########################################################################
    
    def sample(self, param: str = PARAMETRIZATION) -> GroupElement:
        return self.element((np.random.randint(0, 2), _grid('rand', N=1, parametrization=param)[0]))

    def grid(self, type: str, *args, adj=IDENTITY, parametrization: str = PARAMETRIZATION, **kwargs) -> List[GroupElement]:
        r"""
        Method which builds different collections of elements of :math:`\O3`.

        Depending on the value of ``type``, the method accept a different set of parameters using the ``args`` and the
        ``kwargs`` arguments:

        * ``type = "rand"``. Generate ``N`` *random* uniformly distributed samples over the group. The method accepts an integer ``N`` and, optionally, a random seed ``seed`` (or an instance of :class:`numpy.random.RandomState`).
        
        Other options generate grids over :math:`\SO3` (see :meth:`escnn.group.SO3.grid`) and **then repeat it twice**,
        by combining each :math:`\SO3` element with both the identity and the *inversion* elements of :math:`\O3`.

        * ``type = "thomson"``.  Generate ``N`` samples distributed (approximately) uniformly over :math:`\SO3`

        * ``type = "ico", "cube", "tetra"``.  Generate respectively ``60``, ``24`` or ``12`` samples over :math:`\SO3`.

        * ``type = "hopf"``.  Generates *about* ``N`` points on :math:`\SO3`.

        * ``type = "fibonacci"``.  Generates *about* ``N`` points  on :math:`\SO3`.
        
        .. seealso ::
            :meth:`escnn.group.SO3.grid`

        Args:
            type (str): string identifying the type of samples
            *args: arguments specific for the type of samples chosen
            adj (GroupElement, optional): optionally, apply an adjoint transform to the sampled elements.
            parametrization (str, optional):
            **kwargs: arguments specific for the type of samples chosen

        Returns:
            a list of group elements

        """
        if type == 'rand':
            return [
                self.element(g, param=PARAMETRIZATION)
                for g in _random_samples(*args, **kwargs, parametrization=PARAMETRIZATION)
            ]
        else:
            so3_grid = _grid(type, *args, adj=adj, parametrization=parametrization, **kwargs)
            return [
                    self.element((0, g), param=parametrization) for g in so3_grid
            ] + [
                    self.element((1, g), param=parametrization) for g in so3_grid
            ]

    def grid_so3(self, type: str, *args, adj=IDENTITY, parametrization: str = PARAMETRIZATION, **kwargs) -> List[GroupElement]:
        r"""
        Method which builds different collections of elements of the :math:`\SO3` subgroup of :math:`\O3`.
        This method is equivalent to :meth:`escnn.group.SO3.grid`, but the :math:`\SO3` elements will be embedded in
        :math:`\O3`.

        Args:
            type (str): string identifying the type of samples
            *args: arguments specific for the type of samples chosen
            adj (GroupElement, optional): optionally, apply an adjoint transform to the sampled elements.
            parametrization (str, optional):
            **kwargs: arguments specific for the type of samples chosen

        Returns:
            a list of group elements

        """
        so3_grid = _grid(type, *args, adj=adj, parametrization=parametrization, **kwargs)
        return [self.element((0, g), param=parametrization) for g in so3_grid]

    def sphere_grid(self, type: str, *args, adj: GroupElement = None, **kwargs) -> List[GroupElement]:
        r"""

        Method which builds different collections of points over the sphere.

        Here, a sphere is interpreted as the quotient space :math:`\O3 / \O2` (where :math:`\O2` is the "cone" subgroup
        identified by the id ```('cone', -1)```).
        The list returned by this method contains instances of :class:`~escnn.group.GroupElement`.
        These are elements of :class:`~escnn.group.O3` and should be interpreted as *representatives* of cosets in
        :math:`\O3 / \O2`.

        Depending on the value of ``type``, the method accept a different set of parameters using the ``args`` and the
        ``kwargs`` arguments:

        * ``type = "rand"``. Generate ``N`` *random* uniformly distributed samples over the sphere. The method accepts an integer ``N`` and, optionally, a random seed ``seed`` (or an instance of :class:`numpy.random.RandomState`).

        * ``type = "thomson"``.  Generate ``N`` samples distributed (approximately) uniformly over the sphere. The samples are obtained with a numerical method which tries to minimize a potential energy depending on the relative distance between the points. The first call of this method for a particular value of ``N`` might be slow due to this numerical method. However, the resulting set of points is cached on disk such that following calls will return the same result instantly.

        * ``type = "thomson_cube"``.  Generate ``24*N`` samples with perfect *cubic* (*octahedral*) symmetry, distributed (approximately) uniformly over the shere. The samples are obtained with a numerical method which tries to minimize a potential energy depending on the relative distance between the points. The first call of this method for a particular value of ``N`` might be slow due to this numerical method. However, the resulting set of points is cached on disk such that following calls will return the same result instantly.

        * ``type = "ico", "cube", "tetra"``.  Return the orbit of the point :math:`(0, 0, 1)` under the action of the Icosahedral, Octahedral or Tetrahedral group. The resulting samples correspond to the corners, edges or faces of a Platonic solid.

        .. todo ::
            add more types to support vertices, edges and faces for all solids.
            Currently, ``"ico"`` returns the centers of the 30 edges while ``"cube"`` returns the centers of its 6 faces

        .. warning::
            The naming convention might change in future to support faces/edges/vertices of the solids

        .. note ::
            ``type = "tetra"`` does not have inversion symmetry but mirror symmetries (see Symmetries of the Tetrahedron)

        * ``type = "healpix"``.  Generates *about* ``N`` points using the HEAPlix grid on the sphere.

        * ``type = "fibonacci"``.  Generates ``N`` points using the Fibonacci grid on the sphere.

        * ``type = "longlat"``.  Generates an ``N`` by ``M`` grid over the sphere in the longitude (``N``) - latitude (``M``) coordinates.


        Args:
            type (str): string identifying the type of samples
            *args: arguments specific for the type of samples chosen
            adj (GroupElement, optional): optionally, apply an adjoint transform to the sampled elements.
            **kwargs: arguments specific for the type of samples chosen

        Returns:
            a list of group elements

        """

        if adj is None:
            adj = self.identity
        adj = self.standard_representation()(adj)

        points = _sphere_grid(type, *args, adj=adj, **kwargs)

        S = points.shape[0]
        x, y, z = points.T

        theta = np.arccos(np.clip(z, -1., 1.))
        phi = np.arctan2(y, x)

        return [self.element((0, (0, theta[s], phi[s])), 'zyz') for s in range(S)]

    def testing_elements(self, n=3) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        return self.grid('ico')

        so3_samples = np.empty((n**3, 3))
        i = 0
        for a in range(n):
            alpha = 2 * np.pi * a / n
            for b in range(n):
                # beta = np.pi * (2 * b + 1) / (2*n)
                # beta = np.pi * (b + 1) / (n+2)
                # sample around b = 0  and b = np.pi to check singularities
                beta = np.pi * b / n
                for c in range(n):
                    gamma = 2 * np.pi * c / n
                    so3_samples[i, :] = alpha, beta, gamma
                    i+=1

        so3_samples = _change_param(so3_samples, p_from='ZYZ', p_to=self.PARAM)

        samples = [self.element((0, g), self.PARAM) for g in so3_samples]
        samples += [self.element((1, g), self.PARAM) for g in so3_samples]

        return samples

    def __eq__(self, other):
        if not isinstance(other, O3):
            return False
        else:
            return self.name == other.name # and self._maximum_frequency == other._maximum_frequency

    def _process_subgroup_id(self, id):
        
        # (True, None)   -> Inversion C_2 subgroup
        # (False, x)     -> SO(3) subgroup (x)
        # (True, x)      -> SO(3) subgroup (x)  X Inversion
        #   includes also Cylinder as O(2) X Inversion
        # ('fulltetra')  -> full tethraedral is a semidirect prod however; it is however isomorphic to Octa group!!!
        # ('cone', N)    -> conical subgroup, O(2) or D_N depending on N

        if not isinstance(id, tuple):
            id = (id,)

        if id[0] == 'so3':
            id = False, 'so3', *id[1:]

        if id[0] == 'o3':
            id = True, 'so3', *id[1:]
            
        if not isinstance(id[-1], GroupElement):
            id = (*id, self.identity)
    
        assert id[-1].group == self
        
        if len(id) == 4 and isinstance(id[0], bool):
            assert isinstance(id[1], float) or isinstance(id[1], bool)

        if len(id) == 4 and (
                isinstance(id[0], bool) and isinstance(id[1], float)
        ):
            # O(2) (or subgroup) subgroup of SO(3)
            # if id[1]  is bool, it specifies SO(2) vs O(2) subgroup
            # if id[1] is float, it assumes O(2) and indicates the flip axis of O(2)
            # in this last case, we convert it to the boolean convention and include the flip axis inside the adjoing

            # The factor of 2 in id[1] comes from the fact that a flip around the axis theta, as an element of O(2), is
            # the combination of a reflection along the X axis (theta=0) and a rotation by 2*theta.
            # The values id[1] = 2*theta represents the SO(2) component of the flip.

            flip_axis = id[1] / 2.
            flip = np.asarray([0., 0., np.sin(flip_axis / 2), np.cos(flip_axis / 2)])
            flip = self.element((0, flip), 'Q')
            adjoint = (~flip) @ id[-1]
            id = (id[0], True, id[2], adjoint)

        if len(id) == 4 and (
                isinstance(id[0], bool) and id[1] and id[2] == 1
        ):
            # flip subgroup of the O(2) subgroup of SO(3) subgroup
            # this is equivalent to the C_2 subgroup of 180 deg rotations out of the plane
            change_axis = np.asarray([0., np.sin(-np.pi / 4), 0., np.cos(-np.pi / 4)])
            adj = self.element((0, change_axis), 'Q') @ id[3]
            id = (id[0], False, 2, adj)

        if id[0] not in ['fulltetra', 'cone']:
            assert isinstance(id[1], bool) or isinstance(id[1], str)
        
        if len(id) == 4 and id[:3] == (True, False, 1):
            # if it is the C_2 subgroup of the inversion, it is invariant to adjunction
            id = (True, False, 1, self.identity)

        if id[0] == 'cone' and len(id) == 4 and isinstance(id[1], float) and isinstance(id[2], int):
            # A cone subgroup is generally specified by ('cone', N, adj)
            # It is assumed that the flip is always along the X axis in the plane defined by adj
            # If one wants to define the flip along another axis in the plane, one can tune the adj parameter.
            # However, this can be inconvenient ofter.
            # In this case, one can also use the tuple ('cone, theta, N, adj)
            # where theta specifies 2 times the flip axis with respect to the X axis.
            # This is converted here into a tuple ('cone', N, adj) by computing the right adj value
            
            
            # The factor of 2 in id[1] comes from the fact that a flip around the axis theta, as an element of O(2), is
            # the combination of a reflection along the X axis (theta=0) and a rotation by 2*theta.
            # The values id[1] = 2*theta represents the SO(2) component of the flip.
            
            flip_axis = id[1] / 2.
            flip = np.asarray([0., 0., np.sin(flip_axis / 2), np.cos(flip_axis / 2)])
            adj = id[-1]
            adj = self.element((0, flip), 'Q') @ adj
            id = ('cone', id[2], adj)

        return id

    def _subgroup(self, id: Tuple) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        Restrict the current group :math:`O(3)` to the subgroup identified by the input ``id``

        
        Args:
            id (tuple): the identification of the subgroup

        Returns:
            a tuple containing
            
                - the subgroup
                
                - a function which maps an element of the subgroup to its inclusion in the original group and
                
                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
                
        """
        
        # ('so3')        -> SO(3)
        # (False, x)     -> SO(3)'s subgroup
        # (True, x)      -> SO(3)'s subgroup X Inversion
        #   includes also Cylinder as O(2) X Inversion
        # (True, None, 1) = (true, None)   -> Inversion C_2 subgroup
        # ('fulltetra')  -> full tethraedral is a semidirect prod however; it is however isomorphic to Octa group!!!
        # ('cone', x)    -> conical subgroup, O(2) or D_N depending on x

        sg_id, adj = id[:-1], id[-1]
        assert isinstance(adj, GroupElement)
        assert adj.group == self

        if sg_id == (False, 'so3'):
            # SO3 group
            sg = escnn.group.so3_group(self._maximum_frequency)
            parent_mapping = so3_to_o3(adj, sg)
            child_mapping = o3_to_so3(adj, sg)
            
        elif sg_id == (True, 'so3'):
            # the O3 group itself
            sg = self
            parent_mapping = build_adjoint_map(self, ~adj) #lambda x, adj=adj, _adj=~adj: _adj @ x @ adj
            child_mapping = build_adjoint_map(self, adj) #lambda x, adj=adj, _adj=~adj: adj @ x @ _adj

        elif sg_id[0] == 'fulltetra' and len(sg_id) == 1:
            # T_d group
            # the group is isomerphic to the Octahedral group
            # sg = Octahedral()
            raise NotImplementedError()

        elif sg_id[0] == True:
            # the rest of the id identifies a subgroup of the SO3 subgroup
            so3_sg_id = sg_id[1:]
            
            if len(so3_sg_id) == 1 and isinstance(so3_sg_id[0], str):
                if so3_sg_id[0] == 'ico':
                    # I_h
                    sg = escnn.group.full_ico_group()
                    parent_mapping = fullplato_to_o3(adj, sg)
                    child_mapping = o3_to_fullplato(adj, sg)
                elif so3_sg_id[0] == 'octa':
                    # O_h
                    sg = escnn.group.full_octa_group()
                    parent_mapping = fullplato_to_o3(adj, sg)
                    child_mapping = o3_to_fullplato(adj, sg)
                elif so3_sg_id[0] == 'tetra':
                    # T_h
                    # Pyritohedral symmetry
                    # TODO implement group
                    # sg = escnn.group.full_pyrito_group()
                    # parent_mapping = fullplato_to_o3(adj, sg)
                    # child_mapping = o3_to_fullplato(adj, sg)
                    raise NotImplementedError()
                else:
                    raise ValueError(f'Subgroup "{so3_sg_id}" not recognized!')
                
            elif len(so3_sg_id) == 2:
                # planar subgroups
                assert isinstance(so3_sg_id[1], int) or so3_sg_id[1] is None
        
                if so3_sg_id == (False, -1):
                    # SO(2) x C_2
                    sg = escnn.group.cylinder_group(self._maximum_frequency)
                    parent_mapping = cyl_to_o3(adj, sg)
                    child_mapping = o3_to_cyl(adj, sg)
                elif so3_sg_id == (True, -1):
                    # Cylinder,  O(2) x C_2
                    sg = escnn.group.full_cylinder_group(self._maximum_frequency)
                    parent_mapping = fullcyl_to_o3(adj, sg)
                    child_mapping = o3_to_fullcyl(adj, sg)
                elif so3_sg_id[0] is False and so3_sg_id[1] > 0:
                    # C_N x C_2
                    
                    N = so3_sg_id[1]
                    
                    if N == 1:
                        # Inversion subgroup C_2 of O(3)
                        sg = escnn.group.cyclic_group(2)
                        parent_mapping = inv_to_o3(self, sg)
                        child_mapping = o3_to_inv(self, sg)
                    else:
                        # C_N x C_2
                        sg = escnn.group.cylinder_discrete_group(N)
                        parent_mapping = cyl_to_o3(adj, sg)
                        child_mapping = o3_to_cyl(adj, sg)
                elif so3_sg_id[0] is True and so3_sg_id[1] > 1:
                    # Cylinder, D_N x C_2
                    N = so3_sg_id[1]
                    sg = escnn.group.full_cylinder_discrete_group(N)
                    parent_mapping = fullcyl_to_o3(adj, sg)
                    child_mapping = o3_to_fullcyl(adj, sg)
                elif so3_sg_id[0] is True and so3_sg_id[1] == 1:
                    # flip subgroup of the O(2) subgroup, C_2 x C_2
                    # This case should have already been caught in the CyclicGroup case (thanks to the
                    # process_subgroup_id method)
                    # this is equivalent to C_2 x C_2 above
                    # TODO catch this case in process_subgroup_id
                    raise ValueError(f'Subgroup "{sg_id}": this case should have already been caught!')
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')
            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')

        elif sg_id[0] == False:
            
            # the rest of the id identifies a subgroup of the SO3 subgroup
            so3_sg_id = sg_id[1:]
    
            if len(so3_sg_id) == 1 and isinstance(so3_sg_id[0], str):
                if so3_sg_id[0] == 'ico':
                    sg = escnn.group.ico_group()
                    parent_mapping = so3_to_o3(adj, sg)
                    child_mapping = o3_to_so3(adj, sg)
                elif so3_sg_id[0] == 'octa':
                    sg = escnn.group.octa_group()
                    parent_mapping = so3_to_o3(adj, sg)
                    child_mapping = o3_to_so3(adj, sg)
                elif so3_sg_id[0] == 'tetra':
                    # sg = escnn.group.tetra_group()
                    # parent_mapping = so3_to_o3(adj, sg)
                    # child_mapping = o3_to_so3(adj, sg)
                    raise NotImplementedError()
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')

            elif len(so3_sg_id) == 2:
                # planar subgroups
        
                if so3_sg_id == (False, -1):
                    # SO(2)
                    sg = escnn.group.so2_group(self._maximum_frequency)
                    parent_mapping = so2_to_o3(adj, sg)
                    child_mapping = o3_to_so2(adj, sg)
                elif so3_sg_id == (True, -1):
                    # Dihedral,  O(2)
                    sg = escnn.group.o2_group(self._maximum_frequency)
                    parent_mapping = dih_to_o3(adj, sg)
                    child_mapping = o3_to_dih(adj, sg)
                elif so3_sg_id[0] == False and so3_sg_id[1] > 0:
                    # Cyclic subgroup
                    # or flip subgroup of the O(2) subgroup
                    n = so3_sg_id[1]
                    assert isinstance(n, int)
                    sg = escnn.group.cyclic_group(n)
                    parent_mapping = so2_to_o3(adj, sg)
                    child_mapping = o3_to_so2(adj, sg)
        
                elif so3_sg_id[0] == True and so3_sg_id[1] > 1:
                    # Dihedral, dihedral subgroup
                    n = so3_sg_id[1]
                    assert isinstance(n, int)
                    sg = escnn.group.dihedral_group(n)
            
                    parent_mapping = dih_to_o3(adj, sg)
                    child_mapping = o3_to_dih(adj, sg)
        
                elif so3_sg_id[0] is True and so3_sg_id[1] == 1:
                    # flip subgroup of the O(2) subgroup
                    # This case should have already been caught in the CyclicGroup case (thanks to the
                    # process_subgroup_id method)
                    raise ValueError(f'Subgroup "{sg_id}": this case should have already been caught!')
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')
            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')

        elif sg_id[0] == 'cone':
            assert len(sg_id) == 2, sg_id
            
            N = sg_id[1]

            assert isinstance(N, int), N
            
            if N == -1:
                # O(2) cone symmetry
                sg = escnn.group.o2_group(self._maximum_frequency)
                parent_mapping = con_to_o3(adj, sg)
                child_mapping = o3_to_con(adj, sg)
            elif N > 1:
                # D_N pyramid symmetry
                sg = escnn.group.dihedral_group(N)
                parent_mapping = con_to_o3(adj, sg)
                child_mapping = o3_to_con(adj, sg)
            elif N == 1:
                # C_2 mirroring wrt a plane
                sg = escnn.group.cyclic_group(2)
                parent_mapping = mir_to_o3(adj, sg)
                child_mapping = o3_to_mir(adj, sg)

            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')

        else:
            raise ValueError(f'Subgroup "{sg_id}" not recognized!')

        return sg, parent_mapping, child_mapping

    def _combine_subgroups(self, sg_id1, sg_id2):
        
        sg_id1 = self._process_subgroup_id(sg_id1)
        sg1, inclusion, restriction = self.subgroup(sg_id1)
        sg_id2 = sg1._process_subgroup_id(sg_id2)
    
        sg_id1, adjoint1 = sg_id1[:-1], sg_id1[-1]
    
        adjoint2 = None
        sg_id = None
    
        if sg_id1 == (True, 'so3'):
            # subgroup of O3
            sg_id = sg_id2[:-1]
            adjoint2 = sg_id2[-1]
    
        elif sg_id1 == (False, 'so3'):
            # subgroup of SO3
            sg_id = (False,) + sg_id2[:-1]
            adjoint2 = sg_id2[-1]

        elif sg_id1[0] == 'fulltetra':
            # subgroup  of T_d group
            raise NotImplementedError()
    
        elif sg_id1[0] == True:
            # the rest of the id identifies a subgroup of the SO3 subgroup
            so3_sg_id = sg_id1[1:]
            
            # TODO :  these subgroups have the cone subgroups as subgroups
            # but since we model them as direct products, we do not implement this further subgroups operation
            # we should have a new class for cylinder symmetry whihc inheriths from DirectproductGroup and
            # which additionally implements this subgroup

            if so3_sg_id[0] in ['ico', 'octa', 'tetra']:
                # I_h
                # O_h
                # T_h:  Pyritohedral symmetry
                raise NotImplementedError()
        
            elif len(so3_sg_id) == 2:
                # planar subgroups
                assert isinstance(so3_sg_id[1], int) or so3_sg_id[1] is None
            
                if so3_sg_id == (False, -1):
                    # C_2 x SO(2)
                    sg_id = sg_id2[0], False, sg_id2[1]
            
                elif so3_sg_id == (True, -1):
                    # Cylinder,  C_2 x O(2)
                    sg_id = sg_id2[0], sg_id2[1][0], sg_id2[1][1]
            
                elif so3_sg_id[0] is False and so3_sg_id[1] > 0:
                    N = so3_sg_id[1]
                
                    if N == 1:
                        # Inversion subgroup C_2 of O(3)
                        sg_id = sg_id2, False, 1
                    else:
                        # C_2 x C_N
                        sg_id = sg_id2[0], False, sg_id2[1]
            
                elif so3_sg_id[0] is True and so3_sg_id[1] > 1:
                    # Cylinder, C_2 x D_N
                    if sg_id2[1][0] is not None:
                        sg_id = sg_id2[0], sg_id2[1][0] * 2 * np.pi / so3_sg_id[1], sg_id2[1][1]
                    else:
                        sg_id = sg_id2[0], False, sg_id2[1][1]
            
                elif so3_sg_id[0] is True and so3_sg_id[1] == 1:
                    # flip subgroup of the O(2) subgroup, C_2 x C_2
                    # This case should have already been caught in the CyclicGroup case (thanks to the
                    # process_subgroup_id method)
                    # this is equivalent to C_2 x C_2 above
                    raise ValueError(f'Subgroup "{sg_id1}": this case should have already been caught!')
                else:
                    raise ValueError(f'Subgroup "{sg_id1}" not recognized!')
            else:
                raise ValueError(f'Subgroup "{sg_id1}" not recognized!')
    
        elif sg_id1[0] == False:
        
            # the rest of the id identifies a subgroup of the SO3 subgroup
            so3_sg_id = sg1._combine_subgroups(sg_id1[1:], sg_id2)
            sg_id, adjoint2 = (False,) + so3_sg_id[:-1], so3_sg_id[-1]
    
        elif sg_id1[0] == 'cone':
            assert len(sg_id1) == 2
        
            N = sg_id1[1]
        
            assert isinstance(N, int)
        
            if N == -1 or N > 1:
                # O(2) cone symmetry
                # D_N pyramid symmetry

                if sg_id2[0] is not None:
                    sg_id = 'cone', sg_id2[1]
                    adjoint2 = sg1.element((0, sg_id2[0]))
                else:
                    sg_id = False, None, sg_id2[1]
            
            elif N == 1:
                # C_2 mirroring wrt a plane
                sg_id = sg_id1 if sg_id2 else self.subgroup_trivial_id
            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')
    
        else:
            raise ValueError(f'Subgroup "{sg_id}" not recognized!')
    
        if adjoint2 is not None:
            adjoint = adjoint1 @ inclusion(adjoint2)
        else:
            adjoint = adjoint1
        
        return sg_id + (adjoint,)

    def _restrict_irrep(self, irrep: Tuple, id: Tuple) -> Tuple[np.matrix, List[Tuple]]:
        r"""
        Restrict the input irrep of current group to the subgroup identified by "id".
        
        Args:
            irrep (tuple): the identifier of the irrep to restrict
            id (tuple): the identification of the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """
    
        sg_id, adj = id[:-1], id[-1]
    
        irr = self.irrep(*irrep)
        l = irr.attributes['frequency']
        j = irr.attributes['inv_frequency']
    
        sg, _, _ = self.subgroup(id)
    
        irreps = []
        change_of_basis = None
    
        try:
            if sg_id == (True, 'so3'):
                # O3 group itself
                irreps = [irr.id]
                change_of_basis = np.eye(2*l+1)
        
            elif sg_id == (False, 'so3'):
                # SO3 group
                irreps = [(l,)]
                change_of_basis = np.eye(2*l+1)
        
            elif sg_id[0] == 'fulltetra' and len(sg_id) == 1:
                # T_d group
                # the group is isomerphic to the Octahedral group
                # sg = Octahedral()
                raise NotImplementedError()
        
            elif sg_id[0] == True:
                # the rest of the id identifies a subgroup of the SO3 subgroup
                so3_sg_id = sg_id[1:]
            
                if len(so3_sg_id) == 1 and isinstance(so3_sg_id[0], str):
                    if so3_sg_id[0] == 'ico':
                        # I_h
                        raise NotImplementedError()
                    elif so3_sg_id[0] == 'octa':
                        # O_h
                        raise NotImplementedError()
                    elif so3_sg_id[0] == 'tetra':
                        # T_h
                        # Pyritohedral symmetry
                        raise NotImplementedError()
                    else:
                        raise ValueError(f'Subgroup "{so3_sg_id}" not recognized!')
            
                elif len(so3_sg_id) == 2:
                    # planar subgroups
                    assert isinstance(so3_sg_id[1], int) or so3_sg_id[1] is None
                
                    if so3_sg_id == (False, -1):
                        # SO(2) x C_2
                        raise NotImplementedError()
                    elif so3_sg_id == (True, -1):
                        # Cylinder,  O(2) x C_2
                        raise NotImplementedError()
                    elif so3_sg_id[0] is False and so3_sg_id[1] > 0:
                        # C_N x C_2
                    
                        N = so3_sg_id[1]
                    
                        if N == 1:
                            # Inversion subgroup C_2 of O(3)
                            irreps = [(j,)]*(2*l+1)
                            change_of_basis = np.eye(2 * l + 1)
                        else:
                            # C_N x C_2
                            raise NotImplementedError()
                    elif so3_sg_id[0] is True and so3_sg_id[1] > 1:
                        # Cylinder, D_N x C_2
                        raise NotImplementedError()
                    elif so3_sg_id[0] is True and so3_sg_id[1] == 1:
                        # flip subgroup of the O(2) subgroup, C_2 x C_2
                        # This case should have already been caught in the CyclicGroup case (thanks to the
                        # process_subgroup_id method)
                        # this is equivalent to C_2 x C_2 above
                        # TODO catch this case in process_subgroup_id
                        raise ValueError(f'Subgroup "{sg_id}": this case should have already been caught!')
                    else:
                        raise ValueError(f'Subgroup "{sg_id}" not recognized!')
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')
        
            elif sg_id[0] == False:
            
                # the rest of the id identifies a subgroup of the SO3 subgroup
                so3_sg_id = sg_id[1:]
            
                # First restrict to SO(3) and then use the restriction of its irreps to the subgroup
                so3_id = self._process_subgroup_id(('so3'))
                so3, _, _ = self.subgroup(so3_id)
            
                change_of_basis_so3, irreps_so3 = self._restrict_irrep(irrep, so3_id)
            
                so3_sg_id = so3._process_subgroup_id(so3_sg_id)
            
                p = 0
                cob_so3_sg = np.zeros((2 * l + 1, 2 * l + 1))
                for irr in irreps_so3:
                    cob_sg, irr_sg = so3._restrict_irrep(irr, so3_sg_id)
                    irreps += irr_sg
                
                    irr = so3.irrep(*irr)
                    cob_so3_sg[p:p + irr.size, p:p + irr.size] = cob_sg
                    p += irr.size
            
                change_of_basis = change_of_basis_so3 @ cob_so3_sg
        
            elif sg_id[0] == 'cone':
                assert len(sg_id) == 2
            
                N = sg_id[1]
            
                assert isinstance(N, int)
            
                if N == -1:
                    # O(2) cone symmetry
                
                    irreps = [((l + j) % 2, 0)] + [(1, f) for f in range(1, l + 1)]
                    change_of_basis = np.zeros((2 * l + 1, 2 * l + 1))
                    change_of_basis[l, 0] = 1.
                    for f in range(l):
                        change_of_basis[l + f + 1, 2 * f + 1] = 1.
                        change_of_basis[l - f - 1, 2 * f + 2] = 1.
                    
                        if (l + j) % 2 == 1:
                            # TODO - prove this!
                            block = change_of_basis[[l - f - 1, l + f + 1], 2 * f + 1:2 * f + 3]
                            m = np.asarray([
                                [0., -1],
                                [1., 0.]
                            ])
                            block = m @ block
                            change_of_basis[[l - f - 1, l + f + 1], 2 * f + 1:2 * f + 3] = block
            
                elif N > 1:
                    # D_N pyramid symmetry
                    raise NotImplementedError()
                elif N == 1:
                    # C_2 mirroring wrt a plane
                    raise NotImplementedError()
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')
        
            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')
    
        except NotImplementedError:
            change_of_basis, irreps = restrict_irrep(irr, sg_id)

        change_of_basis = self.irrep(*irrep)(adj).T @ change_of_basis
    
        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps for this group
        """
    
        # Build all the Irreducible Representations

        for f in range(2):
            for k in range(self._maximum_frequency + 1):
                self.irrep(f, k)
    
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0,0']

    def standard_representation(self) -> Representation:
        r"""
        Standard representation of :math:`\O3` as 3x3 rotation matrices

        """
        name = f'standard'
    
        if name not in self._representations:
            change_of_basis = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        
            self._representations[name] = change_basis(
                self.irrep(1, 1),
                change_of_basis=change_of_basis,
                name=name,
                supported_nonlinearities=self.irrep(1, 1).supported_nonlinearities,
            )
    
        return self._representations[name]

    def bl_regular_representation(self, L: int) -> Representation:
        r"""
        Band-Limited regular representation up to frequency ``L`` (included).

        Args:
            L(int): max frequency

        """
        irreps = []

        for l in range(L + 1):
            for j in range(2):
                irreps += [self.irrep(j, l)] * (2 * l + 1)
    
        return directsum(irreps, name=f'regular_{L}')

    def bl_quotient_representation(self,
                                   L: int,
                                   subgroup_id,
                                   ) -> escnn.group.Representation:
        r"""
        Band-Limited quotient representation up to frequency ``L`` (included).

        The quotient representation corresponds to the action of the current group :math:`G` on functions over the
        homogeneous space :math:`G/H`, where :math:`H` is the subgroup of :math:`G` identified by ``subgroup_id``.

        Args:
            L(int): max frequency
            subgroup_id: id identifying the subgroup H.

        """

        name = f"quotient[{subgroup_id}]_{L}"

        if name not in self.representations:
            subgroup, _, _ = self.subgroup(subgroup_id)

            homspace = self.homspace(subgroup_id)

            irreps = []

            for l in range(L + 1):
                for j in range(2):
                    irr = self.irrep(j, l)
                    multiplicity = homspace.dimension_basis(irr.id, homspace.H.trivial_representation.id)[1]
                    irreps += [irr] * multiplicity

            self.representations[name] = directsum(irreps, name=name)

        return self.representations[name]

    def bl_sphere_representation(self,
                                   L: int,
                                   ) -> escnn.group.Representation:
        r"""
        Representation of :math:`\O3` acting on functions on the sphere band-limited representation up to frequency
        ``L`` (included).

        Args:
            L(int): max frequency
        """
        return self.bl_quotient_representation(L, ('cone', -1))

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of (rotational) frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L, L
        irreps = []
        for l in range(L + 1):
            irreps += [(0, l), (1, l)]
        return irreps

    def irrep(self, j: int, l: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with reflection and rotation frequencies :math:`j` (reflection) and :math:`l` (rotation) of the
        current group.
        
        Args:
            j (int): the frequency of the reflection in the irrep
            l (int): the frequency of the rotations in the irrep

        Returns:
            the corresponding irrep

        """
    
        assert j in [0, 1]
        assert l >= 0
    
        name = f"irrep_{j},{l}"
        id = (j, l)

        if id not in self._irreps:

            if l == 0 and j == 0:
                # Trivial representation
                irrep = build_trivial_irrep()
                character = build_trivial_character()
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=0,
                                                              inv_frequency=0)
            elif l == 0 and j == 1:
                # Inversion representation
                # irrep = lambda element, identity=np.eye(1): identity * (-1 if element.to(element.group.PARAM)[0] else 1)
                # character = lambda element: (-1 if element.to(element.group.PARAM)[0] else 1)
                irrep = _build_irrep(j, l)
                character = _build_character(j, l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=0,
                                                              inv_frequency=1)
            else:

                # other Irreducible Representations
                
                # inv_action = lambda i, j=j:  (-1 if i else 1)**j
                # irrep = lambda element, l=l, inv_action=inv_action, **kwargs: _wigner_d_matrix(element.to(element.param)[1], l=l, **kwargs) * inv_action(element.to(element.param)[0])
                # character = lambda element, l=l, inv_action=inv_action, **kwargs: _character(element.to(element.param)[1], l=l, **kwargs) * inv_action(element.to(element.param)[0])
                irrep = _build_irrep(j, l)
                character = _build_character(j, l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2 * l + 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=l,
                                                              inv_frequency=j)

        return self._irreps[id]

    def _tensor_product_irreps(self, J: Tuple[int, int], l: Tuple[int, int]) -> List[Tuple[Tuple, int]]:
        J = self.get_irrep_id(J)
        l = self.get_irrep_id(l)
        return [
            (((J[0] + l[0]) % 2, j), 1)
            for j in range(np.abs(J[1] - l[1]), J[1] + l[1] + 1)
        ]

    def _clebsh_gordan_coeff(self, m, n, j) -> np.ndarray:
        group_keys = self._keys
        m = self.get_irrep_id(m)
        n = self.get_irrep_id(n)
        j = self.get_irrep_id(j)

        if (m[0] + n[0])%2 == j[0]:
            return _clebsh_gordan_tensor_so3(m[1], n[1], j[1])
        else:
            return np.zeros(self.irrep(*m).size, self.irrep(*n).size, 0, self.irrep(*j).size)

    _cached_group_instance = None

    @classmethod
    def _generator(cls, maximum_frequency: int = 3) -> 'O3':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = O3(maximum_frequency)
        elif cls._cached_group_instance._maximum_frequency < maximum_frequency:
            cls._cached_group_instance._maximum_frequency = maximum_frequency
            cls._cached_group_instance._build_representations()
    
        return cls._cached_group_instance


def _random_samples(N: int, seed = None, parametrization: str = PARAMETRIZATION):

    if seed is None:
        rng = np.random
    elif isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        assert isinstance(seed, np.random.RandomState)
        rng = seed

    so3_grid = _grid('rand', N=N, seed=rng, parametrization=parametrization)
    return [
        (rng.randint(0, 2), g) for g in so3_grid
    ]


def _build_irrep(j: int, l: int):

    def irrep(e: GroupElement, j: int = j, l: int = l):
        inversion, rotation = e.value

        inv_action = (-1 if inversion else 1) ** j

        if l == 0:
            return np.asarray([[inv_action]], dtype=float)
        else:
            return _wigner_d_matrix(rotation, l=l, param=e.param) * inv_action

    return irrep


def _build_character(j: int, l: int):
    
    def character(e: GroupElement, j: int = j, l: int = l):
        inversion, rotation = e.value
    
        inv_action = (-1 if inversion else 1) ** j
        if l == 0:
            return inv_action
        else:
            return _character(rotation, l, param=e.param) * inv_action
    
    return character


#############################################
# SUBGROUPS MAPS
#############################################

# Full Icosahedral I_h, Full Octahedral O_h or Pyritohedral T_h (not Full Tetra!!) ###############################

def o3_to_fullplato(adj: GroupElement, plato: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, plato=plato, adj=adj):
        o3 = adj.group
        assert e.group == o3

        try:
            return plato.element(
                (adj @ e @ (~adj)).to('Q'), # = (flip, rot)
                '[int | Q]'
            )
        except ValueError:
            return None
    
    return _map


def fullplato_to_o3(adj: GroupElement, plato: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, plato=plato, adj=adj):
        assert e.group == plato
        o3 = adj.group
        return (~adj) @ o3.element(e.to('[int | Q]'), 'Q') @ adj
    
    return _map


# SO3 (and Icosahedral, Octahedral, Tetrtahedral) ###############################

def o3_to_so3(adj: GroupElement, so3: Union[escnn.group.SO3, escnn.group.Icosahedral]):
    # can also support Tetra and Octa
    
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, so3=so3, adj=adj):
        o3 = adj.group
        assert e.group == o3
        
        flip, rotation = (adj @ e @ (~adj)).to('Q')
        
        if flip == 0:
            try:
                return so3.element(rotation, 'Q')
            except ValueError:
                return None
        else:
            return None
    
    return _map


def so3_to_o3(adj: GroupElement, so3: Union[escnn.group.SO3, escnn.group.Icosahedral]):
    # can also support Tetra and Octa
    
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, so3=so3, adj=adj):
        assert e.group == so3
        o3 = adj.group
        return (~adj) @ o3.element(
            (0, e.to('Q')),
            'Q'
        ) @ adj
    
    return _map


# SO(2) (and C_N) #####################################

def o3_to_so2(adj: GroupElement, so2: Union[escnn.group.SO2, escnn.group.CyclicGroup]):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, so2=so2, adj=adj):
        o3 = adj.group
        assert e.group == o3
        
        e = adj @ e @ (~adj)
        
        flip, e = e.to('Q')
        
        if flip == 1:
            return None
        
        if np.allclose(e[:2], 0.):
            # if the rotation is along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            try:
                return so2.element(theta, 'radians')
            except ValueError:
                return None
        else:
            return None
    
    return _map


def so2_to_o3(adj: GroupElement, so2: Union[escnn.group.SO2, escnn.group.CyclicGroup]):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, so2=so2, adj=adj):
        assert e.group == so2
        o3 = adj.group
        
        theta_2 = e.to('radians') / 2.
        
        q = np.asarray([0., 0., np.sin(theta_2), np.cos(theta_2)])
        
        return (~adj) @ o3.element((0, q), 'Q') @ adj
    
    return _map


# Dihedral (O(2) and D_N) ######################################

def o3_to_dih(adj: GroupElement, dih: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, dih=dih, adj=adj):
        o3 = adj.group
        assert e.group == o3
        
        e = adj @ e @ (~adj)
        
        flip, e = e.to('Q')
        
        if flip == 1:
            return None

        if np.allclose(e[:2], 0.):
            # if it is a rotation along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            flip = 0
            try:
                return dih.element((flip, theta), 'radians')
            except ValueError:
                return None
        elif np.allclose(e[2:], 0.):
            # if it is a rotation by 180 degrees around an axis perpendicular to Z, i.e. an axis in the XY plane
            c, s = e[:2]
            theta = 2 * np.arctan2(s, c)
            flip = 1
            try:
                return dih.element((flip, theta), 'radians')
            except ValueError:
                return None
        else:
            return None
    
    return _map


def dih_to_o3(adj: GroupElement, dih: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, O3)
    
    def _map(e: GroupElement, dih=dih, adj=adj):
        assert e.group == dih
        o3 = adj.group
        
        f, theta = e.to('radians')
        
        theta_2 = theta / 2.
        s, c = np.sin(theta_2), np.cos(theta_2)
        
        if f == 0:
            q = np.asarray([0., 0., s, c])
        elif f == 1:
            q = np.asarray([c, s, 0., 0.])
        else:
            raise ValueError()
        
        return (~adj) @ o3.element((0, q), 'Q') @ adj
    
    return _map


# Conical (O(2) and D_N) ######################################


def o3_to_con(adj: GroupElement, con: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, O3)

    o3 = adj.group
    reflection = o3.element(
        (1, np.asarray([0., 1., 0., 0.])),
        'Q'
    )

    def _map(e: GroupElement, con=con, adj=adj, reflection=reflection):
        o3 = adj.group
        assert e.group == o3
        
        e = adj @ e @ (~adj)

        _, p = (e @ reflection).to('Q')

        flip, e = e.to('Q')
        
        if flip == 0 and np.allclose(e[:2], 0.):
            # if it is a rotation along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            try:
                return con.element((0, theta), 'radians')
            except ValueError:
                return None
        elif flip == 1 and np.allclose(p[:2], 0.):
            # if it is a reflection along a plane perpendicular to the XY plane
            s, c = p[2:]
            theta = 2 * np.arctan2(s, c)
            try:
                return con.element((1, theta), 'radians')
            except ValueError:
                return None

        else:
            return None
    
    return _map


def con_to_o3(adj: GroupElement, con: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, O3)
    
    o3 = adj.group
    reflection = o3.element(
        (1, np.asarray([0., 1., 0., 0.])),
        'Q'
    )

    def _map(e: GroupElement, con=con, adj=adj, reflection=reflection):
        assert e.group == con
        o3 = adj.group
        
        f, theta = e.to('radians')
        
        theta_2 = theta / 2.
        s, c = np.sin(theta_2), np.cos(theta_2)

        e = o3.element(
            (0, np.asarray([0., 0., s, c])),
            'Q'
        )
        
        if f == 1:
            e = e @ (~reflection)
            
        return (~adj) @ e @ adj
    
    return _map


# Mirroring wrt a plane ######################################


def o3_to_mir(adj: GroupElement, mir: escnn.group.CyclicGroup):
    
    assert isinstance(adj.group, O3)
    assert isinstance(mir, escnn.group.CyclicGroup) and mir.order() == 2

    o3 = adj.group
    reflection = o3.element(
        (1, np.asarray([0., 1., 0., 0.])),
        'Q'
    )
    
    def _map(e: GroupElement, mir=mir, adj=adj, reflection=reflection):
        o3 = adj.group
        assert e.group == o3
        
        e = adj @ e @ (~adj)
        
        _, p = (e @ reflection).to('Q')
        
        flip, e = e.to('Q')
        
        if flip == 0 and np.allclose(e[:3], 0.):
            # if it is the identity
            return mir.identity
        elif flip == 1 and np.allclose(p[:3], 0.):
            # if it is a mirroring wrt the XZ plane, i.e. along the Y axis
            return mir.element(1, 'int')
        else:
            return None
    
    return _map


def mir_to_o3(adj: GroupElement, mir: escnn.group.CyclicGroup):
    
    assert isinstance(adj.group, O3)
    assert isinstance(mir, escnn.group.CyclicGroup) and mir.order() == 2

    o3 = adj.group
    reflection = o3.element(
        (1, np.asarray([0., 1., 0., 0.])),
        'Q'
    )
    
    def _map(e: GroupElement, mir=mir, adj=adj, reflection=reflection):
        assert e.group == mir
        o3 = adj.group
        
        f = e.to('int')
        
        if f == 0:
            return o3.identity
        else:
            return (~adj) @ reflection @ adj
    
    return _map


# Inversions subgroup ###############################

def o3_to_inv(o3: O3, inv: escnn.group.CyclicGroup):
    
    assert isinstance(inv, escnn.group.CyclicGroup) and inv.order() == 2
    
    def _map(e: GroupElement, inv=inv, o3=o3):
        
        assert e.group == o3

        flip, rotation = e.to('Q')
        
        if np.allclose(rotation[:3], 0.) and np.allclose(np.abs(rotation[3]), 1.):
            return inv.element(flip, 'int')
        else:
            return None
    
    return _map


def inv_to_o3(o3: O3, inv: escnn.group.CyclicGroup):
    
    assert isinstance(inv, escnn.group.CyclicGroup) and inv.order() == 2
    
    def _map(e: GroupElement, inv=inv, o3=o3):
        assert e.group == inv
        
        flip = e.to('int')
        
        return o3.element(
            (flip, np.array([0., 0., 0., 1.])),
            'Q'
        )
    
    return _map


# Cylinder C_2 x SO(2) (and C_2 x C_N) #####################################

def o3_to_cyl(adj: GroupElement, cyl: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    assert isinstance(cyl.G1, escnn.group.CyclicGroup) and cyl.G1.order() == 2
    assert isinstance(cyl.G2, escnn.group.CyclicGroup) or isinstance(cyl.G2, escnn.group.SO2)

    def _map(e: GroupElement, cyl=cyl, adj=adj):
        o3 = adj.group
        assert e.group == o3

        e = adj @ e @ (~adj)

        inv, e = e.to('Q')

        if np.allclose(e[:2], 0.):
            # if the rotation is along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            try:
                return cyl.element((inv, theta), '[int | radians]')
            except ValueError:
                return None
        else:
            return None

    return _map


def cyl_to_o3(adj: GroupElement, cyl: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    assert isinstance(cyl.G1, escnn.group.CyclicGroup) and cyl.G1.order() == 2
    assert isinstance(cyl.G2, escnn.group.CyclicGroup) or isinstance(cyl.G2, escnn.group.SO2)

    def _map(e: GroupElement, cyl=cyl, adj=adj):
        assert e.group == cyl
        o3 = adj.group

        inv, theta = e.to('[int | radians]')
        theta_2 = theta / 2.

        q = np.asarray([0., 0., np.sin(theta_2), np.cos(theta_2)])

        return (~adj) @ o3.element((inv, q), 'Q') @ adj

    return _map


# Full Cylinder C_2 x O(2) (and C_2 x D_N) #####################################

def o3_to_fullcyl(adj: GroupElement, fullcyl: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    assert isinstance(fullcyl.G1, escnn.group.CyclicGroup) and fullcyl.G1.order() == 2
    assert isinstance(fullcyl.G2, escnn.group.DihedralGroup) or isinstance(fullcyl.G2, escnn.group.O2)

    def _map(e: GroupElement, fullcyl=fullcyl, adj=adj):
        o3 = adj.group
        assert e.group == o3

        e = adj @ e @ (~adj)

        inv, e = e.to('Q')

        if np.allclose(e[:2], 0.):
            # if it is a rotation along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            flip = 0
            try:
                return fullcyl.element((inv, (flip, theta)), '[int | radians]')
            except ValueError:
                return None
        elif np.allclose(e[2:], 0.):
            # if it is a rotation by 180 degrees around an axis perpendicular to Z, i.e. an axis in the XY plane
            c, s = e[:2]
            theta = 2 * np.arctan2(s, c)
            flip = 1
            try:
                return fullcyl.element((inv, (flip, theta)), '[int | radians]')
            except ValueError:
                return None
        else:
            return None

    return _map


def fullcyl_to_o3(adj: GroupElement, fullcyl: escnn.group.DirectProductGroup):
    assert isinstance(adj.group, O3)
    assert isinstance(fullcyl.G1, escnn.group.CyclicGroup) and fullcyl.G1.order() == 2
    assert isinstance(fullcyl.G2, escnn.group.DihedralGroup) or isinstance(fullcyl.G2, escnn.group.O2)

    def _map(e: GroupElement, fullcyl=fullcyl, adj=adj):
        assert e.group == fullcyl
        o3 = adj.group

        inv, el_o2 = e.to('[int | radians]')
        f, theta = el_o2

        theta_2 = theta / 2.
        s, c = np.sin(theta_2), np.cos(theta_2)

        if f == 0:
            q = np.asarray([0., 0., s, c])
        elif f == 1:
            q = np.asarray([c, s, 0., 0.])
        else:
            raise ValueError()

        return (~adj) @ o3.element((inv, q), 'Q') @ adj

    return _map

############################################
