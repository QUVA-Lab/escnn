from __future__ import annotations

import escnn.group
from escnn.group import IrreducibleRepresentation
from escnn.group import directsum, change_basis
from escnn.group.irrep import restrict_irrep

from .so3_utils import *

from .utils import *

import numpy as np

from typing import Tuple, Callable, Iterable, List, Dict, Any, Union

try:
    import py3nj
except ImportError:
    import warnings
    warnings.warn("`py3nj` package not found! Will use a numerical method to compute the SO(3) Clebsh-Gordan coefficents. This is much slower but the coefficients will be cached on disk.")

    py3nj = None


__all__ = ["SO3"]


class SO3(Group):
    
    PARAM = PARAMETRIZATION

    PARAMETRIZATIONS = PARAMETRIZATIONS

    def __init__(self, maximum_frequency: int = 2):
        r"""
        Build an instance of the special orthogonal group :math:`SO(3)` which contains continuous rotations in the space.
        
        
        Subgroup Structure:
        
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |    ``id[0]``                      |    ``id[1]``                      |    subgroup                                                                                                                                                                                                   |
        +===================================+===================================+===============================================================================================================================================================================================================+
        |        'so3'                      |                                   |   :math:`SO(3)` group itself                                                                                                                                                                                  |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        'ico'                      |                                   |   Icosahedral :math:`I` subgroup                                                                                                                                                                              |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        'octa'                     |                                   |   Octahedral :math:`O` subgroup                                                                                                                                                                               |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        'tetra'                    |                                   |   Tetrahedral :math:`T` subgroup                                                                                                                                                                              |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        False                      |    -1                             |   :math:`SO(2)` subgroup of planar rotations around Z axis                                                                                                                                                    |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        False                      |     N                             |   :math:`C_N` of N discrete planar rotations around Z axis                                                                                                                                                    |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        True (or float)            |    -1                             |   *dihedral* :math:`O(2)` subgroup of planar rotations around Z axis and out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1] / 2`` (X axis by default)                           |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        True (or float)            |     N>1                           |   *dihedral* :math:`D_N` subgroup of N discrete planar rotations around Z axis and out-of-plane :math:`\pi` rotation around axis in the XY plane defined by ``id[1] / 2`` (X axis by default)                 |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |        True (or float)            |     1                             |   equivalent to ``(False, 2, adj)``                                                                                                                                                                           |
        +-----------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        
        Args:
            maximum_frequency (int, optional): the maximum frequency to consider when building the irreps of the group
        
        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(SO3, self).__init__("SO(3)", True, False)
        
        self._maximum_frequency = maximum_frequency
        
        self._identity = self.element(IDENTITY, PARAMETRIZATION)
        
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
    def elements(self) -> List[GroupElement]:
        return None
     
    @property
    def _keys(self) -> Dict[str, Any]:
        return dict()
    
    @property
    def subgroup_trivial_id(self):
        return (False, 1)

    @property
    def subgroup_self_id(self):
        return 'so3'

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################
    
    def _inverse(self, element, param=PARAM):
        r"""
        Return the inverse element of the input element
        """
        
        return _invert(element, param=param)

    def _combine(self, e1, e2,
                 param=PARAM,
                 param1=None,
                 param2=None
                 ):
        r"""
        Return the sum of the two input elements
        """
        return _combine(e1, e2, param=param, param1=param1, param2=param2)

    def _equal(self, e1, e2,
               param=PARAM,
               param1=None,
               param2=None,
               ) -> bool:
        r"""
        Check if the two input values corresponds to the same element
        """
        return _equal(e1, e2, param=param, param1=param1, param2=param2)

    def _hash_element(self, element, param: str = PARAM):
        return _hash(element, param)

    def _repr_element(self, element, param: str = PARAM) -> str:
        return _repr(element, param)

    def _is_element(self, element, param: str = PARAM, verbose: bool = False) -> bool:
    
        if not _check_param(element, param):
            if verbose:
                print(f"Element {element} is not a rotation")
            return False
    
        return True
    
    def _change_param(self, element, p_from: str, p_to: str):
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS
        return _change_param(element, p_from, p_to)

    ###########################################################################
    
    def sample(self, param: str = PARAM) -> GroupElement:
        return self.element(_grid('rand', N=1, parametrization=param)[0])
    
    def grid(self, type: str, *args, adj: GroupElement = None, parametrization: str = PARAMETRIZATION, **kwargs) -> List[GroupElement]:
        r"""
        Method which builds different collections of elements of :math:`\SO3`.

        Depending on the value of ``type``, the method accept a different set of parameters using the ``args`` and the
        ``kwargs`` arguments:
        
        * ``type = "rand"``. Generate ``N`` *random* uniformly distributed samples over the group. The method accepts an integer ``N`` and, optionally, a random seed ``seed`` (or an instance of :class:`numpy.random.RandomState`).
        
        * ``type = "thomson"``.  Generate ``N`` samples distributed (approximately) uniformly over the group. The samples are obtained with a numerical method which tries to minimize a potential energy depending on the relative distance between the points. The first call of this method for a particular value of ``N`` might be slow due to this numerical method. However, the resulting set of points is cached on disk such that following calls will return the same result instantly.

        * ``type = "thomson_cube"``.  Generate ``24*N`` samples with *cubic* (*octahedral*) symmetry, distributed (approximately) uniformly over the group. The samples are obtained with a numerical method which tries to minimize a potential energy depending on the relative distance between the points. The first call of this method for a particular value of ``N`` might be slow due to this numerical method. However, the resulting set of points is cached on disk such that following calls will return the same result instantly.

        * ``type = "ico", "cube", "tetra"``.  Generate respectively ``60``, ``24`` or ``12`` samples corresponding to the rotational symmetries of respectively the Icosahedron (or Dodecahedron), Octahedron (or Cube) or the Tetrahedron.

        * ``type = "hopf"``.  Generates *about* ``N`` points by creating a HEAPlix grid on the sphere and combining it with a regular grid over :math:`\SO2` using Hopf fibration.
        
        .. todo ::
            add reference to paper explaining Hopf fibration based grid
                              
        * ``type = "fibonacci"``.  Generates *about* ``N`` points by creating a Fibonacci grid on the sphere and combining it with a regular grid over :math:`\SO2` using Hopf fibration.
        
        Args:
            type (str): string identifying the type of samples
            *args: arguments specific for the type of samples chosen
            adj (GroupElement, optional): optionally, apply an adjoint transform to the sampled elements.
            parametrization (str, optional):
            **kwargs: arguments specific for the type of samples chosen

        Returns:
            a list of group elements

        """
        if adj is None:
            adj = self.identity
        adj = adj.to(PARAMETRIZATION)

        return [
            self.element(g, param=parametrization)
            for g in _grid(type, *args, adj=adj, parametrization=parametrization, **kwargs)
        ]

    def sphere_grid(self, type: str, *args, adj: GroupElement = None, **kwargs) -> List[GroupElement]:
        r"""
        
        Method which builds different collections of points over the sphere.
        
        Here, a sphere is interpreted as the quotient space :math:`\SO3 / \SO2`.
        The list returned by this method contains instances of :class:`~escnn.group.GroupElement`.
        These are elements of :class:`~escnn.group.SO3` and should be interpreted as *representatives* of cosets in
        :math:`\SO3 / \SO2`.

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
        adj = adj.to('MAT')

        points = _sphere_grid(type, *args, adj=adj, **kwargs)

        elements = np.zeros((len(points), 3, 3))
        elements[:, :, 2] = points
        y = np.random.randn(len(points), 3) - points
        elements[:, :, 1] = y - (y*points).sum(axis=1, keepdims=True)*points
        elements[:, :, 1] /= np.linalg.norm(elements[:, :, 1], axis=1, keepdims=True)
        elements[:, :, 0] = np.cross(elements[:, :, 2], elements[:, :, 1])
        det = np.linalg.det(elements)
        assert np.allclose(np.abs(det), 1.)
        elements[:, :, 2] *= det.reshape(-1, 1)

        return [
            self.element(g, param='MAT') for g in elements
        ]

    def testing_elements(self, n=3) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        return self.grid('ico')
        
        samples = np.empty((n**3, 3))
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
                    samples[i, :] = alpha, beta, gamma
                    i+=1

        samples = self._change_param(samples, p_from='ZYZ', p_to=self.PARAM)

        # samples = set()
        # i = 0
        # for a in range(n):
        #     alpha = 2 * np.pi * a / n
        #     for b in range(n):
        #         # beta = np.pi * (2 * b + 1) / (2*n)
        #         # beta = np.pi * (b + 1) / (n+2)
        #         # sample around b = 0  and b = np.pi to check singularities
        #         beta = np.pi * b / n
        #         for c in range(n):
        #             gamma = 2 * np.pi * c / n
        #             i+=1
        #             sample = alpha, beta, gamma
        #
        #             sample = self._change_param(sample, p_from='ZYZ', p_to='Q')
        #
        #             sign = np.heaviside(sample, 1.)*2 -1.
        #             sample = sample * sign.prod()
        #
        #             norm = np.linalg.norm(sample)
        #             assert np.allclose(norm, 1.), (alpha, beta, gamma, sample, sign, norm)
        #
        #             samples.add(tuple(sample.tolist()))
        #
        # samples = np.array(list(samples))
        # samples = self._change_param(samples, p_from='Q', p_to=PARAMETRIZATION)
        
        samples = [self.element(g, self.PARAM) for g in samples]
        
        return samples

    def __eq__(self, other):
        if not isinstance(other, SO3):
            return False
        else:
            return self.name == other.name # and self._maximum_frequency == other._maximum_frequency

    def _process_subgroup_id(self, id):

        if not isinstance(id, tuple):
            id = (id,)
    
        if not isinstance(id[-1], GroupElement):
            id = (*id, self.identity)
    
        assert id[-1].group == self

        if len(id) == 3 and isinstance(id[0], float):
            # O(2) (or subgroup) subgroup
            # if id[0] is bool, it specifies SO(2) vs O(2) subgroup
            # if id[0] is float, it assumes O(2) and it is twice the flip axis of O(2)
            # in this last case, we convert it to the boolean convention and include the flip axis inside the adjoing

            # The factor of 2 in id[0] comes from the fact that a flip around the axis theta, as an element of O(2), is
            # the combination of a reflection along the X axis (theta=0) and a rotation by 2*theta.
            # The values id[0] = 2*theta represents the SO(2) component of the flip.

            flip_axis = id[0] / 2.
            flip = np.asarray([0., 0., np.sin(flip_axis / 2), np.cos(flip_axis / 2)])
            flip = self.element(flip, 'Q')
            adjoint = (~flip) @ id[-1]
            id = (True, id[1], adjoint)
        
        assert isinstance(id[0], bool) or isinstance(id[0], str), id[0]

        if len(id) == 3 and id[0] == True and id[1] == 1:
            # flip subgroup of the O(2) subgroup of SO(3)
            # this is equivalent to the C_2 subgroup of 180 deg rotations out of the plane
            change_axis = np.asarray([0., np.sin(-np.pi / 4), 0., np.cos(-np.pi / 4)])
            adj = self.element(change_axis, 'Q') @ id[-1]
            id = (False, 2, adj)
    
        return id

    def _subgroup(self, id: Tuple) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        id can either be
            - a string in ['ico', 'octa' or 'tetra']
        or
            - a tuple [flip (out of plane 180 rot), N (number of discrete rots)]
        
        id can also be a tuple, whose first elements contain the id above, and the last is an element of SO(3) used
        to perform adjunction
        
        
        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
              
        """
        
        sg_id, adj = id[:-1], id[-1]
        assert isinstance(adj, GroupElement) and adj.group == self
        
        if len(sg_id) == 1 and isinstance(sg_id[0], str):
            sg_id = sg_id[0]
            if sg_id == 'so3':
                sg = self
                # parent_mapping = lambda x, adj=adj, _adj=~adj: _adj @ x @ adj
                parent_mapping = build_adjoint_map(self, ~adj)
                # child_mapping = lambda x, adj=adj, _adj=~adj: adj @ x @ _adj
                child_mapping = build_adjoint_map(self, adj)
            elif sg_id == 'ico':
                sg = escnn.group.ico_group()
                parent_mapping = rotgroup_to_so3(adj, sg)
                child_mapping = so3_to_rotgroup(adj, sg)
            elif sg_id == 'octa':
                sg = escnn.group.octa_group()
                parent_mapping = rotgroup_to_so3(adj, sg)
                child_mapping = so3_to_rotgroup(adj, sg)
            elif sg_id == 'tetra':
                # sg = Tetrahedral()
                # parent_mapping = ico_to_so3(adj, sg)
                # child_mapping = so3_to_ico(adj, sg)
                raise NotImplementedError()
            else:
                raise ValueError(f'Subgroup "{sg_id}" not recognized!')
            
        elif len(sg_id) == 2:
            # planar subgroups
            
            if sg_id == (False, -1):
                # SO(2)
                sg = escnn.group.so2_group(self._maximum_frequency)
                parent_mapping = so2_to_so3(adj, sg)
                child_mapping = so3_to_so2(adj, sg)
            elif sg_id == (True, -1):
                # O(2)

                sg = escnn.group.o2_group(self._maximum_frequency)
                parent_mapping = o2_to_so3(adj, sg)
                child_mapping = so3_to_o2(adj, sg)
            elif sg_id[0] == False and sg_id[1] > 0:
                # Cyclic subgroup
                # or flip subgroup of the O(2) subgroup
                n = sg_id[1]
                assert isinstance(n, int)
                sg = escnn.group.cyclic_group(n)
                parent_mapping = so2_to_so3(adj, sg)
                child_mapping = so3_to_so2(adj, sg)

            elif sg_id[0] == True and sg_id[1] > 1:
                # Dihedral subgroup
                n = sg_id[1]
                assert isinstance(n, int)
                sg = escnn.group.dihedral_group(n)

                parent_mapping = o2_to_so3(adj, sg)
                child_mapping = so3_to_o2(adj, sg)

            elif sg_id[0] == True and sg_id[1] == 1:
                # flip subgroup of the O(2) subgroup
                # This case should have already been caught in the CyclicGroup case (thanks to the
                # process_subgroup_id method)
                raise ValueError(f'Subgroup "{sg_id}": this case should have already been caught!')
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
    
        if sg_id1 == ('so3',):
            # subgroup of SO3
            sg_id = sg_id2[:-1]
            adjoint2 = inclusion(sg_id2[-1])

        elif sg_id1[0] == 'ico' and len(sg_id1) == 1:
            # I
            if sg_id2[0] in ['ico', 'tetra']:
                sg_id = sg_id2[0],
                adjoint2 = inclusion(sg_id2[1])
            else:
                flip, n, adjoint_ico = sg_id2
                sg_id = flip, n

                adjoint2 = inclusion(adjoint_ico)

                if n == 3:
                    axis = np.asarray([1., 1., 1.]) / np.sqrt(3)
                    rot = find_rotation_pole2point(axis)

                    adjoint2 = ~self.element(rot, 'MAT') @ adjoint2
                elif n == 5:
                    from escnn.group.groups.ico import _PHI
                    axis = np.asarray([_PHI, 0., 1.])
                    axis /= np.linalg.norm(axis)
                    rot = find_rotation_pole2point(axis)
                    adjoint2 = ~self.element(rot, 'MAT') @ adjoint2
                else:
                    assert n in [1, 2]

        elif sg_id1[0] in ['octa', 'tetra'] and len(sg_id1) == 1:
            # O
            # T
            raise NotImplementedError()

        # planar subgroups
        elif sg_id1[0] == False and isinstance(sg_id1[1], int):
            # SO(2) or C_N
            sg_id = False, sg_id2

        elif sg_id1[0] == True and isinstance(sg_id1[1], int):
            # O(2) or D_N
            flip = sg_id2[0] is not None
            if flip and sg_id1[1] == -1:
                sg_id = sg_id2
            elif flip and sg_id1[1] > 0:
                sg_id = sg_id2[0] * 2 * np.pi / sg_id1[1], sg_id2[1]
            else:
                sg_id = False, sg_id2[1]

        elif sg_id1[0] is True and sg_id1[1] == 1:
            # flip subgroup of O(2), C_2
            # This case should have already been caught in the CyclicGroup case (thanks to the
            # process_subgroup_id method)
            # this is equivalent to C_2 above
            raise ValueError(f'Subgroup "{sg_id1}": this case should have already been caught!')
        else:
            raise ValueError(f'Subgroup "{sg_id1}" not recognized!')
    
        if adjoint2 is not None:
            adjoint = adjoint1 @ adjoint2
        else:
            adjoint = adjoint1
    
        return sg_id + (adjoint,)

    def _restrict_irrep(self, irrep: Tuple, id) -> Tuple[np.matrix, List[Tuple]]:
        r"""
        
        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """
        
        sg_id, adj = id[:-1], id[-1]
        
        irr = self.irrep(*irrep)
        l = irr.attributes['frequency']

        sg, _, _ = self.subgroup(id)

        irreps = []
        change_of_basis = None
        
        try:
            if len(sg_id) == 1 and isinstance(sg_id[0], str):
                sg_id = sg_id[0]
                if sg_id == 'so3':
                    change_of_basis = irr.change_of_basis
                    irreps = irr.irreps
                elif sg_id == 'ico':
                    if l <= 2:
                        change_of_basis = np.eye(irr.size)
                        irreps = [(l,)]
                    else:
                        raise NotImplementedError()
                elif sg_id == 'octa':
                    if l <= 1:
                        change_of_basis = np.eye(irr.size)
                        irreps = [(l,)]
                    else:
                        raise NotImplementedError()
                elif sg_id == 'tetra':
                    raise NotImplementedError()
                else:
                    raise ValueError(f'Subgroup "{sg_id}" not recognized!')

            elif len(sg_id) == 2:
                # planar subgroups
        
                if sg_id == (False, -1):
                    # SO(2)
                    irreps = [(f,) for f in range(l+1)]
                    change_of_basis = np.zeros((2*l+1, 2*l+1))
                    change_of_basis[l, 0] = 1.
                    for f in range(l):
                        change_of_basis[l+f+1, 2*f+1] = 1.
                        change_of_basis[l-f-1, 2*f+2] = 1.

                elif sg_id == (True, -1):
                    # O(2)
                    j = l % 2
                    irreps = [(j, 0)] + [(1, f) for f in range(1, l+1)]
                    change_of_basis = np.zeros((2*l+1, 2*l+1))
                    change_of_basis[l, 0] = 1.
                    for f in range(l):
                        change_of_basis[l + f + 1, 2 * f + 1] = 1.
                        change_of_basis[l - f - 1, 2 * f + 2] = 1.
                        
                        if (l+f) % 2 == 0:
                            # TODO - prove this!
                            block = change_of_basis[[l - f - 1, l+f+1], 2*f+1:2*f+3]
                            block = block @ np.asarray([
                                [0., -1],
                                [1., 0.]
                            ])
                            change_of_basis[[l - f - 1, l + f + 1], 2 * f + 1:2 * f + 3] = block
                            
                elif sg_id[0] == False and sg_id[1] > 0:
                    # Cyclic Group
                    
                    # restrict first to SO(2) and then reuse the irreps restriction from SO(2) to C_N
                    
                    so2_sgid = self._process_subgroup_id((False, -1))
                    change_of_basis_so2, irreps_so2 = self._restrict_irrep(irrep, so2_sgid)
                    so2, _, _ = self.subgroup(so2_sgid)
                    
                    p = 0
                    cob_so2_cn = np.zeros((2*l+1, 2*l+1))
                    for irr in irreps_so2:
                        cob_cn, irr_cn = so2._restrict_irrep(irr, sg_id[1])
                        irreps += irr_cn
                        
                        irr = so2.irrep(*irr)
                        cob_so2_cn[p:p+irr.size, p:p+irr.size] = cob_cn
                        p += irr.size
                    
                    change_of_basis = change_of_basis_so2 @ cob_so2_cn

                elif sg_id[0] == True and sg_id[1] > 1:
                    # Dihedral Group
                    
                    # restrict first to O(2) and then reuse the irreps restriction from O(2) to D_N

                    o2_sgid = self._process_subgroup_id((True, -1))
                    change_of_basis_o2, irreps_o2 = self._restrict_irrep(irrep, o2_sgid)
                    o2, _, _ = self.subgroup(o2_sgid)

                    p = 0
                    cob_o2_cn = np.zeros((2 * l + 1, 2 * l + 1))
                    for irr in irreps_o2:
                        cob_cn, irr_cn = o2._restrict_irrep(irr, (0., sg_id[1]))
                        irreps += irr_cn
        
                        irr = o2.irrep(*irr)
                        cob_o2_cn[p:p + irr.size, p:p + irr.size] = cob_cn
                        p += irr.size

                    change_of_basis = change_of_basis_o2 @ cob_o2_cn
                    
                elif sg_id[0] == True and sg_id[1] == 1:
                    # flip subgroup of the O(2) subgroup
                    # This case should have already been caught in the CyclicGroup case (thanks to the
                    # process_subgroup_id method)
                    raise ValueError(f'Subgroup "{sg_id}": this case should have already been caught!')
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
    
        k = 0
    
        # add Trivial representation
        self.irrep(k)
    
        # add other irreducible representations
        for k in range(1, self._maximum_frequency + 1):
            self.irrep(k)
    
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

    @property
    def trivial_representation(self) -> Representation:
        return self.irrep(0)

    def bl_regular_representation(self, L: int) -> Representation:
        r"""
        Band-Limited regular representation up to frequency ``L`` (included).

        Args:
            L(int): max frequency

        """
        name = f'regular_{L}'
        
        if name not in self._representations:
            irreps = []
            
            for l in range(L+1):
                irreps += [self.irrep(l)]*(2*l+1)
            
            self._representations[name] = directsum(irreps, name=name)

        return self._representations[name]

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

            for l in range(L+1):
                irr = self.irrep(l)
                multiplicity = homspace.dimension_basis(irr.id, homspace.H.trivial_representation.id)[1]
                irreps += [irr] * multiplicity

            self.representations[name] = directsum(irreps, name=name)

        return self.representations[name]

    def bl_sphere_representation(self,
                                 L: int,
                                 ) -> escnn.group.Representation:
        r"""
        Representation of :math:`\SO3` acting on functions on the sphere band-limited representation up to frequency
        ``L`` (included).

        Args:
            L(int): max frequency
        """
        return self.bl_quotient_representation(L, (False, -1))

    def bl_induced_representation(self,
                                  L: int,
                                  repr: escnn.group.IrreducibleRepresentation,
                                  subgroup_id,
                                  ) -> escnn.group.Representation:
        r"""
        Band-Limited induced representation from the input representation ``repr`` up to frequency ``L`` (included).

        Args:
            L(int): max frequency
            repr (IrreducibleRepresentation): the representation of the subgroup
            subgroup_id: id identifying the subgroup H.

        """

        name = f"induced[{subgroup_id}][{repr.name}]_{L}"

        if name not in self.representations:
            subgroup, _, _ = self.subgroup(subgroup_id)

            homspace = self.homspace(subgroup_id)
            assert isinstance(repr, IrreducibleRepresentation)
            assert repr.group == subgroup

            irreps = []

            for l in range(L + 1):
                irr = self.irrep(l)
                multiplicity = homspace.dimension_basis(irr.id, repr.id)[1]
                irreps += [irr] * multiplicity

            assert len(irreps) > 0, f'Error! The induced representation from {repr.name} band-limited to {L} does not contain any irreps'

            self.representations[name] = directsum(irreps, name=name)

        return self.representations[name]

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L, L
        return [(l,) for l in range(L+1)]

    # @property
    def standard_representation(self) -> Representation:
        r"""
        Standard representation of :math:`\SO3` as 3x3 rotation matrices

        """
        name = f'standard'
    
        if name not in self._representations:
            
            change_of_basis = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            
            self._representations[name] = change_basis(
                self.irrep(1),
                change_of_basis=change_of_basis,
                name=name,
                supported_nonlinearities=self.irrep(1).supported_nonlinearities,
            )
    
        return self._representations[name]

    def irrep(self, l: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with rotational frequency :math:`l` of :math:`SO(3)`.
        Notice: the frequency has to be a non-negative integer.
        
        Args:
            l (int): the frequency of the irrep

        Returns:
            the corresponding irrep

        """
    
        assert l >= 0
        name = f"irrep_{l}"
        id = (l,)

        if id not in self._irreps:

            if l == 0:
                # Trivial representation
                irrep = build_trivial_irrep()
                character = build_trivial_character()
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=0
                                                              )
            else:
        
                # other Irreducible Representations
                irrep = _build_irrep(l)
                character = _build_character(l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2*l+1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=l)

        return self._irreps[id]

    def _clebsh_gordan_coeff(self, m, n, j) -> np.ndarray:
        group_keys = self._keys
        m = self.get_irrep_id(m)
        n = self.get_irrep_id(n)
        j = self.get_irrep_id(j)

        return _clebsh_gordan_tensor_so3(m[0], n[0], j[0])

    def _tensor_product_irreps(self, J: int, l: int) -> List[Tuple[Tuple, int]]:
        J, = self.get_irrep_id(J)
        l, = self.get_irrep_id(l)
        return [
            ((j,), 1)
            for j in range(np.abs(J - l), J + l + 1)
        ]

    _cached_group_instance = None

    @classmethod
    def _generator(cls, maximum_frequency: int = 3) -> 'SO3':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = SO3(maximum_frequency)
        elif cls._cached_group_instance._maximum_frequency < maximum_frequency:
            cls._cached_group_instance._maximum_frequency = maximum_frequency
            cls._cached_group_instance._build_representations()
    
        return cls._cached_group_instance


def _clebsh_gordan_tensor_so3(m: int, n: int, j: int):
    if py3nj is None:
        return escnn.group._clebsh_gordan._clebsh_gordan_tensor((m,), (n,), (j,), SO3.__name__)
    else:
        m1 = np.arange(-m, m + 1).reshape(-1, 1, 1)
        m2 = np.arange(-n, n + 1).reshape(1, -1, 1)
        M = np.arange(-j, j + 1).reshape(1, 1, -1)
        _m = np.array([m]).reshape(1, 1, 1)
        _n = np.array([n]).reshape(1, 1, 1)
        _j = np.array([j]).reshape(1, 1, 1)
        cg = py3nj.clebsch_gordan(2 * _m, 2 * _n, 2 * _j, 2 * m1, 2 * m2, 2 * M)

        cob_m = _change_of_basis_real2complex(m)
        cob_n = _change_of_basis_real2complex(n)
        cob_j = _change_of_basis_real2complex(j).T.conj()

        cg = np.einsum('jlJ,KJ->jlK', cg, cob_j)
        cg = np.einsum('jlJ,jk->klJ', cg, cob_m)
        cg = np.einsum('jlJ,lk->jkJ', cg, cob_n)
        if (n + m + j) % 2 == 0:
            return cg.real.reshape(2 * m + 1, 2 * n + 1, 1, 2 * j + 1)
        else:
            return cg.imag.reshape(2 * m + 1, 2 * n + 1, 1, 2 * j + 1)


def _build_irrep(l: int):
    def irrep(e: GroupElement, l: int = l):
        return _wigner_d_matrix(e.value, l, param=e.param)
    
    return irrep


def _build_character(l: int):
    
    def character(e: GroupElement, l: int = l):
        return _character(e.value, l, param=e.param)
    
    return character

#############################################
# SUBGROUPS MAPS
#############################################


# SO(3) and Polyhedrons' Symmetries ###############################

def so3_to_rotgroup(adj: GroupElement, rotgroup: escnn.group.Group):
    assert isinstance(adj.group, SO3)
    assert 'Q' in rotgroup.PARAMETRIZATIONS
    
    def _map(e: GroupElement, rotgroup=rotgroup, adj=adj):
        so3 = adj.group
        assert e.group == so3
        try:
            return rotgroup.element(
                (adj @ e @ (~adj)).to('Q'),
                'Q'
            )
        except ValueError:
            return None

    return _map


def rotgroup_to_so3(adj: GroupElement, rotgroup: escnn.group.Group):
    assert isinstance(adj.group, SO3)
    assert 'Q' in rotgroup.PARAMETRIZATIONS

    def _map(e: GroupElement, rotgroup=rotgroup, adj=adj):
        assert e.group == rotgroup
        so3 = adj.group
        return (~adj) @ so3.element(e.to('Q'), 'Q') @ adj
    
    return _map


# SO(2) (and C_N) #####################################

def so3_to_so2(adj: GroupElement, so2: Union[escnn.group.SO2, escnn.group.CyclicGroup]):
    assert isinstance(adj.group, SO3)
    
    def _map(e: GroupElement, so2=so2, adj=adj):
        so3 = adj.group
        assert e.group == so3
        
        e = adj @ e @ (~adj)
        
        e = e.to('Q')
        
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


def so2_to_so3(adj: GroupElement, so2: Union[escnn.group.SO2, escnn.group.CyclicGroup]):
    assert isinstance(adj.group, SO3)
    
    def _map(e: GroupElement, so2=so2, adj=adj):
        assert e.group == so2
        so3 = adj.group
        
        theta_2 = e.to('radians') / 2.
        
        q = np.asarray([0., 0., np.sin(theta_2), np.cos(theta_2)])
        
        return (~adj) @ so3.element(q, 'Q') @ adj
    
    return _map


# O(2) (and D_N) ######################################

def so3_to_o2(adj: GroupElement, o2: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, SO3)
    
    def _map(e: GroupElement, o2=o2, adj=adj):
        so3 = adj.group
        assert e.group == so3
        
        e = adj @ e @ (~adj)
        
        e = e.to('Q')
        
        if np.allclose(e[:2], 0.):
            # if it is a rotation along the Z axis, i.e. on the XY plane
            s, c = e[2:]
            theta = 2 * np.arctan2(s, c)
            flip = 0
            try:
                return o2.element((flip, theta), 'radians')
            except ValueError:
                return None
        elif np.allclose(e[2:], 0.):
            # if it is a rotation by 180 degrees around an axis perpendicular to Z, i.e. an axis in the XY plane
            c, s = e[:2]
            theta = 2 * np.arctan2(s, c)
            flip = 1
            try:
                return o2.element((flip, theta), 'radians')
            except ValueError:
                return None
        else:
            return None
    
    return _map


def o2_to_so3(adj: GroupElement, o2: Union[escnn.group.O2, escnn.group.DihedralGroup]):
    assert isinstance(adj.group, SO3)
    
    def _map(e: GroupElement, o2=o2, adj=adj):
        assert e.group == o2
        so3 = adj.group
        
        f, theta = e.to('radians')
        
        theta_2 = theta / 2.
        s, c = np.sin(theta_2), np.cos(theta_2)

        if f == 0:
            q = np.asarray([0., 0., s, c])
        elif f == 1:
            q = np.asarray([c, s, 0., 0.])
        else:
            raise ValueError()

        return (~adj) @ so3.element(q, 'Q') @ adj
    
    return _map


############################################


def check_parametrization():
    # check change of parametrizations
    
    parameterizations = (
        'Q',
        'MAT',
        'EV',
        'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', # Type-I Euler angles AKA Tait-Bryan angles
        'xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz',  # Type-II Euler angles
        
        'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX',  # Type-I Euler angles AKA Tait-Bryan angles
        'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ',  # Type-II Euler angles
    )
    
    G = SO3(2)
    
    er = set()
    
    for p2 in ['Q']:
        for p1 in parameterizations:
            for g in G.testing_elements(12):
                
                g_ = g._element
                g0 = _change_param(g_, p_from=PARAMETRIZATION, p_to=p1)
                g2 = _change_param(g0, p_from=p1, p_to=p2)
                g1 = _change_param(g2, p_from=p2, p_to=p1)
                
                if not G._equal(g0, g1, param1=p1, param2=p1):
                    print(g_)
                    print(g0)
                    print(g2)
                    print(g1)
                    print('-----------------------')
                    er.add((p1, p2))
                
                # assert G._equal(g0, g1, param1=p1, param2=p1, param=PARAMETRIZATION)#, (g, p1, p2)
    
    print(sorted(list(er)))
    
    for p1 in parameterizations:
        for g in G.testing_elements(5):
            g = g._element
            g0 = _change_param(g, p_from=PARAMETRIZATION, p_to=p1)
            g1 = _change_param(g0, p_from=p1, p_to='Q')
            
            assert np.allclose(np.linalg.norm(g1), 1.), (p1, g)
        

if __name__ == '__main__':
    check_parametrization()
    
    print(_change_param(np.array([0., 0., 0.]), 'zyz', 'Q'))
    
    for i in range(100):
        q = np.random.randn(4)
        q[np.random.randint(4)] = 0.
        q /= np.linalg.norm(q)

        m = _change_param(q, 'Q', 'MAT')
        q2 = _change_param(m, 'MAT', 'Q')
        np.set_printoptions(precision=3, suppress=True)
        if not (np.allclose(q, q2) or np.allclose(q, -q2)):
            print(q)
            print(q2)
            print('--------------')






