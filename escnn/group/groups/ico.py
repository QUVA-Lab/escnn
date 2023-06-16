from __future__ import annotations

from escnn.group import change_basis, directsum
from escnn.group.irrep import generate_irrep_matrices_from_generators
from escnn.group.irrep import restrict_irrep
from escnn.group.utils import cycle_isclose

from .utils import *

from .so3_utils import PARAMETRIZATION as PARAMETRIZATION_SO3
from .so3_utils import PARAMETRIZATIONS
from .so3_utils import IDENTITY, _grid, _combine, _equal, _invert, _change_param, _check_param, _hash

from .so3group import _build_character, _build_irrep

import numpy as np

from typing import Tuple, Callable, Iterable, List, Dict, Any, Union


__all__ = ["Icosahedral"]

_PHI = (1. + np.sqrt(5)) / 2


class Icosahedral(Group):

    PARAM = PARAMETRIZATION_SO3
    
    PARAMETRIZATIONS = PARAMETRIZATIONS
    
    def __init__(self):
        r"""
        
        Subgroup Structure:

        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |    ``id[0]``                      |    ``id[1]``                      |    subgroup                                                                                                       |
        +===================================+===================================+===================================================================================================================+
        |        'ico'                      |                                   |   The Icosahedral :math:`I` group itself                                                                          |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        'tetra'                    |                                   |   Tetrahedral :math:`T` subgroup                                                                                  |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        False                      |     N = 1, 2, 3, 5                |   :math:`C_N` of N discrete planar rotations                                                                      |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        True                       |     N = 2, 3, 5                   |   *dihedral* :math:`D_N` subgroup of N discrete planar rotations and out-of-plane :math:`\pi` rotation            |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        True                       |     1                             |   equivalent to ``(False, 2, adj)``                                                                               |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+

        """
        
        super(Icosahedral, self).__init__("Icosahedral", False, False)
        
        self._identity = self.element(IDENTITY)
        
        self._elements = [self.element(g) for g in _grid('ico')]
        assert len(self._elements) == 60
            
        # self._identity = self._elements[3]

        self._generators = [
            self._elements[21],                           # Cyclic Group of order 5
            self._elements[0],                            # Cyclic Group of order 2
            # self._elements[38]                          # Cyclic Group of order 3
            # self._elements[0] @ self._elements[21]      # Cyclic Group of order 3
        ]
        
        self._build_representations()

    @property
    def generators(self) -> List[GroupElement]:
        return self._generators

    @property
    def identity(self) -> GroupElement:
        return self._identity

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements
     
    @property
    def _keys(self) -> Dict[str, Any]:
        return dict()

    @property
    def subgroup_trivial_id(self):
        return (False, 1)

    @property
    def subgroup_self_id(self):
        return 'ico'

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

    def _hash_element(self, element, param=PARAM):
        return _hash(element, param)

    def _repr_element(self, element, param=PARAM):
        return element.__repr__()

    def _is_element(self, element,
                    param: str = PARAM,
                    verbose: bool = False,
                    ) -> bool:
    
        ATOL = 1e-7
        RTOL = 1e-5
        angle_ATOL = 1e-6
        
        if not _check_param(element, param):
            if verbose:
                print(f"Element {element} is not a rotation")
            return False
    
        element = self._change_param(element, param, 'Q')
        
        v = element[:3]
        theta = 2*np.arccos(np.clip(element[3], -1., 1.))
        
        if cycle_isclose(theta, 0., 2*np.pi, atol=angle_ATOL, rtol=0.):
            return True
        
        v = v / np.sin(theta/2.)
        
        if cycle_isclose(theta, 0., 2 * np.pi / 2, atol=angle_ATOL, rtol=0.):
            # rotation of order 2
            return _is_axis_aligned(v, 2, verbose=verbose, ATOL=ATOL, RTOL=RTOL)

        elif cycle_isclose(theta, 0., 2 * np.pi / 3, atol=angle_ATOL, rtol=0.):
            # rotation or order 3
            return _is_axis_aligned(v, 3, verbose=verbose, ATOL=ATOL, RTOL=RTOL)

        elif cycle_isclose(theta, 0., 2 * np.pi / 5, atol=angle_ATOL, rtol=0.):
            # rotation or order 5
            return _is_axis_aligned(v, 5, verbose=verbose, ATOL=ATOL, RTOL=RTOL)

        else:
            if verbose:
                print(f'Group element is neither a 2-fold, a 3-fold nor a 5-fold rotation.')
            return False

    def _change_param(self, element, p_from: str, p_to: str):
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS
        return _change_param(element, p_from, p_to)

    ###########################################################################

    def sample(self, param: str = PARAM) -> GroupElement:
        return self._elements[
            np.random.randint(self.order())
        ]

    def testing_elements(self) -> Iterable[GroupElement]:
    
        r"""
        A finite number of group elements to use for testing.
        """
        return iter(self._elements)

    def change_param(self, element, p_from, p_to):
        return _change_param(element, p_from, p_to)

    def __eq__(self, other):
        if not isinstance(other, Icosahedral):
            return False
        else:
            return self.name == other.name

    def _process_subgroup_id(self, id):

        if not isinstance(id, tuple):
            id = (id,)

        assert isinstance(id[0], bool) or isinstance(id[0], str), id[0]

        if not isinstance(id[-1], GroupElement):
            id = (*id, self.identity)

        assert id[-1].group == self

        if isinstance(id[0], bool):
            assert id[1] in [1, 2, 3, 5]

            if id[0] == True and id[1] == 1:
                # flip subgroup of the O(2) subgroup of SO(3)
                # this is equivalent to the C_2 subgroup of 180 deg rotations out of the plane (around X axis)

                # V = np.asarray([0., -_PHI, 1 / _PHI])
                V = np.array([1., 1., -1.])
                V /= np.linalg.norm(V)

                change_axis = np.zeros(4)
                change_axis[:3] = V * np.sin(np.pi/3.)
                change_axis[3] = np.cos(np.pi/3.)

                adj = self.element(change_axis, 'Q') @ id[-1]
                id = (False, 2, adj)

        return id

    def _subgroup(self, id) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        
        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
              
        """
        # TODO : implement this!

        sg = None
        parent_map = None
        child_map = None

        id, adj = id[:-1], id[-1]

        if id == ('ico',):
            sg = self
            parent_map = build_adjoint_map(self, ~adj)
            child_map = build_adjoint_map(self, adj)
        elif id == ('tetra',):
            raise NotImplementedError()
        elif id == (False, 1):
            sg = escnn.group.cyclic_group(1)
            parent_map, child_map = build_trivial_subgroup_maps(self)
        elif id == (False, 2):
            sg = escnn.group.cyclic_group(2)
            axis = np.asarray([0., 0., 1.])
            parent_map = cn_to_ico(adj, sg, axis=axis)
            child_map = ico_to_cn(adj, sg, axis=axis)
        elif id == (False, 3):
            sg = escnn.group.cyclic_group(3)
            axis = np.asarray([1., 1., 1.]) / np.sqrt(3)
            parent_map = cn_to_ico(adj, sg, axis=axis)
            child_map = ico_to_cn(adj, sg, axis=axis)
        elif id == (False, 5):
            sg = escnn.group.cyclic_group(5)
            axis = np.asarray([_PHI, 0., 1.])
            axis /= np.linalg.norm(axis)
            parent_map = cn_to_ico(adj, sg, axis=axis)
            child_map = ico_to_cn(adj, sg, axis=axis)
        elif id == (True, 2):
            sg = escnn.group.dihedral_group(2)
            parent_map, child_map = None, None
            raise NotImplementedError()
        elif id == (True, 3):
            sg = escnn.group.dihedral_group(3)
            parent_map, child_map = None, None
            raise NotImplementedError()
        elif id == (True, 5):
            sg = escnn.group.dihedral_group(5)
            parent_map, child_map = None, None
            raise NotImplementedError()
        else:
            raise ValueError(f'Subgroup id {id} not recognized!')

        return sg, parent_map, child_map

    def _restrict_irrep(self, irrep: str, id) -> Tuple[np.matrix, List[str]]:
        r"""
        
        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """
        sg_id, adj = id[:-1], id[-1]

        irr = self.irrep(*irrep)

        sg, _, _ = self.subgroup(id)

        irreps = []
        change_of_basis = None

        try:
            if sg_id == ('ico', ):
                change_of_basis = irr.change_of_basis
                irreps = irr.irreps
            elif sg_id == (False, 1):
                change_of_basis = np.eye(irr.size)
                irreps = [(0,)]*irr.size
            else:
                raise NotImplementedError()

        except NotImplementedError:
            if sg.order() > 0:
                change_of_basis, irreps = restrict_irrep(irr, sg_id)
            else:
                raise

        change_of_basis = self.irrep(*irrep)(adj).T @ change_of_basis

        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps for this group

        """

        # Build all the Irreducible Representations

        # add Trivial representation
        self.irrep(0)

        # add other irreducible representations
        self.irrep(1)
        self.irrep(2)

        # SO(3)'s freq 3 irrep decomposes in another 3 dimensional irrep and a 4 dimensional one
        self.irrep(3)
        self.irrep(4)

        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

        # build the regular representation

        # N.B.: it represents the LEFT-ACTION of the elements
        self.representations['regular'] = self.regular_representation

    @property
    def trivial_representation(self) -> Representation:
        return self.irrep(0)

    @property
    def standard_representation(self) -> Representation:
        r"""
        Restriction of the standard representation of SO(3) as 3x3 rotation matrices

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

    @property
    def ico_vertices_representation(self) -> Representation:
        # action on the 12 vertices of the icosahedron (or faces of the dodecahedron)
        
        # quotient repr wrt C_5 subgroup?
        return self.quotient_representation((False, 5), name='ico_vertices')

    @property
    def ico_faces_representation(self) -> Representation:
        # action on the 20 faces of the icosahedron (or vertices of the dodecahedron)
        
        # quotient repr wrt C_3 subgroup?
        return self.quotient_representation((False, 3), name='ico_faces')

    @property
    def ico_edges_representation(self) -> Representation:
        # action on the 30 edges of the icosahedron or dodecahedron
        
        # quotient repr wrt C_2 subgroup

        # n.b.: C_2 is the symmetry group of an edge

        return self.quotient_representation((False, 2), name='ico_edges')

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L <= 4, (L)
        return [(l,) for l in range(L+1)]

    def bl_regular_representation(self, L: int) -> Representation:
        r"""
        Band-Limited regular representation up to frequency ``L`` (included).

        Args:
            L(int): max frequency

        """

        assert isinstance(L, int)
        assert 0 <= L <= 4

        name = f'regular_{L}'

        if name not in self._representations:
            irreps = []

            for l in range(L + 1):
                irr = self.irrep(l)
                irreps += [irr] * irr.size

            self._representations[name] = directsum(irreps, name=name)

        return self._representations[name]

    def irrep(self, l: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep of :math:`I` identified by the non-negative integer :math:`l`.
        For :math:`l = 0, 1, 2`, the irrep is equivalent to the Wigner D matrix of the same frequency :math:`l`.
        For :math:`l=3`, the 7-dimensional Wigner D matrix is decomposed in a 3-dimensional and a 4-dimensional irrep,
        here identified respectively by :math:`l=3` and :math:`l=4`.
        
        Args:
            l (int): identifier of the irrep

        Returns:
            the corresponding irrep

        """
        
        assert isinstance(l, int)
        assert 0 <= l <= 4
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
            elif l <= 2:
        
                # other Irreducible Representations which are equivalent to Wigner D matrices
                # irrep = lambda element, l=l: _wigner_d_matrix(element.to(element.param), l=l, param=element.param)
                # character = lambda element, l=l: _character(element.to(element.param), l=l, param=element.param)
                irrep = _build_irrep(l)
                character = _build_character(l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2*l+1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=l)
            elif l == 3 or l == 4:
    
                irrep = _build_ico_irrep(self, l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, irrep[self.identity].shape[0], 'R',
                                                             supported_nonlinearities=supported_nonlinearities,
                                                             frequency=l)

            else:
                raise ValueError()

        return self._irreps[id]

    _cached_group_instance = None

    @classmethod
    def _generator(cls) -> 'Icosahedral':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = Icosahedral()
    
        return cls._cached_group_instance


def _is_axis_aligned(v: np.ndarray, n: int, verbose: bool = False, ATOL=1e-7, RTOL = 1e-5) -> bool:

    norm = np.linalg.norm(v)
    v = v / norm

    if n == 2:
        # rotation of order 2

        # the rotation axis need to pass through the center of an edge of the icosahedron or the dodecahedron
        # These 30 points can be found as any cyclic permutation or change of sign of each element of
        # these 2 vectors:
        p = np.array([1., 0., 0.])  # 6 combinations
        q = np.array([_PHI + 1, _PHI, 1.])  # 24 combinations
        q /= np.linalg.norm(q)

        # remove sign ambiguity
        v = np.abs(v)
        # fix a permutation making the highest value first
        m = np.argmax(v)
        v = np.roll(v, -m)

        ans = np.allclose(v, p, atol=ATOL, rtol=RTOL) or np.allclose(v, q, atol=ATOL, rtol=RTOL)

        if not ans and verbose:
            print(f'Rotation by a multiple of pi/2 not aligned with a 2-fold rotational axis of the Icosahedron.')

        return ans

    elif n == 3:
        # rotation or order 3

        # the rotation axis need to pass through a vertex of the dodecahedron
        # These 20 points can be found as any cyclic permutation or change of sign of each element of
        # these 2 vectors:

        p = np.array([1., 1., 1.])  # 8 combinations
        q = np.array([_PHI, 1 / _PHI, 0.])  # 12 combinations

        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)

        # remove sign ambiguity
        v = np.abs(v)
        # fix a permutation making the highest value first
        m = np.argmax(v)
        v = np.roll(v, -m)

        ans = np.allclose(v, p, atol=ATOL, rtol=RTOL) or np.allclose(v, q, atol=ATOL, rtol=RTOL)

        if not ans and verbose:
            print(f'Rotation by a multiple of 2pi/3 not aligned with a 3-fold rotational axis of the Icosahedron.')

        return ans

    elif n == 5:
        # rotation or order 5

        # the rotation axis need to pass through a vertex of the icosahedron
        # These 12 points can be found as any cyclic permutation or change of sign of each element of this vector:
        p = np.array([_PHI, 0., 1.])  # 12 combinations
        p /= np.linalg.norm(p)

        # remove sign ambiguity
        v = np.abs(v)
        # fix a permutation making the highest value first
        m = np.argmax(v)
        v = np.roll(v, -m)

        ans = np.allclose(v, p, atol=ATOL, rtol=RTOL)

        if not ans and verbose:
            print(f'Rotation by a multiple of 2pi/5 not aligned with a 5-fold rotational axis of the Icosahedron.')

        return ans

    else:
        raise  ValueError('The rotation order must be one of {2, 3, 5}.')


#############################################
# SUBGROUPS MAPS
#############################################

# C_N #####################################

def ico_to_cn(adj: GroupElement, cn: escnn.group.CyclicGroup, axis: np.ndarray):
    assert isinstance(adj.group, Icosahedral)

    assert axis.shape == (3,)
    assert np.isclose(np.linalg.norm(axis), 1.)

    assert cn.order() in [2, 3, 5]

    assert _is_axis_aligned(axis, cn.order())

    def _map(e: GroupElement, cn=cn, adj=adj, axis=axis):
        ico = adj.group
        assert e.group == ico

        e = adj @ e @ (~adj)

        e = e.to('Q')

        v = e[:3]

        n = np.linalg.norm(v)

        if np.allclose(n, 0.):
            return cn.identity
        elif np.allclose(v / n, axis):
            # if the rotation is along the axis
            s, c = n, e[-1]
            theta = 2 * np.arctan2(s, c)
            try:
                return cn.element(theta, 'radians')
            except ValueError:
                return None
        else:
            return None

    return _map


def cn_to_ico(adj: GroupElement, cn: escnn.group.CyclicGroup, axis: np.ndarray):
    assert isinstance(adj.group, Icosahedral)

    assert axis.shape == (3,)
    assert np.isclose(np.linalg.norm(axis), 1.)

    assert cn.order() in [2, 3, 5]

    assert _is_axis_aligned(axis, cn.order())

    def _map(e: GroupElement, cn=cn, adj=adj, axis=axis):
        assert e.group == cn
        ico = adj.group

        theta_2 = e.to('radians') / 2.

        q = np.empty(4)
        q[:3] = axis * np.sin(theta_2)
        q[-1] = np.cos(theta_2)

        return (~adj) @ ico.element(q, 'Q') @ adj

    return _map


#############################################
# Generate irreps
#############################################

from joblib import Memory
from escnn.group import __cache_path__
cache = Memory(__cache_path__, verbose=2)


def _build_ico_irrep(ico: Icosahedral, l: int):
    # To enable caching, the output of _build_ico_irrep_picklable needs to be picklable so it can not return a
    # dictionary with group elements as keys. In this method, we retrieved the cached results and wrap the keys into
    # group elements again
    irreps = _build_ico_irrep_picklable(ico, l)
    return {
            ico.element(g, param): v
            for g, param, v in irreps
    }


@cache.cache(ignore=['ico'])
def _build_ico_irrep_picklable(ico: Icosahedral, l: int) -> List[Tuple]:
    # To enable caching, the output of this method needs to be picklable so we can not return a dictionary with
    # group elements as keys

    if l == 3:
        
        # Representation of the generator of the cyclic subgroup of order 5
        rho_p = np.zeros((3, 3))
        
        rho_p[0, 0] = rho_p[1, 1] = np.cos(144 / 180. * np.pi)
        rho_p[1, 0] = np.sin(144 / 180. * np.pi)
        rho_p[0, 1] = -np.sin(144 / 180. * np.pi)
        rho_p[2, 2] = 1.
        
        # Representation of the generator of the cyclic subgroup of order 2
        rho_q = np.zeros((3, 3))
        rho_q[0, 0] = 1. / np.sqrt(5)
        rho_q[0, 2] = - 2. / np.sqrt(5)
        rho_q[1, 1] = - 1
        rho_q[2, 0] = - 2. / np.sqrt(5)
        rho_q[2, 2] = - 1. / np.sqrt(5)
        
    elif l == 4:

        # Representation of the generator of the cyclic subgroup of order 5
        rho_p = np.zeros((4, 4))

        rho_p[0, 0] = rho_p[1, 1] = np.cos(72 / 180. * np.pi)
        rho_p[1, 0] = np.sin(72 / 180. * np.pi)
        rho_p[0, 1] = -np.sin(72 / 180. * np.pi)

        rho_p[2, 2] = rho_p[3, 3] = np.cos(144 / 180. * np.pi)
        rho_p[3, 2] = np.sin(144 / 180. * np.pi)
        rho_p[2, 3] = -np.sin(144 / 180. * np.pi)

        # Representation of the generator of the cyclic subgroup of order 2
        rho_q = np.zeros((4, 4))
        rho_q[0, 2] = -1
        rho_q[1, 1] = 2. / np.sqrt(5)
        rho_q[1, 3] = 1. / np.sqrt(5)
        rho_q[2, 0] = -1
        rho_q[3, 1] = 1. / np.sqrt(5)
        rho_q[3, 3] = -2. / np.sqrt(5)

    else:
        raise ValueError()

    generators = [
        (ico._generators[0], rho_p),
        (ico._generators[1], rho_q),
    ]
    
    irreps = generate_irrep_matrices_from_generators(ico, generators)
    return [
        (k.value, k.param, v)
        for k, v in irreps.items()
    ]
