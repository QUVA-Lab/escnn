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


__all__ = ["Octahedral"]

_PHI = (1. + np.sqrt(5)) / 2


class Octahedral(Group):

    PARAM = PARAMETRIZATION_SO3
    
    PARAMETRIZATIONS = PARAMETRIZATIONS
    
    def __init__(self):
        r"""

        Subgroup Structure:

        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |    ``id[0]``                      |    ``id[1]``                      |    subgroup                                                                                                       |
        +===================================+===================================+===================================================================================================================+
        |        'octa'                     |                                   |   The Octahedral :math:`O` group itself                                                                           |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        'tetra'                    |                                   |   Tetrahedral :math:`T` subgroup                                                                                  |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        False                      |     N = 1, 2, 3, 4                |   :math:`C_N` of N discrete planar rotations                                                                      |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        True                       |     N = 2, 3, 4                   |   *dihedral* :math:`D_N` subgroup of N discrete planar rotations and out-of-plane :math:`\pi` rotation            |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+
        |        True                       |     1                             |   equivalent to ``(False, 2, adj)``                                                                               |
        +-----------------------------------+-----------------------------------+-------------------------------------------------------------------------------------------------------------------+


        """
        
        super(Octahedral, self).__init__("Octahedral", False, False)
        
        self._identity = self.element(IDENTITY)
        
        self._elements = [self.element(g) for g in _grid('cube')]
        assert len(self._elements) == 24
            
        # self._identity = self._elements[3]

        self._generators = [
            self._elements[17],  # Cyclic Group of order 4
            self._elements[11],  # Cyclic Group of order 3
            self._elements[22],  # Cyclic Group of order 2
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
        raise NotImplementedError

    @property
    def subgroup_self_id(self):
        raise NotImplementedError
        return 'octa'

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

        if not _check_param(element, param):
            if verbose:
                print(f"Element {element} is not a rotation")
            return False

        # convert to matrix representation
        element = self._change_param(element, param, 'MAT')

        # take absolute value of the elements
        # note that we have already ensured that the determinant is positive using `_check_param` above since it checks
        # that it is a rotation
        at = np.abs(element)

        # check if the matrix is a permutation matrix
        ans = (
                np.isclose(at.sum(axis=0), 1., atol=ATOL, rtol=RTOL).all()
            and np.isclose(at.sum(axis=1), 1., atol=ATOL, rtol=RTOL).all()
            and (np.isclose(at, 1., atol=ATOL, rtol=RTOL) | np.isclose(at, 0., atol=ATOL, rtol=RTOL)).all()
        )

        return ans

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

    def __eq__(self, other):
        if not isinstance(other, Octahedral):
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
            assert id[1] in [1, 2, 3, 4]

            if id[0] == True and id[1] == 1:
                # flip subgroup of the O(2) subgroup of SO(3)
                # this is equivalent to the C_2 subgroup of 180 deg rotations out of the plane (around X axis)

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

        if id == ('octa',):
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
            parent_map = cn_to_octa(adj, sg, axis=axis)
            child_map = octa_to_cn(adj, sg, axis=axis)
        elif id == (False, 3):
            sg = escnn.group.cyclic_group(3)
            axis = np.asarray([1., 1., 1.]) / np.sqrt(3)
            parent_map = cn_to_octa(adj, sg, axis=axis)
            child_map = octa_to_cn(adj, sg, axis=axis)
        elif id == (False, 4):
            sg = escnn.group.cyclic_group(4)
            axis = np.asarray([0., 0., 1.])
            axis /= np.linalg.norm(axis)
            parent_map = cn_to_octa(adj, sg, axis=axis)
            child_map = octa_to_cn(adj, sg, axis=axis)
        elif id == (True, 2):
            sg = escnn.group.dihedral_group(2)
            parent_map, child_map = None, None
            raise NotImplementedError()
        elif id == (True, 3):
            sg = escnn.group.dihedral_group(3)
            parent_map, child_map = None, None
            raise NotImplementedError()
        elif id == (True, 4):
            sg = escnn.group.dihedral_group(4)
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
            if sg_id == ('octa', ):
                change_of_basis = irr.change_of_basis
                irreps = irr.irreps
            elif sg_id == (False, 1):
                change_of_basis = np.eye(irr.size)
                irreps = [(1,)]*irr.size
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

        # Frequency 1 Wigner D matrix
        self.irrep(1)

        # Frequency 2 Wigner D matrix decomposes as a direct sum of a 2 and a 3 dimensional irrep
        self.irrep(-1) # 3 dimensional irrep
        self.irrep(2) # 2 dimensional irrep

        # SO(3)'s freq 3 irrep decomposes in a 1-dimensional irrep and the sum of the two previous 3 dimensional irreps
        self.irrep(3) # 1 dimensional

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
    def cube_vertices_representation(self) -> Representation:
        # action on the 8 vertices of the cube (or faces of the octahedron)

        sgid = (False, 3)
        return self.quotient_representation(sgid, name='cube_vertices')

    @property
    def cube_faces_representation(self) -> Representation:
        # action on the 6 faces of the cube (or vertices of the octahedron)

        sgid = (False, 4)
        return self.quotient_representation(sgid, name='cube_faces')

    @property
    def cube_edges_representation(self) -> Representation:
        # action on the 12 edges of the cube or octahedron

        sgid = (True, 1)
        return self.quotient_representation(sgid, name='cube_edges')

    def irrep(self, l: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep of :math:`O` identified by the integer :math:`l`.
        For :math:`l = 0, 1`, the irrep is equivalent to the Wigner D matrix of the same frequency :math:`l`.
        For :math:`l=2`, the 5-dimensional Wigner D matrix is decomposed in a 3-dimensional and a 2-dimensional irreps,
        here identified respectively by :math:`l=-1` and :math:`l=2`.
        For :math:`l=3`, the 7-dimensional Wigner D matrix is decomposed in a 1-dimensional irrep and the two previous
        3-dimensional irreps, here identified respectively by :math:`l=3` and :math:`l=1, -1`.
        
        Args:
            l (int): identifier of the irrep

        Returns:
            the corresponding irrep

        """
        
        assert isinstance(l, int)
        assert -1 <= l <= 3

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
                                                              )
            elif l == 1:
        
                # Irreducible Representation equivalent to the frequency 1 Wigner D matrices
                irrep = _build_irrep(l)
                character = _build_character(l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 3, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character)
            elif l == -1 or l == 2:
    
                irrep = _build_octa_irrep(self, l)
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, irrep[self.identity].shape[0], 'R',
                                                             supported_nonlinearities=supported_nonlinearities)

            elif l == 3:
                irrep = _build_octa_irrep(self, l)
                supported_nonlinearities = ['norm', 'gated', 'concatenated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, irrep[self.identity].shape[0], 'R',
                                                             supported_nonlinearities=supported_nonlinearities)

            else:
                raise ValueError()

        return self._irreps[id]

    _cached_group_instance = None

    @classmethod
    def _generator(cls) -> 'Octahedral':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = Octahedral()
    
        return cls._cached_group_instance


def _is_axis_aligned(v: np.ndarray, n: int, verbose: bool = False, ATOL=1e-7, RTOL = 1e-5) -> bool:

    norm = np.linalg.norm(v)
    v = v / norm

    if n == 2:
        # rotation of order 2

        # the rotation axis need to be aligned with one of the axes X, Y, Z or to
        # the bisector of a pair of axes XY, XZ, YZ
        # There are in total 6 + 12 possible vectors

        # remove sign ambiguity
        v = np.abs(v)

        axes = np.eye(3)
        bisectors = np.array([
            [1., 0., 1.],
            [1., 1., 0.],
            [0., 1., 1.],
        ]) / np.sqrt(2)

        ans = (
                # axes aligned
                np.allclose(v, axes[0], atol=ATOL, rtol=RTOL)
                or np.allclose(v, axes[1], atol=ATOL, rtol=RTOL)
                or np.allclose(v, axes[2], atol=ATOL, rtol=RTOL)
                # bisectors aligned
                or np.allclose(v, bisectors[0], atol=ATOL, rtol=RTOL)
                or np.allclose(v, bisectors[1], atol=ATOL, rtol=RTOL)
                or np.allclose(v, bisectors[2], atol=ATOL, rtol=RTOL)
        )

        if not ans and verbose:
            print(f'Rotation by a multiple of 2pi/{n} not aligned with a {n}-fold rotational axis of the Octahedron.')

        return ans

    elif n == 4:
        # rotation of order 4

        # the rotation axis need to be aligned with one of the axes X, Y, Z
        # There are in total 6 possible vectors

        # remove sign ambiguity
        v = np.abs(v)

        axes = np.eye(3)

        ans = (
               np.allclose(v, axes[0], atol=ATOL, rtol=RTOL)
            or np.allclose(v, axes[1], atol=ATOL, rtol=RTOL)
            or np.allclose(v, axes[2], atol=ATOL, rtol=RTOL)
        )

        if not ans and verbose:
            print(f'Rotation by a multiple of 2pi/{n} not aligned with a {n}-fold rotational axis of the Octahedron.')

        return ans

    elif n == 3:
        # rotation or order 3

        # the rotation axis need to pass through one of the vertices of the cube
        # There are in total 8 possible vectors

        # remove sign ambiguity
        v = np.abs(v)

        # since the vector is normalized, `v` should now be `(1, 1, 1)^T * 1/sqrt(3)`

        ans = np.allclose(v, 1./np.sqrt(3), atol=ATOL, rtol=RTOL)

        if not ans and verbose:
            print(f'Rotation by a multiple of 2pi/{n} not aligned with a {n}-fold rotational axis of the Octahedron.')

        return ans

    else:
        raise ValueError('The rotation order must be one of {2, 3, 4}.')


#############################################
# SUBGROUPS MAPS
#############################################

# C_N #####################################

def octa_to_cn(adj: GroupElement, cn: escnn.group.CyclicGroup, axis: np.ndarray):
    assert isinstance(adj.group, Octahedral)

    assert axis.shape == (3,)
    assert np.isclose(np.linalg.norm(axis), 1.)

    assert cn.order() in [2, 3, 4]

    assert _is_axis_aligned(axis, cn.order())

    def _map(e: GroupElement, cn=cn, adj=adj, axis=axis):
        octa = adj.group
        assert e.group == octa

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


def cn_to_octa(adj: GroupElement, cn: escnn.group.CyclicGroup, axis: np.ndarray):
    assert isinstance(adj.group, Octahedral)

    assert axis.shape == (3,)
    assert np.isclose(np.linalg.norm(axis), 1.)

    assert cn.order() in [2, 3, 4]

    assert _is_axis_aligned(axis, cn.order())

    def _map(e: GroupElement, cn=cn, adj=adj, axis=axis):
        assert e.group == cn
        octa = adj.group

        theta_2 = e.to('radians') / 2.

        q = np.empty(4)
        q[:3] = axis * np.sin(theta_2)
        q[-1] = np.cos(theta_2)

        return (~adj) @ octa.element(q, 'Q') @ adj

    return _map


#############################################
# Generate irreps
#############################################

from joblib import Memory
from escnn.group import __cache_path__

cache = Memory(__cache_path__, verbose=2)


def _build_octa_irrep(octa: Octahedral, l: int):
    # See `_build_ico_irrep()` for an explanation of why this function is split 
    irreps = _build_octa_irrep_picklable(octa, l)
    return {
            octa.element(g, param): v
            for g, param, v in irreps
    }


@cache.cache(ignore=['octa'])
def _build_octa_irrep_picklable(octa: Octahedral, l: int):
    
    if l == -1:
        
        # matrix coefficients from https://arxiv.org/pdf/1110.6376.pdf
        
        # the matrix coefficients there are expressed wrt a different set of generators
        # we fist build this set of generators
        
        r3 = octa.generators[0]
        r = r3 @ r3 @ r3
        
        k = octa.elements[0]
        s = octa.generators[1]
        t = ~s @ k @ s @ r
        
        # Representation of `t`
        rho_t = np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        
        # Representation of `k`
        rho_k = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
        ])
        
        # Representation of `s`
        rho_s = np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
        ])
        
        #  https://arxiv.org/pdf/1110.6376.pdf defines the irrep `l = 1` (denoted by 3 there) as our
        #  `standard_representation`, which is expressed on a different basis than the Wigner D matrix with l=1.
        # Since `l=-1` (their 3') is defined as the tensor product between `l=1` and `l=3` (their 1')
        # we apply the inverse change of basis used in `standard_representation` to ensure that
        # `-1 = 1 \tensor 3` for us as well
        
        change_of_basis = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        rho_t = change_of_basis.T @ rho_t @ change_of_basis
        rho_k = change_of_basis.T @ rho_k @ change_of_basis
        rho_s = change_of_basis.T @ rho_s @ change_of_basis
        
    elif l == 2:

        # matrix coefficients from https://arxiv.org/pdf/1110.6376.pdf

        # the matrix coefficients there are expressed wrt a different set of generators
        # we fist build this set of generators

        r3 = octa.generators[0]
        r = r3 @ r3 @ r3

        k = octa.elements[0]
        s = octa.generators[1]
        t = ~s @ k @ s @ r

        # Representation of `t`
        rho_t = np.array([
            [0., 1.],
            [1., 0.],
        ])

        # Representation of `k`
        rho_k = np.array([
            [1., 0.],
            [0., 1.],
        ])

        # Representation of `s`
        rho_s = 0.5 * np.array([
            [-1., -np.sqrt(3)],
            [np.sqrt(3), -1.],
        ])

    elif l == 3:

        # matrix coefficients from https://arxiv.org/pdf/1110.6376.pdf

        # the matrix coefficients there are expressed wrt a different set of generators
        # we fist build this set of generators

        r3 = octa.generators[0]
        r = r3 @ r3 @ r3

        k = octa.elements[0]
        s = octa.generators[1]
        t = ~s @ k @ s @ r

        # Representation of `t`
        rho_t = np.array([[-1.]])

        # Representation of `k`
        rho_k = np.array([[1.]])

        # Representation of `s`
        rho_s = np.array([[1.]])

    else:
        raise ValueError()

    generators = [
        (t, rho_t),
        (k, rho_k),
        (s, rho_s),
    ]
    
    irreps = generate_irrep_matrices_from_generators(octa, generators)
    return [
        (k.value, k.param, v)
        for k, v in irreps.items()
    ]

