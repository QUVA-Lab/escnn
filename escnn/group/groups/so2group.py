from __future__ import annotations

import escnn.group
from escnn.group import Group, GroupElement
from escnn.group import IrreducibleRepresentation, Representation, directsum
from escnn.group import utils

from .utils import build_identity_map

import numpy as np

from typing import Tuple, Callable, Iterable, List, Dict, Any


__all__ = ["SO2"]


class SO2(Group):
    
    PARAM = 'radians'
    PARAMETRIZATIONS = [
        'radians',      # real in 0., 2pi/N, ... i*2pi/N, ...
        # 'C',          # point in the unit circle (i.e. cos and sin of 'radians')
        'MAT',  # 2x2 rotation matrix
    ]

    def __init__(self, maximum_frequency: int = 6):
        r"""
        Build an instance of the special orthogonal group :math:`SO(2)` which contains continuous planar rotations.
        
        A group element is a rotation :math:`r_\theta` of :math:`\theta \in [0, 2\pi)`, with group law
        :math:`r_\alpha \cdot r_\beta = r_{\alpha + \beta}`.
        
        Elements are implemented as floating point numbers :math:`\theta \in [0, 2\pi)`.
        
        .. note ::
            Since the group has infinitely many irreducible representations, it is not possible to build all of them.
            Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
            the maximum frequency of the irreps to build.
            New irreps (associated to higher frequencies) can be manually created by calling the method
            :meth:`~escnn.group.SO2.irrep` (see the method's documentation).
        
        
        Subgroup Structure.
        
        A subgroup of :math:`SO(2)` is a cyclic group :math:`C_M` with order :math:`M` which is generated
        by :math:`r_{\frac{2\pi}{M}}`.
        
        The ``id`` which identifies a subgroup is the integer :math:`M` (the order of the subgroup).
        
        
        Args:
            maximum_frequency (int, optional): the maximum frequency to consider when building the irreps of the group
        
        Attributes:
            ~.rotation_order (int): this is equal to ``-1``, which means the group contains an infinite number of rotations

        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(SO2, self).__init__("SO(2)", True, True)
        
        self._maximum_frequency = maximum_frequency
        self.rotation_order = -1

        self._identity = self.element(0.)
        
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
     
    # @property
    # def elements_names(self) -> List[str]:
    #     return None

    @property
    def _keys(self) -> Dict[str, Any]:
        return dict()

    @property
    def subgroup_trivial_id(self):
        return 1

    @property
    def subgroup_self_id(self):
        return -1

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################
    
    def _inverse(self, element: float, param: str = PARAM) -> float:
        r"""
        Return the inverse element of the input element: given an angle, the method returns its opposite

        Args:
            element (float): an angle :math:`\theta`

        Returns:
            its opposite :math:`-\theta \mod 2\pi`
        """

        element = self._change_param(element, p_from=param, p_to='radians')
        inverse = (-element) % (2*np.pi)
        return self._change_param(inverse, p_from='radians', p_to=param)

    def _combine(self, e1: float, e2: float, param: str = PARAM, param1: str = None, param2: str = None) -> float:
        r"""
        Return the sum of the two input elements: given two angles, the method returns their sum

        Args:
            e1 (float): an angle :math:`\alpha`
            e2 (float): another angle :math:`\beta`

        Returns:
            their sum :math:`(\alpha + \beta) \mod 2\pi`
            
        """
        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='radians')
        e2 = self._change_param(e2, p_from=param2, p_to='radians')
        product = (e1 + e2) % (2.*np.pi)
        return self._change_param(product, p_from='radians', p_to=param)

    def _equal(self, e1: float, e2: float, param: str = PARAM, param1: str = None, param2: str = None) -> bool:
        r"""
        
        Check if the two input values corresponds to the same element, i.e. the same angle.
        
        The method accounts for a small absolute and relative tollerance and for the cyclicity of the the angles,
        i.e. the fact that :math:`0 + \epsilon \simeq 2\pi - \epsilon` for small :math:`\epsilon`
        
        Args:
            e1 (float): an angle :math:`\alpha`
            e2 (float): another angle :math:`\beta`

        Returns:
            whether the two angles are the same, i.e. if :math:`\beta \simeq \alpha \mod 2\pi`

        """
        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='radians')
        e2 = self._change_param(e2, p_from=param2, p_to='radians')
        return utils.cycle_isclose(e1, e2, 2 * np.pi)

    def _hash_element(self, element, param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='radians')
        return hash(tuple(np.around(np.array([np.cos(element), np.sin(element)]), 5)))

    def _repr_element(self, element, param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='radians')
        return element.__repr__()

    def _is_element(self, element: float, param: str = PARAM, verbose: bool = False) -> bool:
        element = self._change_param(element, p_from=param, p_to='radians')
        return isinstance(element, float)

    def _change_param(self, element, p_from: str, p_to: str):
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS
        
        if p_from == 'MAT':
            assert isinstance(element, np.ndarray)
            assert element.shape == (2, 2)
            assert np.isclose(np.linalg.det(element), 1.)
            assert np.allclose(element @ element.T, np.eye(2))
            
            cos = (element[0, 0] + element[1, 1]) / 2.
            sin = (element[1, 0] - element[0, 1]) / 2.

            element = np.arctan2(sin, cos)
        elif p_from == 'radians':
            assert isinstance(element, float)
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_from))

        element = element % (2.*np.pi)

        if p_to == 'MAT':
    
            cos = np.cos(element)
            sin = np.sin(element)
            element = np.array(([
                [cos, -sin],
                [sin, cos],
            ]))

        elif p_to == 'radians':
            pass
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_to))

        return element

    ###########################################################################
    
    def sample(self) -> GroupElement:
        return self.element(np.random.random()*2*np.pi, param='radians')

    def grid(self, N: int, type: str = 'regular', seed: int = None) -> List[GroupElement]:
        r"""
            .. todo::
                Add documentation

        """

        if type == 'regular':
            grid = [i * 2*np.pi / N for i in range(N)]
        elif type == 'rand':
            if isinstance(seed, int):
                rng = np.random.RandomState(seed)
            elif seed is None:
                rng = np.random
            else:
                assert isinstance(seed, np.random.RandomState)
                rng = seed
            grid = [rng.random()*2*np.pi for i in range(N)]
        else:
            raise ValueError(f'Grid type {type} not recognized')

        return [
            self.element(g, param='radians')
            for g in grid
        ]

    def testing_elements(self, n=4*13) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        return iter([self.element(i * 2. * np.pi / n) for i in range(n)])

    def __eq__(self, other):
        if not isinstance(other, SO2):
            return False
        else:
            return self.name == other.name # and self._maximum_frequency == other._maximum_frequency

    def _subgroup(self, id: int) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        Restrict the current group to the cyclic subgroup :math:`C_M` with order :math:`M` which is generated
        by :math:`r_{\frac{2\pi}{M}}`.
        
        The method takes as input the integer :math:`M` identifying of the subgroup to build
        (the order of the subgroup).
        
        Args:
            id (int): the integer :math:`M` identifying the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
              
        """

        assert isinstance(id, int)

        order = id
    
        # Build the subgroup

        if id > 0:
            # take the elements of the group generated by "2pi/order"
            sg = escnn.group.cyclic_group(order)
            # parent_mapping = lambda e, order=order: self.element(e._element * 2 * np.pi / order)
            parent_mapping = _build_parent_map(self, order)
            # child_mapping = lambda e, order=order, sg=sg: None if divmod(e.g, 2.*np.pi/order)[1] > 1e-15 else sg.element(int(round(e.g * order / (2.*np.pi))))
            # child_mapping = lambda e, order=order, sg=sg: None if not utils.cycle_isclose(e._element, 0., 2. * np.pi / order) else sg.element(int(round(e._element * order / (2. * np.pi))))
            child_mapping = _build_child_map(sg)
        elif id == -1:
            sg = self
            parent_mapping = build_identity_map()
            child_mapping = build_identity_map()
        else:
            raise ValueError()

        return sg, parent_mapping, child_mapping

    def _combine_subgroups(self, sg_id1, sg_id2):
    
        sg_id1 = self._process_subgroup_id(sg_id1)
        sg1, inclusion, restriction = self.subgroup(sg_id1)
        sg_id2 = sg1._process_subgroup_id(sg_id2)
    
        return sg_id2

    def _restrict_irrep(self, irrep: Tuple, id: int) -> Tuple[np.matrix, List[Tuple]]:
        r"""
        
        Restrict the input irrep to the subgroup :math:`C_M` with order "M".
        More precisely, it restricts to the subgroup generated by :math:`2 \pi /order`.
        
        The method takes as input the integer :math:`M` identifying of the subgroup to build (the order of the subgroup)

        Args:
            irrep (tuple): the identifier of the irrep to restrict
            id (int): the integer :math:`M` identifying the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """

        irr = self.irrep(*irrep)
    
        # Build the subgroup
        sg, _, _ = self.subgroup(id)
    
        order = id
        
        if id == -1:
            return irr.change_of_basis, irr.irreps
        else:
    
            change_of_basis = None
            irreps = []
            
            f = irr.attributes["frequency"] % order
        
            if f > order/2:
                f = order - f
                change_of_basis = np.array([[1, 0], [0, -1]])
            else:
                change_of_basis = np.eye(irr.size)
        
            r = (f,)
        
            irreps.append(r)
            if sg.irrep(*r).size < irr.size:
                irreps.append(r)
        
            return change_of_basis, irreps
    
    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
    
        # Build all the Irreducible Representations
    
        k = 0
    
        # add Trivial representation
        self.irrep(k)
    
        for k in range(self._maximum_frequency + 1):
            self.irrep(k)
    
        # Build all Representations
    
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

    def bl_regular_representation(self, L: int) -> Representation:
        r"""
        Band-Limited regular representation up to frequency ``L`` (included).
        
        Args:
            L(int): max frequency

        """
        name = f'regular_{L}'

        if name not in self._representations:
            irreps = []

            for l in range(L + 1):
                irreps += [self.irrep(l)]

            self._representations[name] = directsum(irreps, name=name)

        return self._representations[name]

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L, L
        return [(l,) for l in range(L+1)]

    @property
    def trivial_representation(self) -> Representation:
        return self.irrep(0)

    def standard_representation(self) -> Representation:
        r"""
        Standard representation of :math:`\SO2` as 2x2 rotation matrices

        This is equivalent to ``self.irrep(1)``.

        """
        return self.irrep(1)

    def irrep(self, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with rotational frequency :math:`k` of :math:`SO(2)`.
        Notice: the frequency has to be a non-negative integer.
        
        Args:
            k (int): the frequency of the irrep

        Returns:
            the corresponding irrep

        """
    
        assert k >= 0
    
        name = f"irrep_{k}"
        id = (k, )

        if id not in self._irreps:

            irrep = _build_irrep_so2(k)
            character = _build_char_so2(k)
    
            if k == 0:
                # Trivial representation
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              # trivial=True,
                                                              frequency=0
                                                              )
            else:

                # 2 dimensional Irreducible Representations
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2, 'C',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k)

        return self._irreps[id]

    def _clebsh_gordan_coeff(self, m, n, j) -> np.ndarray:
        m, = self.get_irrep_id(m)
        n, = self.get_irrep_id(n)
        j, = self.get_irrep_id(j)
        
        rho_m = self.irrep(m)
        rho_n = self.irrep(n)
        rho_j = self.irrep(j)
        
        if m == 0 or n == 0:
            if j == m+n:
                return np.eye(rho_j.size).reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            else:
                return np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))
        else:
            cg = np.array([
                [1., 0., 1., 0.],
                [0., 1., 0., 1.],
                [0., -1., 0., 1.],
                [1., 0., -1., 0.],
            ]) / np.sqrt(2)
            if j == m + n:
                cg = cg[:, 2:]
            elif j == m - n:
                cg = cg[:, :2]
            elif j == n - m:
                cg = cg[:, :2]
                cg[:, 1] *= -1
            else:
                cg = np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))

            return cg.reshape(rho_n.size, rho_m.size, -1, rho_j.size).transpose(1, 0, 2, 3)

    def _tensor_product_irreps(self, J: int, l: int) -> List[Tuple[Tuple, int]]:
        J, = self.get_irrep_id(J)
        l, = self.get_irrep_id(l)
        
        if J == 0:
            return [
                ((l,), 1)
            ]
        elif l == 0:
            return [
                ((J,), 1)
            ]
        elif l == J:
            return [
                ((0,), 2),
                ((l+J,), 1),
            ]
        else:
            return [
                ((np.abs(l-J),), 1),
                ((l+J,), 1),
            ]

    _cached_group_instance = None

    @classmethod
    def _generator(cls, maximum_frequency: int = 3) -> 'SO2':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = SO2(maximum_frequency)
        elif cls._cached_group_instance._maximum_frequency < maximum_frequency:
            cls._cached_group_instance._maximum_frequency = maximum_frequency
            cls._cached_group_instance._build_representations()
    
        return cls._cached_group_instance


def _build_irrep_so2(k: int):
    assert k >= 0
    
    def irrep(e: GroupElement, k: int = k):
        if k == 0:
            # Trivial representation
            return np.eye(1)
        else:
            # 2 dimensional Irreducible Representations
            return utils.psi(e.to('radians'), k=k)
        
    return irrep


def _build_char_so2(k: int):
    assert k >= 0
    
    def character(e: GroupElement, k: int = k) -> float:
        if k == 0:
            # Trivial representation
            return 1.
        else:
            # 2 dimensional Irreducible Representations
            return 2 * np.cos(k * e.to('radians'))
        
    return character


def _build_parent_map(G: SO2, order: int):

    def parent_mapping(e: GroupElement, G: Group = G, order=order) -> GroupElement:
        return G.element(e.to('int') * 2 * np.pi / order)
    
    return parent_mapping


def _build_child_map(sg: 'CyclicGroup'):
    
    def child_mapping(e: GroupElement, sg: Group = sg) -> GroupElement:
        # return None if divmod(e.g, 2.*np.pi/order)[1] > 1e-15 else sg.element(int(round(e.g * order / (2.*np.pi))))
        radians = e.to('radians')
        if not utils.cycle_isclose(radians, 0., 2. * np.pi / sg.order()):
            return None
        else:
            return sg.element(int(round(radians * sg.order() / (2. * np.pi))))

    return child_mapping


