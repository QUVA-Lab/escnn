
from __future__ import annotations

from typing import Tuple, Callable, Iterable, List, Any, Dict

import escnn.group

from escnn.group import Group, GroupElement, IrreducibleRepresentation
from escnn.group.irrep import restrict_irrep

import numpy as np
import itertools
import re

__all__ = [
    "DirectProductGroup",
    'direct_product'
]


param_split_regex = re.compile(r"\[(.+)\ \|\ (.+)\]")


def _split_param(param: str) -> Tuple[str, str]:
    if param is None:
        return None, None
    
    params = param_split_regex.match(param).groups()
    assert len(params) == 2
    return params


class DirectProductGroup(Group):

    def __init__(self, G1: str, G2: str, name: str = None, **groups_keys):
        r"""
        
        Class defining the direct product of two groups.
        
        .. warning::
            This class should not be directly instantiated to ensure caching is performed correclty.
            You should instead use the function :func:`~escnn.group.direct_product`.
        
        .. warning::
            This class does not support all possible subgroups of the direct product!
            For :math:`G = G_1 \times G_2`, only subgroups of the form :math:`H = H_1 \times H_2` with :math:`H_1 < G_1`
            and :math:`H_2 < G_2` are supported.
            
            A subgroup `id` is a tuple containing a pair `(id1, id2)`, where `id1` identifies a subgroup of :math:`G_1`
            while `id2` identifies a subgroup of :math:`G_2`.
            See the documentation of the two groups used for more details about their subgroup structures.
            
            In case either :math:`H_1` or :math:`H_2` is the trivial subgroup (containing only the identity),
            the subgroup returned is just an instance of the other one and not of
            :class:`~escnn.group.DirectProductGroup`.
            
        
        Args:
            G1 (Group): first group
            G2 (Group): second group
            name (str, optional): name assigned to the resulting group
            groups_keys: additional keywords argument used for identifying the groups and perform caching
            
        Attributes:
            ~.G1 (Group): the first group
            ~.G2 (Group): the second group

        """
        
        g1_keys = {
            k[3:]: v
            for k, v in groups_keys.items()
            if k[:3] == 'G1_'
        }
        g2_keys = {
            k[3:]: v
            for k, v in groups_keys.items()
            if k[:3] == 'G2_'
        }
        
        self._G1 = escnn.group.groups_dict[G1]._generator(**g1_keys)
        self._G2 = escnn.group.groups_dict[G2]._generator(**g2_keys)
        
        continuous = self.G1.continuous or self.G2.continuous
        abelian = self.G1.abelian and self.G2.abelian
        
        self._defaulf_name = name is None
        if name is None:
            name = self.G1.name + ' X ' + self.G2.name
        
        assert isinstance(name, str)
        
        super(DirectProductGroup, self).__init__(name, continuous, abelian)
        
        self._build_representations()

    @property
    def G1(self) -> Group:
        return self._G1
    
    @property
    def G2(self) -> Group:
        return self._G2

    @property
    def PARAM(self) -> str:
        return f"[{self.G1.PARAM} | {self.G2.PARAM}]"

    @property
    def PARAMETRIZATIONS(self) -> List[str]:
        params = []
        for p1 in self.G1.PARAMETRIZATIONS:
            for p2 in self.G2.PARAMETRIZATIONS:
                params.append(
                    f"[{p1} | {p2}]"
                )
        return params

    @property
    def identity(self) -> GroupElement:
        return self.pair_elements(self.G1.identity, self.G2.identity)

    @property
    def elements(self) -> List[GroupElement]:
        e1 = self.G1.elements
        e2 = self.G2.elements
        if e1 is None or e2 is None:
            return None
        else:
            elements = []
            for g1 in e1:
                for g2 in e2:
                    elements.append(
                        self.pair_elements(g1, g2)
                    )
            return elements

    @property
    def subgroup_trivial_id(self):
        r"""
            The subgroup id associated with the trivial subgroup containing only the identity element :math:`{e}`.
            The id can be used in the method :meth:`~escnn.group.Group.subgroup` to generate the subgroup.
        """
        return (self.G1.subgroup_trivial_id, self.G2.subgroup_trivial_id)
    
    @property
    def subgroup_self_id(self):
        r"""
            The subgroup id associated with the group itself.
            The id can be used in the method :meth:`~escnn.group.Group.subgroup` to generate the subgroup.
        """
        return (self.G1.subgroup_self_id, self.G2.subgroup_self_id)

    @property
    def subgroup1_id(self):
        r"""
            The subgroup id associated with :math:`G1`.
        """
        return (self.G1.subgroup_self_id, self.G2.subgroup_trivial_id)

    @property
    def subgroup2_id(self):
        r"""
            The subgroup id associated with :math:`G2`.
        """
        return (self.G1.subgroup_trivial_id, self.G2.subgroup_self_id)

    @property
    def _keys(self) -> Dict[str, Any]:
        keys = dict()
        keys['G1'] = self.G1.__class__.__name__
        keys['G2'] = self.G2.__class__.__name__
        if not self._defaulf_name:
            keys['name'] = self.name
            
        keys.update({
            'G1_' + k: v
            for k, v in self.G1._keys.items()
        })
        keys.update({
            'G2_' + k: v
            for k, v in self.G2._keys.items()
        })
        return keys

    @property
    def generators(self) -> List[GroupElement]:
        return [
            self.inclusion1(g) for g in self.G1.generators
        ] + [
            self.inclusion2(g) for g in self.G2.generators
        ]
    
    def inclusion1(self, g1: GroupElement):
        return self.pair_elements(g1, self.G2.identity)
    
    def inclusion2(self, g2: GroupElement):
        return self.pair_elements(self.G1.identity, g2)

    def pair_elements(self, g1: GroupElement, g2: GroupElement):
        assert g1.group == self.G1
        assert g2.group == self.G2
        return self.element((
            g1.to(self.G1.PARAM),
            g2.to(self.G2.PARAM),
            self.PARAM
        ))

    def split_element(self, g: GroupElement) -> Tuple[GroupElement, GroupElement]:
        assert g.group == self
        g1, g2 = g.to(self.PARAM)
        return (
            self.G1.element(g1),
            self.G2.element(g2)
        )

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################

    def _combine(self, e1, e2,
                 param: str = None,
                 param1: str = None,
                 param2: str = None
                 ):
        r"""

        Method that returns the combination of two group elements according to the *group law*.
        
        Args:
            e1: an element of the group
            e2: another element of the group
    
        Returns:
            the group element :math:`e_1 \cdot e_2`
            
        """
        if param is None:
            param = self.PARAM
            
        param_1, param_2 = _split_param(param)
        param1_1, param1_2 = _split_param(param1)
        param2_1, param2_2 = _split_param(param2)
        
        return (
            self.G1._combine(e1[0], e2[0], param=param_1, param1=param1_1, param2=param2_1),
            self.G2._combine(e1[1], e2[1], param=param_2, param1=param1_2, param2=param2_2),
        )

    def _inverse(self, element, param: str = None):
        r"""
        Method that returns the inverse in the group of the element given as input

        Args:
            element: an element of the group

        Returns:
            its inverse
        """
        if param is None:
            param = self.PARAM

        param_1, param_2 = _split_param(param)

        return (
            self.G1._inverse(element[0], param=param_1),
            self.G2._inverse(element[1], param=param_2)
        )

    def _is_element(self,
                    element,
                    param: str = None,
                    verbose: bool = False
                    ) -> bool:
        r"""
        Check whether the input is an element of this group or not.

        Args:
            element: input object to test

        Returns:
            if the input is an element of the group

        """
        if param is None:
            param = self.PARAM

        param_1, param_2 = _split_param(param)

        return (
                self.G1._is_element(element[0], param=param_1, verbose=verbose) and
                self.G2._is_element(element[1], param=param_2, verbose=verbose)
        )

    def _equal(self, e1, e2,
               param: str = None,
               param1: str = None,
               param2: str = None
               ) -> bool:
        r"""
        Method that checks whether the two inputs are the same element of the group.

        This is especially useful for continuous groups with periodicity; see for instance
        :meth:`escnn.group.SO2.equal`.

        Args:
            e1: an element of the group
            e2: another element of the group

        Returns:
            if they are equal

        """
        if param is None:
            param = self.PARAM

        param_1, param_2 = _split_param(param)
        param1_1, param1_2 = _split_param(param1)
        param2_1, param2_2 = _split_param(param2)

        return (
            self.G1._equal(e1[0], e2[0], param=param_1, param1=param1_1, param2=param2_1) and
            self.G2._equal(e1[1], e2[1], param=param_2, param1=param1_2, param2=param2_2)
        )

    def _change_param(self, element, p_from: str, p_to: str):
        p_from_1, p_from_2 = _split_param(p_from)
        p_to_1, p_to_2 = _split_param(p_to)
    
        return (
                self.G1._change_param(element[0], p_from=p_from_1, p_to=p_to_1),
                self.G2._change_param(element[1], p_from=p_from_2, p_to=p_to_2)
        )

    def _hash_element(self, element, param: str = None):
        r"""
        Method that returns a unique hash for a group element given in input

        Args:
            element: an element of the group

        Returns:
            a unique hash
        """
        if param is None:
            param = self.PARAM

        param_1, param_2 = _split_param(param)

        return (
            self.G1._hash_element(element[0], param=param_1) +
            self.G2._hash_element(element[1], param=param_2)
        )

    def _repr_element(self, element, param: str = None):
        r"""
        Method that returns a representative string for a group element given in input

        Args:
            element: an element of the group

        Returns:
            a unique hash
        """
        if param is None:
            param = self.PARAM

        param_1, param_2 = _split_param(param)

        return (
                self.G1._repr_element(element[0], param=param_1) +
                ' ; ' +
                self.G2._repr_element(element[1], param=param_2)
        )

    ###########################################################################

    def __eq__(self, other):
        if not isinstance(other, DirectProductGroup):
            return False
        else:
            return self.G1 == other.G1 and self.G2 == other.G2

    def sample(self):
        return self.element((
            self.G1.sample().to(self.G1.PARAM),
            self.G2.sample().to(self.G2.PARAM)
        ))

    def grid(self, type: str = None, N: int = None, **kwargs):
        r"""

            .. todo ::
                Write documentation

            if `type = "rand"`, generates `N` random samples otherwise split the keywords between the two groups
            (check if they start with `"G1_"` or `"G2_"`) and generate the direct product of the two sets


        """

        if type == 'rand':
            assert N is not None and N >= 0

            grid1 = self.G1.grid(type='rand', N=N)
            grid2 = self.G2.grid(type='rand', N=N)

            return [
                self.element((
                    g1.to(self.G1.PARAM),
                    g2.to(self.G2.PARAM)
                ))
                for g1, g2 in zip(grid1, grid2)
            ]
        else:

            kwargs1 = {
                k[(len('G1_')):] : v
                for k, v in kwargs.items()
                if k.startswith('G1_')
            }
            kwargs2 = {
                k[(len('G2_')):] : v
                for k, v in kwargs.items()
                if k.startswith('G2_')
            }

            grid1 = self.G1.grid(**kwargs1)
            grid2 = self.G2.grid(**kwargs2)

            return [
                self.element((
                    g1.to(self.G1.PARAM),
                    g2.to(self.G2.PARAM)
                ))
                for g1 in grid1
                for g2 in grid2
            ]

    def _process_subgroup_id(self, id):
        r'''
        
        .warning::
            This class does not support all possible subgroups of the direct product!
            For :math:`G = G_1 \times G_2`, only subgroups of the form :math:`H = H_1 \times H_2` with :math:`H_1 < G_1`
            and :math:`H_2 < G_2` are supported.

        '''
        assert isinstance(id, tuple) and len(id) == 2
        id = (
            self.G1._process_subgroup_id(id[0]),
            self.G2._process_subgroup_id(id[1])
        )
        return id

    def _subgroup(self, id) -> Tuple[
        escnn.group.Group,
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement],
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement]
    ]:
        r"""
        Restrict the current group to the subgroup identified by the input ``id``.
        
        .warning::
            This class does not support all possible subgroups of the direct product!
            For :math:`G = G_1 \times G_2`, only subgroups of the form :math:`H = H_1 \times H_2` with :math:`H_1 < G_1`
            and :math:`H_2 < G_2` are supported.

        Args:
            id: the identifier of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                -a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)

        """
        
        H1, parent1, child1 = self.G1.subgroup(id[0])
        H2, parent2, child2 = self.G2.subgroup(id[1])
        
        if H1.order() == 1:
            return H2, inclusion_right(self, H2, parent2), restrict_right(self, H2, child2)
        elif H2.order() == 1:
            return H1, inclusion_left(self, H1, parent1), restrict_left(self, H1, child1)
        else:
            H = direct_product(H1, H2)
            return H, inclusion(self, H, parent1, parent2), restrict(self, H, child2, child2)

    def _combine_subgroups(self, sg_id1, sg_id2):
        assert isinstance(sg_id1, tuple) and len(sg_id1) == 2
        assert isinstance(sg_id2, tuple) and len(sg_id2) == 2

        id = (
            self.G1._combine_subgroups(sg_id1[0], sg_id2[0]),
            self.G2._combine_subgroups(sg_id1[1], sg_id2[1]),
        )
        return id

    @property
    def trivial_representation(self) -> escnn.group.IrreducibleRepresentation:
        r"""
        Builds the trivial representation of the group.
        The trivial representation is a 1-dimensional representation which maps any element to 1,
        i.e. :math:`\forall g \in G,\ \rho(g) = 1`.
        
        Returns:
            the trivial representation of the group

        """
        return self.irrep(
            self.G1.trivial_representation.id,
            self.G2.trivial_representation.id
        )

    def irrep(self, id1: Tuple, id2: Tuple, i: int = 0) -> escnn.group.IrreducibleRepresentation:
        r"""

        Builds the irreducible representation (:class:`~escnn.group.IrreducibleRepresentation`) of the group which is
        specified by the input arguments.

        .. seealso ::

            Check the documentation of the specific group subclass used for more information on the valid ``id`` values.

        Args:
            *id: parameters identifying the specific irrep.

        Returns:
            the irrep built

        """
        
        psi1 = self.G1.irrep(*id1)
        psi2 = self.G2.irrep(*id2)
        
        if psi1.type == 'C' and psi2.type == 'C':
            assert i in [0, 1]
        else:
            assert i == 0

        name = f"irrep_[{id1},{id2}]({i})"
        id = (id1, id2, i)
        if id not in self._irreps:
    
            if psi1.type == 'R' or psi2.type == 'R':
                assert i == 0
                type = psi2.type if psi1.type == 'R' else psi1.type
                size = psi1.size * psi2.size
                irrep, character = tensor_product_irrep(self, psi1, psi2)

            elif psi1.type == 'C' or psi2.type == 'C':
                assert i in [0, 1]
                type = 'C'
                size = psi1.size * psi2.size // 2
                irrep, character = tensor_product_irrep_complex(self, psi1, psi2, i)
                
            else:
                assert i == 0
                type = 'H'
                size = psi1.size * psi2.size // 2
                irrep, character = tensor_product_irrep_complex(self, psi1, psi2, 0)

            supported_nonlinearities = ['norm', 'gated']
            self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, size, type,
                                                         supported_nonlinearities=supported_nonlinearities,
                                                         character=character,
                                                         id1=id1,
                                                         id2=id2,
                                                         i=i)
        
        return self._irreps[id]

    def _restrict_irrep(self, irrep: Tuple, id) -> Tuple[np.matrix, List[Tuple]]:

        # compute the restriction numerically
        # it should be possible to derive the decomposition analitically for the special case in which the subgroup is
        # itself the direct product of two subgroups of the two groups composing this group.
        # While in the complex field it is rather trivial, in the real one it is much more complicated and requires
        # us to assume that the irrep restriction implemented by both subgroups resemble the "realification" of the
        # irrep decomposition in the complex field.
        # it is much easier to solve this numerically

        irr = self.irrep(*irrep)

        change_of_basis, irreps = restrict_irrep(irr, id)
        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
    
        # Build all the Irreducible Representations
        for psi1 in self.G1.irreps():
            for psi2 in self.G2.irreps():
                self.irrep(psi1.id, psi2.id, 0)
                if psi1.type == 'C' and psi2.type == 'C':
                    self.irrep(psi1.id, psi2.id, 1)

        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

    def testing_elements(self) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        for g1, g2 in itertools.product(self.G1.testing_elements(), self.G2.testing_elements()):
            yield self.pair_elements(g1, g2)

    def _decode_subgroup_id_pickleable(self, id: Tuple) -> Tuple:
        if isinstance(id, tuple):
            if id[0] == 'G':
                id = self.element(id[1], id[2])
            elif id[0] == 'G1':
                id = self.G1.element(id[1], id[2])
            elif id[0] == 'G2':
                id = self.G2.element(id[1], id[2])
            else:
                id = list(id)
                for i in range(len(id)):
                    id[i] = self._decode_subgroup_id_pickleable(id[i])
                id = tuple(id)

        return id

    def _encode_subgroup_id_pickleable(self, id: Tuple) -> Tuple:

        if isinstance(id, GroupElement):
            G = id.group
            if G == self:
                id = 'G', id.value, id.param
            elif G == self.G1:
                id = 'G1', id.value, id.param
            elif G == self.G2:
                id = 'G2', id.value, id.param
            else:
                raise ValueError
        elif isinstance(id, tuple):
            id = list(id)
            for i in range(len(id)):
                id[i] = self._encode_subgroup_id_pickleable(id[i])
            id = tuple(id)
        return id

    _cached_group_instance = dict()

    @classmethod
    def _generator(cls, G1: str, G2: str, **group_keys) -> 'DirectProductGroup':
        
        key = {
            'G1': G1,
            'G2': G2,
        }
        key.update(**group_keys)
        
        key = tuple(sorted(key.items()))

        if key not in cls._cached_group_instance:
            cls._cached_group_instance[key] = DirectProductGroup(G1, G2, **group_keys)
            
        cls._cached_group_instance[key]._build_representations()

        return cls._cached_group_instance[key]


def direct_product(G1: Group, G2: Group, name: str = None):
    r'''
    
    Generates the direct product of the two input groups `G1` and `G2`.
    
    Args:
        G1 (Group): first group
        G2 (Group): second group
        name (str, optional): name assigned to the resulting group

    Returns:
        an instance of :class:`~escnn.group.DirectProductGroup`
 
    '''
    
    group_keys = dict()
    group_keys.update(**{
        'G1_' + k: v
        for k, v in G1._keys.items()
    })
    group_keys.update(**{
        'G2_' + k: v
        for k, v in G2._keys.items()
    })
    return DirectProductGroup._generator(
        G1.__class__.__name__,
        G2.__class__.__name__,
        name=name,
        **group_keys
    )


#############################################
# SUBGROUPS MAPS
#############################################

# Restrict G1 X G2 to a general H1 X H2
# with Hi < Gi

def restrict(G: DirectProductGroup, H: DirectProductGroup, restrict1, restrict2):
    
    def _map(e: GroupElement, G=G, H=H, restrict1=restrict1, restrict2=restrict2):
        assert e.group == G
        
        e1, e2 = G.split_element(e)
        
        _e1 = restrict1(e1)
        _e2 = restrict2(e2)
        
        if _e1 is None or _e2 is None:
            return None

        try:
            return H.element(
                (_e1.value, _e2.value),
                param=f'[{_e1.param} | {_e2.param}]'
            )
        except ValueError:
            return None
    
    return _map


def inclusion(G: DirectProductGroup, H: DirectProductGroup, inclusion1, inclusion2):
    
    def _map(e: GroupElement, G=G, H=H, inclusion1=inclusion1, inclusion2=inclusion2):
        assert e.group == H
        
        e1, e2 = H.split_element(e)
        
        _e1 = inclusion1(e1)
        _e2 = inclusion2(e2)
        
        return G.element(
            (_e1.value, _e2.value),
            param=f'[{_e1.param} | {_e2.param}]'
        )
    
    return _map


# Restrict G1 X G2 to H X {e}  = H
# with H < G1


def restrict_left(G: DirectProductGroup, H: Group, restrict1):
    def _map(e: GroupElement, G=G, H=H, restrict1=restrict1):
        assert e.group == G
        
        e1, e2 = G.split_element(e)
        
        if e2 != G.G2.identity:
            return None
        
        return restrict1(e1)
    
    return _map


def inclusion_left(G: DirectProductGroup, H: Group, inclusion1):
    def _map(e: GroupElement, G=G, H=H, inclusion1=inclusion1):
        assert e.group == H
        
        _e1 = inclusion1(e)
        
        return G.element((
                _e1.value,
                G.G2.identity.value
            ),
            param=f"[{_e1.param} | {G.G2.identity.param}]"
        )

    return _map


# Restrict G1 X G2 to {e} X H  = H
# with H < G2

def restrict_right(G: DirectProductGroup, H: Group, restrict2):
    def _map(e: GroupElement, G=G, H=H, restrict2=restrict2):
        assert e.group == G
        
        e1, e2 = G.split_element(e)
        
        if e1 != G.G1.identity:
            return None
        
        return restrict2(e2)
    
    return _map


def inclusion_right(G: DirectProductGroup, H: Group, inclusion2):
    def _map(e: GroupElement, G=G, H=H, inclusion2=inclusion2):
        assert e.group == H
        
        _e2 = inclusion2(e)

        return G.element((
                G.G1.identity.value,
                _e2.value
            ),
            param=f"[{G.G1.identity.param} | {_e2.param}]"
        )

    return _map


########################################################################################################################
# Generate Irreps
########################################################################################################################

def tensor_product_irrep(G: DirectProductGroup, psi1: IrreducibleRepresentation, psi2: IrreducibleRepresentation):
    
    # for a direct product group G = A x B, if
    # alpha_i is an irrep of A
    # beta_j is an irrep of B
    # and at least one of alpha_i and beta_j is a real-type irrep
    # G has an irrep of the form
    #       gamma_ij = alpha_i x beta_j
    # the type of gamma_ij is the type of the other irrep among alpha_i and beta_j
    
    # Therefore, gamma_ij can be simply computed as the tensor product of alpha_i and beta_j

    assert psi1.group == G.G1
    assert psi2.group == G.G2
    
    size = psi1.size * psi2.size
    size2_half = psi2.size // 2
    
    def tensor_product_left(g: GroupElement, G=G, psi1=psi1, psi2=psi2, size=size, size2_half=size2_half):
        assert g.group == G
        g1, g2 = G.split_element(g)
        psi_g1 = psi1(g1)
        psi_g2 = psi2(g2)

        real_2 = psi_g2[:size2_half, :size2_half]
        imag_2 = psi_g2[size2_half:, :size2_half]

        R = np.empty((size, size))
        real = np.kron(psi_g1, real_2)
        imag = np.kron(psi_g1, imag_2)

        R[:size // 2, :size // 2] = real
        R[size // 2:, size // 2:] = real
        R[size // 2:, :size // 2] = imag
        R[:size // 2:, size // 2:] = -imag

        return R

    def tensor_product_right(g: GroupElement, G=G, psi1=psi1, psi2=psi2):
        assert g.group == G
        g1, g2 = G.split_element(g)
        return np.kron(psi1(g1), psi2(g2))

    def character(g: GroupElement, G=G, psi1=psi1, psi2=psi2):
        assert g.group == G
        g1, g2 = G.split_element(g)
        return psi1.character(g1) * psi2.character(g2)

    if psi1.type == 'R' and psi2.type != 'R':
        return tensor_product_left, character
    elif psi2.type == 'R':
        return tensor_product_right, character
    else:
        raise ValueError()


def tensor_product_irrep_complex(G: DirectProductGroup, psi1: IrreducibleRepresentation, psi2: IrreducibleRepresentation, i: int):
    
    # assume a direct product group G = A x B, with
    # alpha_i is an irrep of A
    # beta_j is an irrep of B
    # neither alpha_i nor beta_j are real type-type irreps; then
    # alpha_i is isomorphic to Realification(a_i)
    # beta_j is isomorphic to Realification(b_i)
    # where a_i is a complex irrep of A, b_j is a complex irrep of B
    # and "Realification(X)" is a real matrix with the following block structure
    #  +---------+----------+
    #  | Real(X) | -Imag(X) |
    #  +---------+----------+
    #  | Imag(X) |  Real(X) |
    #  +---------+----------+

    # G has a real irrep of the form
    #       gamma_ij = Realification(a_i x b_j)
    # and one of the form
    #       delta_ij = Realification(a_i x ~b_j)
    # where ~ is the complex conjugate
    # In particular,
    #       gamma_ij + delta_ij = alpha_i  x beta_j
    # where = represent isomorphism
    # Additionally, if either alpha_i or beta_j have quaternionic-type
    # (or, equivalently, a_i = ~a_i or b_j = ~b_j)
    # then gamma_ij = delta_ij
    
    # Finally, if both alpha_i and beta_j are quaternionic type, then so is gamma_ij (or delta_ij)
    # Otherwise, gamma_ij (and delta_ij) has complex-type.
    
    # To compute gamma_ij (i=0) and delta_ij (i=1), note the following:
    # Real(a_i x b_j) = Real(a_i) x Real(b_j) - Imag(a_i) x Imag(b_j)
    # Imag(a_i x b_j) = Real(a_i) x Imag(b_j) + Real(a_i) x Imag(b_j)
    
    # We assume that both alpha_i and beta_j are expressed in a basis such that
    # alpha_i = Realification(a_i)
    # beta_j = Realification(b_i)
    # where = is an equality here
    # This allows us to extract the real and the imaginary parts of a_i and b_j from alpha_i and beta_j, to then compute
    # gamma_ij using the expressions above and the definition of Realification()
    # The computation of delta_ij is equivalent, up to a change of sign of Imag(b_j) in all expressions.

    assert psi1.group == G.G1
    assert psi2.group == G.G2

    size1 = psi1.size
    size2 = psi2.size
    size = psi1.size * psi2.size // 2
    
    assert i in [0, 1]
    
    def tensor_product(g: GroupElement, G=G, psi1=psi1, psi2=psi2, i=i, size=size, size1=size1, size2=size2):
        assert g.group == G
        
        g1, g2 = G.split_element(g)
        
        psi_g1 = psi1(g1)
        psi_g2 = psi2(g2)
        
        real_1 = psi_g1[:size1//2, :size1//2]
        imag_1 = psi_g1[size1//2:, :size1//2]
        
        real_2 = psi_g2[:size2//2, :size2//2]
        imag_2 = psi_g2[size2//2:, :size2//2]
        
        if i == 1:
            imag_2 *= -1
            
        R = np.empty((size, size))
        real = np.kron(real_1, real_2) - np.kron(imag_1, imag_2)
        imag = np.kron(real_1, imag_2) + np.kron(imag_1, real_2)
        
        R[:size//2, :size//2] = real
        R[size//2:, size//2:] = real
        R[size//2:, :size//2] = imag
        R[:size//2:, size//2:] = -imag
        
        return R

    def character(g: GroupElement, G=G, psi1=psi1, psi2=psi2, i=i):
        assert g.group == G
        g1, g2 = G.split_element(g)
        
        psi_g1 = psi1(g1)
        psi_g2 = psi2(g2)

        real_1 = psi_g1[:size1 // 2, :size1 // 2]
        imag_1 = psi_g1[size1 // 2:, :size1 // 2]

        real_2 = psi_g2[:size2 // 2, :size2 // 2]
        imag_2 = psi_g2[size2 // 2:, :size2 // 2]

        if i == 1:
            imag_2 *= -1

        return 2 * (
                np.trace(real_1) * np.trace(real_2)
              - np.trace(imag_1) * np.trace(imag_2)
        )

    return tensor_product, character




