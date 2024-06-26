
from __future__ import annotations

from typing import Tuple, Callable, Iterable, List, Any, Dict

import escnn.group

from escnn.group import Group, GroupElement, IrreducibleRepresentation, DirectProductGroup
from escnn.group.irrep import restrict_irrep

import numpy as np
import itertools
import re

__all__ = [
    'DoubleGroup',
    'double_group'
]


class DoubleGroup(DirectProductGroup):

    def __init__(self, G: str, name: str = None, **group_keys):
        r"""

        Class defining the direct product of a group with itself.
        
        This is a special case (a subclass) of :class:`~escnn.group.DirectProductGroup`.
        :class:`~escnn.group.DirectProductGroup` can not automatically support all possible subgroups and, in particular,
        it does not support the diagonal subgroup of :math:`G \times G` which is isomorphic to :math:`G`.
        This subclass supports this special subgroup, which is identified by the id `"diagonal"`.
        See also :meth:`~escnn.group.DoubleGroup.subgroup_diagonal_id`.
        
        .. warning::
            This class should not be directly instantiated to ensure caching is performed correclty.
            You should instead use the function :func:`~escnn.group.double_group`.

        .. warning::
            This class does not support all possible subgroups of the direct product!
            If :math:`G = G_1 \times G_1`, only the diagonal subgroup isomorphic to :math:`G_1` and the subgroups of
            the form :math:`H = H_1 \times H_2` with :math:`H_1 < G_1` and :math:`H_2 < G_2` are supported.

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
    
        assert all(k.startswith('G_') for k in group_keys.keys())
    
        keys1 = {
            'G1_' + k[2:]: v for k, v in group_keys.items()
        }
        keys2 = {
            'G2_' + k[2:]: v for k, v in group_keys.items()
        }
        super(DoubleGroup, self).__init__(G, G, name, **keys1, **keys2)

    @property
    def _keys(self) -> Dict[str, Any]:
        keys = dict()
        keys['G'] = self.G1.__class__.__name__

        if not self._defaulf_name:
            keys['name'] = self.name

        keys.update({
            'G_' + k: v
            for k, v in self.G1._keys.items()
        })
        return keys

    @property
    def subgroup_diagonal_id(self):
        r"""
            The subgroup id associated with the diagonal group.
            This is the subgroup containing elements in the form :math:`(g, g)` for :math:`g \in G` and is isomorphic
            to :math:`G` itself.
            The id can be used in the method :meth:`~escnn.group.Group.subgroup` to generate the subgroup.
        """
        return 'diagonal'

    def _process_subgroup_id(self, id):
        if id == 'diagonal':
            return id
        else:
            return super(DoubleGroup, self)._process_subgroup_id(id)

    def _combine_subgroups(self, sg_id1, sg_id2):
        raise NotImplementedError

    def _subgroup(self, id) -> Tuple[
        escnn.group.Group,
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement],
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement]
    ]:
        if id == 'diagonal':
            return self.G1, inclusion(self), restriction(self)

        else:
            return super(DoubleGroup, self)._subgroup(id)

    _cached_group_instance = dict()

    @classmethod
    def _generator(cls, G: str, **group_keys) -> 'DirectProductGroup':

        key = {
            'G': G,
        }
        key.update(**group_keys)

        key = tuple(sorted(key.items()))

        if key not in cls._cached_group_instance:
            cls._cached_group_instance[key] = DoubleGroup(G, **group_keys)

        cls._cached_group_instance[key]._build_representations()

        return cls._cached_group_instance[key]


def restriction(G: DoubleGroup):
    def _map(e: GroupElement, G=G):
        assert e.group == G

        e1, e2 = G.split_element(e)

        if e1 == e2:
            return e1
        else:
            return None
    return _map


def inclusion(G: DoubleGroup):
    def _map(e: GroupElement, G=G):
        assert e.group == G.G1

        return G.element(
            (e.value, e.value),
            param=f'[{e.param} | {e.param}]'
        )

    return _map


def double_group(G: Group, name: str = None):
    r'''

    Generates the direct product of the input group `G` with itself.

    Args:
        G (Group): the group
        name (str, optional): name assigned to the resulting group

    Returns:
        an instance of :class:`~escnn.group.DoubleGroup`

    '''
    
    group_keys = {
        'G_' + k: v
        for k, v in G._keys.items()
    }
    return DoubleGroup._generator(
        G.__class__.__name__,
        name=name,
        **group_keys
    )

