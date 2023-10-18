from __future__ import annotations

import numpy as np

import escnn.group
from escnn.group import Group, GroupElement
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation

class OrthoGroupEq:
    r"""
    Mixin class for orthogonal groups---O(2), O(3), SO(2), SO(3)---providing 
    an implementation of `__eq__()` that only considers if two groups are of 
    the same type.  In particular, the `maximum_frequency` attribute is 
    ignored.

    Because all group objects are singletons, you can still use `is` to check 
    if two orthogonal groups have the same type *and* maximum frequency.

    This mixin should come first in the inheritance list, because it is meant 
    to override methods provided by `Group`.
    """

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self.__class__ is other.__class__

def build_adjoint_map(G: Group, adj: GroupElement):
    
    _adj = ~adj
    
    def adj_map(e: GroupElement, adj=adj, _adj=_adj, G=G) -> GroupElement:
        assert e.group == G
        return adj @ e @ _adj
    
    return adj_map


def build_trivial_subgroup_maps(G: Group):

    C = escnn.group.cyclic_group(1)

    def parent_map(e: GroupElement, C=C, G=G) -> GroupElement:
        assert e.group == C
        return G.identity

    def child_map(e: GroupElement, C=C, G=G) -> GroupElement:
        assert e.group == G
        if e == G.identity:
            return C.identity
        else:
            return None

    return parent_map, child_map


def build_trivial_irrep():
    
    def trivial_irrep(element: GroupElement) -> np.ndarray:
        return np.eye(1)
    
    return trivial_irrep


def build_trivial_character():
    def trivial_character(element: GroupElement) -> float:
        return 1.
    
    return trivial_character


def build_identity_map():
    def identity(x):
        return x
    return identity
