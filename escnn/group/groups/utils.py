from __future__ import annotations

import numpy as np

import escnn.group
from escnn.group import Group, GroupElement
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation


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
