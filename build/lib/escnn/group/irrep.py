from __future__ import annotations

import escnn.group
from escnn.group import Representation, GroupElement, Group
from escnn.group._numerical import decompose_representation_finitegroup
from escnn.group._numerical import decompose_representation_general

from typing import Callable, Any, List, Union, Dict, Tuple, Type

import numpy as np

__all__ = [
    "IrreducibleRepresentation",
    "build_irrep_from_generators",
    "generate_irrep_matrices_from_generators",
    "restrict_irrep"
]


class IrreducibleRepresentation(Representation):
    
    def __init__(self,
                 group: escnn.group.Group,
                 id: Tuple,
                 name: str,
                 representation: Union[Dict[escnn.group.GroupElement, np.ndarray], Callable[[Any], np.ndarray]],
                 size: int,
                 type: str,
                 supported_nonlinearities: List[str],
                 character: Union[Dict[escnn.group.GroupElement, float], Callable[[Any], float]] = None,
                 **kwargs
                 ):
        """
        Describes an "*irreducible representation*" (*irrep*).
        It is a subclass of a :class:`~escnn.group.Representation`.
        
        Irreducible representations are the building blocks into which any other representation decomposes under a
        change of basis.
        Indeed, any :class:`~escnn.group.Representation` is internally decomposed into a direct sum of irreps.
        
        Args:
            group (Group): the group which is being represented
            id (tuple): args to generate this irrep using ``group.irrep(*id)``
            name (str): an identification name for this representation
            representation (dict or callable): a callable implementing this representation or a dict mapping
                    each group element to its representation.
            size (int): the size of the vector space where this representation is defined (i.e. the size of the matrices)
            type (str): type of the irrep. It needs to be one of `R`, `C` or `H`, which represent respectively
                        real, complex and quaternionic types.
                        NOTE: this parameter substitutes the old `sum_of_squares_constituents` from *e2cnn*.
            supported_nonlinearities (list): list of nonlinearitiy types supported by this representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dict mapping each group element to its character.
            **kwargs: custom attributes the user can set and, then, access from the dictionary
                    in :attr:`escnn.group.Representation.attributes`
        
        Attributes:
            ~.id (tuple): tuple which identifies this irrep; it can be used to generate this irrep as ``group.irrep(*id)``
            ~.sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                    irreducible constituents of the character of this representation over a non-splitting field (see
                    `Character Orthogonality Theorem <https://groupprops.subwiki.org/wiki/Character_orthogonality_theorem#Statement_over_general_fields_in_terms_of_inner_product_of_class_functions>`_
                    over general fields).
                    This attribute is fully determined by the irrep's `type` as:
                    
                    +----------+---------------------------------+
                    |  `type`  |  `sum_of_squares_constituents`  |
                    +==========+=================================+
                    |  'R'     |    `1`                          |
                    +----------+---------------------------------+
                    |  'C'     |    `2`                          |
                    +----------+---------------------------------+
                    |  'H'     |    `4`                          |
                    +----------+---------------------------------+
            
        """
        
        assert type in {'R', 'C', 'H'}

        if type == 'C':
            assert size % 2 == 0
        elif type == 'H':
            assert size % 4 == 0

        super(IrreducibleRepresentation, self).__init__(group,
                                                        name,
                                                        [id],
                                                        np.eye(size),
                                                        supported_nonlinearities,
                                                        representation=representation,
                                                        character=character,
                                                        **kwargs)
        assert isinstance(id, tuple)
        self.id: tuple = id
        self.irreducible = True
        self.type = type
        
        if self.type == 'R':
            self.sum_of_squares_constituents = 1
        elif self.type == 'C':
            self.sum_of_squares_constituents = 2
        elif self.type == 'H':
            self.sum_of_squares_constituents = 4
        else:
            raise ValueError()

    def endomorphism_basis(self) -> np.ndarray:
        if self.type == 'R':
            return np.eye(self.size).reshape(1, self.size, self.size)
        elif self.type == 'C':
            basis = np.stack([
                np.eye(2),
                np.diag([1., -1.])[::-1]
            ], axis=0)
            return np.kron(basis, np.eye(self.size // 2))
        elif self.type == 'H':
            basis = np.stack([
                np.eye(4),
                np.diag([1., -1., 1., -1.])[::-1],
                np.array([
                    [ 0.,  0., -1.,  0.],
                    [ 0.,  0.,  0., -1.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0.],
                ]),
                np.array([
                    [0., -1., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., -1., 0.],
                ]),
            ], axis=0)
            return np.kron(basis, np.eye(self.size // 4))
        else:
            raise ValueError()


def build_irrep_from_generators(
        group: escnn.group.Group,
        generators: List[Tuple[escnn.group.GroupElement, np.ndarray]],
        id: Tuple,
        name: str,
        type: str,
        supported_nonlinearities: List[str],
        **kwargs
) -> IrreducibleRepresentation:
    
    rho = generate_irrep_matrices_from_generators(group, generators)
    d = generators[0][1].shape[0]

    return IrreducibleRepresentation(
        group,
        id,
        name,
        rho,
        d,
        type,
        supported_nonlinearities,
        **kwargs
    )


def generate_irrep_matrices_from_generators(
        group: escnn.group.Group,
        generators: List[Tuple[escnn.group.GroupElement, np.ndarray]],
) -> List[np.ndarray]:
    assert group.order() > 0
    
    d = generators[0][1].shape[0]
    
    for g, rho_g in generators:
        assert group == g.group
        assert rho_g.shape == (d, d)
    
    elements = set()
    added = set()
    
    identity = group.identity
    
    added.add(identity)
    elements.add(identity)
    
    rho = {
        g: rho_g
        for g, rho_g in generators
    }
    rho[identity] = np.eye(d)
    
    generators = [g for g, _ in generators]
    
    while len(added) > 0:
        new = set()
        for g in generators:
            for e in added:
                if g @ e not in rho:
                    rho[g @ e] = rho[g] @ rho[e]
                
                if ~g @ e not in rho:
                    rho[~g @ e] = rho[g].T @ rho[e]
                
                new |= {g @ e, ~g @ e}
        added = new - elements
        elements |= added
    
    assert len(elements) == group.order(), 'Error! The set of generators passed does not generate the whole group'
    
    for a in elements:
        assert ~a in elements
        
        assert np.allclose(
            rho[~a],
            rho[a].T
        )
        
        for b in elements:
            assert a @ b in elements
            
            assert np.allclose(
                rho[a] @ rho[b],
                rho[a @ b]
            )
    
    return rho


from joblib import Memory
# import os
# cache = Memory(os.path.join(os.path.dirname(__file__), '_jl_restricted_irreps'), verbose=2)

from escnn.group import __cache_path__
cache = Memory(__cache_path__, verbose=0)


@cache.cache
def _restrict_irrep(irrep_id: Tuple, id, group_class: str, **group_keys) -> Tuple[np.matrix, List[Tuple[Tuple, int]]]:
    
    group = escnn.group.groups_dict[group_class]._generator(**group_keys)
    
    irrep = group.irrep(*irrep_id)

    id = irrep.group._decode_subgroup_id_pickleable(id)

    subgroup, parent, child = group.subgroup(id)
    
    if subgroup.order() == 1:
        # if the subgroup is the trivial group, just return the identity cob and the list of trivial reprs
        return np.eye(irrep.size), [subgroup.trivial_representation.id]*irrep.size
    
    if subgroup.order() > 1:
        # if it is a finite group, we can sample all the element and use the precise method based on Character theory

        representation = {
            g: irrep(parent(g)) for g in subgroup.elements
        }

        # to solve the Sylvester equation and find the change of basis matrix, it is sufficient to sample
        # the generators of the subgroup
        change_of_basis, multiplicities = decompose_representation_finitegroup(
            representation,
            subgroup,
        )

    else:
        # if the group is not finite, we rely on the numerical method which is based on some samples of the group

        representation = lambda g: irrep(parent(g))

        change_of_basis, multiplicities = decompose_representation_general(
            representation,
            subgroup,
        )

    irreps = []
    
    for irr, m in multiplicities:
        irreps += [irr]*m
    
    return change_of_basis, irreps


def restrict_irrep(irrep: IrreducibleRepresentation, id) -> Tuple[np.matrix, List[Tuple[str, int]]]:
    r"""
        Restrict the input `irrep` to the subgroup identified by `id`.
    """
    group_keys = irrep.group._keys

    id = irrep.group._encode_subgroup_id_pickleable(id)

    return _restrict_irrep(irrep.id, id, irrep.group.__class__.__name__, **group_keys)
