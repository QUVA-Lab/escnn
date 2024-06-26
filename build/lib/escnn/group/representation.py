from __future__ import annotations

import escnn.group
from escnn.group import Group, GroupElement

from ._numerical import decompose_representation_finitegroup

from collections import defaultdict

from typing import Callable, Any, List, Tuple, Dict, Union, Set

import math
import numpy as np
import scipy as sp
from scipy import linalg, sparse
from scipy.sparse.csgraph import connected_components


__all__ = [
    "Representation",
    "build_from_discrete_group_representation",
    "directsum",
    "disentangle",
    "change_basis",
    "build_regular_representation",
    "build_quotient_representation",
    "build_induced_representation",
    "homomorphism_space"
]


class Representation:
    
    def __init__(self,
                 group: Group,
                 name: str,
                 irreps: List[Tuple],
                 change_of_basis: np.ndarray,
                 supported_nonlinearities: Union[List[str], Set[str]] = [],
                 representation: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]] = None,
                 character: Union[Dict[GroupElement, float], Callable[[Any], float]] = None,
                 change_of_basis_inv: np.ndarray = None,
                 **kwargs):
        r"""
        Class used to describe a group representation.
        
        A (real) representation :math:`\rho` of a group :math:`G` on a vector space :math:`V=\mathbb{R}^n` is a map
        (a *homomorphism*) from the group elements to invertible matrices of shape :math:`n \times n`, i.e.:
        
        .. math::
            \rho : G \to \GL{V}
            
        such that the group composition is modeled by a matrix multiplication:
        
        .. math::
            \rho(g_1 g_2) = \rho(g_1) \rho(g_2) \qquad  \forall \ g_1, g_2 \in G \ .
        
        Any representation (of a compact group) can be decomposed into the *direct sum* of smaller, irreducible
        representations (*irreps*) of the group up to a change of basis:
        
        .. math::
            \forall \ g \in G, \ \rho(g) = Q \left( \bigoplus\nolimits_{i \in I} \psi_i(g) \right) Q^{-1} \ .
        
        Here :math:`I` is an index set over the irreps of the group :math:`G` which are contained in the
        representation :math:`\rho`.
        
        This property enables one to study a representation by its irreps and it is used here to work with arbitrary
        representations.
        
        :attr:`escnn.group.Representation.change_of_basis` contains the change of basis matrix :math:`Q` while
        :attr:`escnn.group.Representation.irreps` is an ordered list containing the names of the irreps :math:`\psi_i`
        indexed by the index set :math:`I`.
        
        A ``Representation`` instance can be used to describe a feature field in a feature map.
        It is the building block to build the representation of a feature map, by "stacking" multiple representations
        (taking their *direct sum*).
        
        .. note ::
            In most of the cases, it should not be necessary to manually instantiate this class.
            Indeed, the user can build the most common representations or some custom representations via the following
            methods and functions:
            
            - :meth:`escnn.group.Group.irrep`,
            
            - :meth:`escnn.group.Group.regular_representation`,
            
            - :meth:`escnn.group.Group.quotient_representation`,
            
            - :meth:`escnn.group.Group.induced_representation`,
            
            - :meth:`escnn.group.Group.restrict_representation`,
            
            - :meth:`escnn.group.Representation.tensor`,
            
            - :func:`escnn.group.directsum`,
            
            - :func:`escnn.group.change_basis`
            
            Additionally, the direct sum of two representations can be quickly generated using the binary operator `+`,
            see :meth:`~escnn.group.Representation.__add__`.
            
            
        
        If ``representation`` is ``None`` (default), it is automatically inferred by evaluating each irrep, stacking
        their results (through direct sum) and then applying the changes of basis. **Warning**: the representation of an
        element is built at run-time every time this object is called (through ``__call__``), so this approach might
        become computationally expensive with large representations.
        
        Analogously, if the ``character`` of the representation is ``None`` (default), it is automatically inferred
        evaluating ``representation`` and computing its trace.
        
        .. note ::
            It is assumed that both ``representation`` and ``character`` expect input group elements in the default
            parametrization of ``group``, i.e. :attr:`escnn.group.Group.PARAM`.
        
        .. todo::
            improve the interface for "supported non-linearities" and write somewhere the available options
        
        Args:
            group (Group): the group to be represented.
            name (str): an identification name for this representation.
            irreps (list): a list of irreps' ids. Each id is a tuple representing one of the *irreps* of the
                    group (see :attr:`escnn.group.Group.irreps` and :attr:`escnn.group.IrreducibleRepresentation.id`).
            change_of_basis (~numpy.ndarray, optional): the matrix which transforms the direct sum of the irreps
                    in this representation. By default (`None`), the identity is assumed.
            supported_nonlinearities (list or set, optional): a list or set of nonlinearity types supported by this
                    representation.
            representation (dict or callable, optional): a callable implementing this representation or a dict mapping
                    each group element to its representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dict mapping each group element to its character.
            change_of_basis_inv (~numpy.ndarray, optional): the inverse of the ``change_of_basis`` matrix; if not
                    provided (``None``), it is computed from ``change_of_basis``.
            **kwargs: custom attributes the user can set and, then, access from the dictionary in
                    :attr:`escnn.group.Representation.attributes`
            
        Attributes:
            ~.group (Group): The group which is being represented.
            ~.name (str): A string identifying this representation.
            ~.size (int): Dimensionality of the vector space of this representation. In practice, this is the size of the
                matrices this representation maps the group elements to.
            ~.change_of_basis (~numpy.ndarray): Change of basis matrix for the irreps decomposition.
            ~.change_of_basis_inv (~numpy.ndarray): Inverse of the change of basis matrix for the irreps decomposition.
            ~.representation (callable): Method implementing the map from group elements to their representation matrix.
            ~.supported_nonlinearities (set): A set of strings identifying the non linearities types supported by this representation.
            ~.irreps (list): List of irreps into which this representation decomposes.
            ~.attributes (dict): Custom attributes set when creating the instance of this class.
            ~.irreducible (bool): Whether this is an irreducible representation or not (i.e. if it can't be decomposed into further invariant subspaces).

        
        """
        
        # can't have the name of an already existing representation
        assert name not in group.representations, f"A representation for {group.name} with name {name} already exists!"

        assert len(change_of_basis.shape) == 2 and change_of_basis.shape[0] == change_of_basis.shape[1], change_of_basis.shape
        assert change_of_basis.shape[0] > 0, change_of_basis.shape

        if change_of_basis_inv is None:
            change_of_basis_inv = sp.linalg.inv(change_of_basis)

        assert len(change_of_basis_inv.shape) == 2
        assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
        assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
        assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
        assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
        
        # Group: A string identifying this representation.
        self.group = group
        
        # str: The group this is a representation of.
        self.name = name
        
        # int: Dimensionality of the vector space of this representation.
        # In practice, this is the size of the matrices this representation maps the group elements to.
        self.size = change_of_basis.shape[0]

        if len(irreps) > 1:
            irreps_size = sum(self.group.irrep(*irr).size for irr in irreps)
            assert irreps_size == self.size, \
                f"Error! The size of the change of basis ({self.size}) does not match the sum of the sizes of the irreps ({irreps_size})."
        
        # np.ndarray: Change of basis matrix for the irreps decomposition.
        self.change_of_basis = change_of_basis

        # np.ndarray: Inverse of the change of basis matrix for the irreps decomposition.
        self.change_of_basis_inv = change_of_basis_inv
        
        # list(tuple): List of irreps this representation decomposes into
        self.irreps = irreps

        if representation is None:
            irreps_instances = [group.irrep(*n) for n in irreps]
            if np.allclose(change_of_basis, np.eye(self.size)):
                representation = direct_sum_factory(irreps_instances, None, None)
            else:
                representation = direct_sum_factory(irreps_instances, change_of_basis, change_of_basis_inv)
        elif isinstance(representation, dict):
            assert set(representation.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._stored_representations = representation
            representation = _build_representation_callable_from_dict(self._stored_representations)
            
        elif not callable(representation):
            raise ValueError('Error! "representation" is neither a dictionary nor callable')
        
        # Callable: Method implementing the map from group elements to matrix representations.
        self.representation = representation

        if isinstance(character, dict):
            
            assert set(character.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._characters = character

        elif callable(character):
            self._characters = character
        elif character is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            self._characters = None
        else:
            raise ValueError('Error! "character" must be a dictionary, a callable or "None"')

        # TODO - assert size matches size of the matrix returned by the callable
        
        self.supported_nonlinearities = set(supported_nonlinearities)
        
        # dict: Custom attributes set when creating the instance of this class
        self.attributes = kwargs

        # TODO : remove the condition of an identity change of basis?
        # bool: Whether this is an irreducible representation or not (i.e.: if it can't be decomposed further)
        self.irreducible = len(self.irreps) == 1 and np.allclose(self.change_of_basis, np.eye(self.change_of_basis.shape[0]))

        self._cached_restricted_representations = dict()
        
        self._irreps_multiplicities = defaultdict(int)
        
        for irrep in self.irreps:
            self._irreps_multiplicities[irrep] += 1
    
    def character(self, e: escnn.group.GroupElement) -> float:
        r"""

        The *character* of a finite-dimensional real representation is a function mapping a group element
        to the trace of its representation:

        .. math::

            \chi_\rho: G \to \mathbb{C}, \ \ g \mapsto \chi_\rho(g) := \operatorname{tr}(\rho(g))

        It is useful to perform the irreps decomposition of a representation using *Character Theory*.
        
        Args:
            e (GroupElement): an element of the group of this representation
        
        Returns:
            the character of the element
        
        """

        assert e.group == self.group, f"Error: the element {e} belongs to the group {e.group.name} and not {self.group.name}"
        
        if self._characters is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            repr = self(e)
            return np.trace(repr)
        elif isinstance(self._characters, dict):
            return self._characters[e]
        elif callable(self._characters):
            return self._characters(e)
        else:
            raise RuntimeError('Error! Character not recognized!')

    def is_trivial(self) -> bool:
        r"""
        
        Whether this representation is trivial or not.
        
        Returns:
            if the representation is trivial

        """
        return self.irreducible and self.group.trivial_representation.id == self.irreps[0]
    
    def contains_trivial(self) -> bool:
        r"""

        Whether this representation contains the trivial representation among its irreps.
        This is an alias for::
            
            any(self.group.irreps[irr].is_trivial() for irr in self.irreps)

        Returns:
           if it contains the trivial representation

        """
        for irrep in self.irreps:
            if self.group.irrep(*irrep).is_trivial():
                return True
        return False

    def restrict(self, id) -> escnn.group.Representation:
        r"""
        
        Restrict the current representation to the subgroup identified by ``id``.
        Check the documentation of the :meth:`~escnn.group.Group.subgroup` method in the underlying group to see the
        available subgroups and accepted ids.

        .. note ::
            This operation is cached.
            Multiple calls using the same subgroup ``id`` will return the same instance instead of computing a new
            restriction.

        Args:
            id: identifier of the subgroup

        Returns:
            the restricted representation
        """

        id = self.group._process_subgroup_id(id)

        if id not in self._cached_restricted_representations:
            self._cached_restricted_representations[id] = self.group.restrict_representation(id, self)

        return self._cached_restricted_representations[id]
    
    def __call__(self, element: GroupElement) -> np.ndarray:
        """
        An instance of this class can be called and it implements the mapping from an element of a group to its
        representation.
        
        This is equivalent to calling :meth:`escnn.group.Representation.representation`,
        though ``__call__`` first checks ``element`` is a valid input (i.e. an element of the group).
        It is recommended to use this call.

        Args:
            element (GroupElement): an element of the group

        Returns:
            A matrix representing the input element

        """
        
        assert element.group == self.group, f"Error: the element {element} belongs to the group {element.group.name} and not {self.group.name}"
        
        return self.representation(element)

    def __add__(self, other: escnn.group.Representation) -> escnn.group.Representation:
        r"""

        Compute the *direct sum* of two representations of a group.

        The two representations need to belong to the same group.

        Args:
            other (Representation): another representation

        Returns:
            the direct sum

        """
        
        return directsum([self, other])
    
    def __eq__(self, other: escnn.group.Representation) -> bool:
        if not isinstance(other, Representation):
            return False
        
        return (self.name == other.name
                and self.group == other.group
                and np.allclose(self.change_of_basis, other.change_of_basis)
                and self.irreps == other.irreps
                and self.supported_nonlinearities == other.supported_nonlinearities)
    
    def __repr__(self) -> str:
        return f"{self.group.name}|[{self.name}]:{self.size}"
    
    def __hash__(self):
        return hash(repr(self))
    
    def tensor(self, other: Representation) -> Representation:
        r"""
            Compute the tensor product representation of the current representation and the input representaton.
        """
        return self.group._tensor_product(self, other)
    
    def endomorphism_basis(self) -> np.ndarray:
        r"""
            Compute a basis for the space of endomorphism of this representation from the endomorphism spaces of the
            irreps contained in this representation.
            
            .. warning::
                This basis might become quite big for large representations and, therefore, this method might get
                computationally expensive.
                In practice, we recommend using this method only to generate the basis for small representations (e.g.
                for irreps).
                This basis is generated and stored more efficiently in the :doc:`escnn.nn` package, when parameterizing
                neural network modules; e.g. see :class:`~escnn.nn.BlocksBasisExpansion`.
            
        """

        dim = 0
        for in_irrep in self.irreps:
            in_irrep = self.group.irrep(*in_irrep)
            for out_irrep in self.irreps:
                out_irrep = self.group.irrep(*out_irrep)
                if in_irrep == out_irrep:
                    dim += in_irrep.sum_of_squares_constituents

        basis = np.zeros((dim, self.size, self.size))
    
        p = 0
        i = 0
        for in_irrep in self.irreps:
            in_irrep = self.group.irrep(*in_irrep)
            
            j = 0
            for out_irrep in self.irreps:
                out_irrep = self.group.irrep(*out_irrep)
                
                if out_irrep == in_irrep:
    
                    basis[p:p+in_irrep.sum_of_squares_constituents, j:j+in_irrep.size, i:i+in_irrep.size] = in_irrep.endomorphism_basis()
                    
                    p += in_irrep.sum_of_squares_constituents
                    
                j += out_irrep.size
            i += in_irrep.size

        basis = np.einsum('Mm,kmn,nN->kMN', self.change_of_basis, basis, self.change_of_basis_inv)
        return basis
    
    def multiplicity(self, irrep: tuple) -> int:
        r"""
            Returns the multiplicity of the ``irrep`` in the current representation.
        """
        return self._irreps_multiplicities[irrep]


def _build_representation_callable_from_dict(repr_dict: Dict[GroupElement, np.ndarray]):
    
    def representation(e: GroupElement, repr_dict: Dict = repr_dict) -> np.ndarray:
        return repr_dict[e]
    
    return representation


def _build_character_from_irreps_multiplicities(irreps: List[Tuple['IrreducibleRepresentation', int]]):
    
    def character(element: GroupElement, irreps=irreps) -> float:
        return sum([m * irrep.character(element) for (irrep, m) in irreps])

    return character


def build_from_discrete_group_representation(representation: Dict[GroupElement, np.array],
                                             name: str,
                                             group: escnn.group.Group,
                                             supported_nonlinearities: List[str]
                                             ) -> escnn.group.Representation:
    r"""
    Given a representation of a *finite* group as a dictionary of matrices, the method decomposes it as a direct sum
    of the irreps of the group and computes the change-of-basis matrix. Then, a new instance of
    :class:`~escnn.group.Representation` is built using the direct sum of irreps and the change-of-basis matrix as
    representation taking as input elements from the continuous parent group.
    
    For instance, given a regular representation of a cyclic group of order :math:`n` implemented as a
    list of permutations matrices, the method builds a representation of SO(2) whose values are these permutation
    matrices when evaluated to the angles corresponding to the elements of the cyclic group (i.e. any angle in the
    form :math:`k 2 \pi / n` with :math:`k` in :math:`[0, \dots, n-1]`)
    
    Args:
        representation (dict): a dictionary mapping an element of ``group`` to a numpy array (must be a squared matrix)
        name (str): an identification name of the representation
        group (Group): the group whose representation has to be built
        supported_nonlinearities (list): list of non linearities types supported by this representation.
    
    Returns:
        a new representation
        
    """

    assert group.order() > 0

    assert set(representation.keys()) == set(group.elements), "Error! Keys don't match group's elements"

    # decompose the representation
    cob, multiplicities = decompose_representation_finitegroup(representation, group)

    # build a list of representation instances with their multiplicities
    irreps_with_multiplicities = [(group.irrep(*id), m) for (id, m) in multiplicities]

    # build the character of this representation
    new_character = _build_character_from_irreps_multiplicities(irreps_with_multiplicities)

    irreps = []
    for irr, m in multiplicities:
       irreps += [irr] * m

    # build the representation object
    return Representation(group,
                          name,
                          irreps,
                          cob,
                          supported_nonlinearities,
                          representation=representation,
                          character=new_character)


# TODO when built from "directsum" we can "optimize" the representation by sorting the internal irreps
#      and permuting the change of basis matrix's columns accordingly. Could be useful when one uses GNORM batchnorm

def directsum(reprs: List[escnn.group.Representation],
              change_of_basis: np.ndarray = None,
              name: str = None
              ) -> escnn.group.Representation:
    r"""

    Compute the *direct sum* of a list of representations of a group.
    
    The direct sum of two representations is defined as follow:
    
    .. math::
        \rho_1(g) \oplus \rho_2(g) = \begin{bmatrix} \rho_1(g) & 0 \\ 0 & \rho_2(g) \end{bmatrix}
    
    This can be generalized to multiple representations as:
    
    .. math::
        \bigoplus_{i=1}^I \rho_i(g) = (\rho_1(g) \oplus (\rho_2(g) \oplus (\rho_3(g) \oplus \dots = \begin{bmatrix}
            \rho_1(g) &         0 &  \dots &      0 \\
                    0 & \rho_2(g) &  \dots & \vdots \\
               \vdots &    \vdots & \ddots &      0 \\
                    0 &     \dots &      0 & \rho_I(g) \\
        \end{bmatrix}
    

    .. note::
        All the input representations need to belong to the same group.

    Args:
        reprs (list): the list of representations to sum.
        change_of_basis (~numpy.ndarray, optional): an invertible square matrix to use as change of basis after computing the direct sum.
                By default (``None``), an identity matrix is used, such that only the direct sum is evaluated.
        name (str, optional): a name for the new representation.

    Returns:
        the direct sum

    """
    
    group = reprs[0].group
    for r in reprs:
        assert group == r.group
    
    if name is None:
        name = "_".join([f"[{r.name}]" for r in reprs])
    
    irreps = []
    for r in reprs:
        irreps += r.irreps
    
    size = sum([r.size for r in reprs])
    
    cob = np.zeros((size, size))
    cob_inv = np.zeros((size, size))
    p = 0
    for r in reprs:
        cob[p:p + r.size, p:p + r.size] = r.change_of_basis
        cob_inv[p:p + r.size, p:p + r.size] = r.change_of_basis_inv
        p += r.size

    if change_of_basis is not None:
        change_of_basis = change_of_basis @ cob
        change_of_basis_inv = sp.linalg.inv(change_of_basis)
    else:
        change_of_basis = cob
        change_of_basis_inv = cob_inv

    supported_nonlinearities = set.intersection(*[r.supported_nonlinearities for r in reprs])
    
    return Representation(group, name, irreps, change_of_basis, supported_nonlinearities, change_of_basis_inv=change_of_basis_inv)


def disentangle(repr: Representation) -> Tuple[np.ndarray, List[Representation]]:
    r"""
    
    If possible, disentangle the input representation by decomposing it into the direct sum of smaller representations
    and a change of basis acting as a permutation matrix.
    
    This method is useful to decompose a feature vector transforming with a complex representation into multiple feature
    vectors which transform independently with simpler representations.
    
    Note that this method only decomposes a representation by applying a permutation of axes.
    A more general decomposition using any invertible matrix is possible but is just a decomposition into
    irreducible representations (see :class:`~escnn.group.Representation`).
    However, since the choice of change of basis is relevant for the kind of operations which can be performed
    (e.g. non-linearities), it is often not desirable to discard any change of basis and completely disentangle a
    representation.
    
    Considering only change of basis matrices which are permutation matrices is sometimes more useful.
    For instance, the restriction of the regular representation of a group to a subgroup results in a representation containing
    multiple regular representations of the subgroup (one for each `coset`).
    However, depending on how the original representation is built, the restricted representation might not be
    block-diagonal and, so, the subgroup's regular representations might not be clearly separated.
    
    For example, this happens when restricting the regular representation of :math:`\D3`
    
    +-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |    :math:`g`                      |          :math:`e`                                                                                                                                                                       |          :math:`r`                                                                                                                                                                       |        :math:`r^2`                                                                                                                                                                       |          :math:`f`                                                                                                                                                                       |         :math:`rf`                                                                                                                                                                       |       :math:`r^2f`                                                                                                                                                                       |
    +===================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+
    |  :math:`\rho_\text{reg}^{\D3}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}` |
    +-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    
    to the reflection group :math:`\C2`
    
    +--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |    :math:`g`                                     |          :math:`e`                                                                                                                                                                       |          :math:`f`                                                                                                                                                                       |
    +==================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+
    |  :math:`\Res{\C2}{\D3} \rho_\text{reg}^{\D3}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}` |
    +--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    
    Indeed, in :math:`\Res{\C2}{\D3} \rho_\text{reg}^{\D3}(g)` the three pairs of entries (1, 4), (2, 6) and (3, 5)
    never mix with each other but only permute internally.
    Moreover, each pair transform according to the regular representation of :math:`\C2`.
    Through a permutation of the entries, it is possible to make all the entries belonging to the same pair contiguous.
    This this reshuffled representation is then equal to
    :math:`\rho_\text{reg}^{\C2} \oplus \rho_\text{reg}^{\C2} \oplus \rho_\text{reg}^{\C2}`.
    Though theoretically equivalent, an implementation of this representation where the entries are contiguous is
    convenient when computing functions over single fields like batch normalization.
    
    Notice that applying the change of basis returned to the input representation (e.g. through
    :func:`escnn.group.change_basis`) will result in a representation containing the direct sum of the representations
    in the list returned.
    
    .. seealso::
        :func:`~escnn.group.directsum`,
        :func:`~escnn.group.change_basis`
    
    Args:
        repr (Representation): the input representation to disentangle

    Returns:
        a tuple containing
        
            - **change of basis**: a (square) permutation matrix of the size of the input representation
            
            - **representation**: the list of representations the input one is decomposed into
        
    """
    
    rsize = repr.size
    nirreps = len(repr.irreps)
    
    cob_mask = np.isclose(repr.change_of_basis, np.zeros_like(repr.change_of_basis))
    cob_mask = np.invert(cob_mask)
    
    irreps = [repr.group.irrep(*irr) for irr in repr.irreps]
    irreps_pos = np.cumsum([0] + [irr.size for irr in irreps])
    
    masks = []
    i_pos = 0
    for i, irr in enumerate(irreps):
        mask = cob_mask[:, i_pos:i_pos + irr.size].any(axis=1)
        masks.append(mask)
        i_pos += irr.size
    
    cob_mask = np.array(masks, dtype=bool)
    
    graph = np.zeros((nirreps + rsize, nirreps + rsize), dtype=bool)
    graph[:nirreps, nirreps:] = cob_mask
    graph[nirreps:, :nirreps] = cob_mask.T
    
    n_blocks, labels = connected_components(graph, directed=False, return_labels=True)
    
    irreps_labels = labels[:nirreps]
    field_labels = labels[nirreps:]
    
    blocks = [([], []) for _ in range(n_blocks)]
    
    for i in range(nirreps):
        blocks[irreps_labels[i]][0].append(i)
    for i in range(rsize):
        blocks[field_labels[i]][1].append(i)
    
    change_of_basis = np.zeros_like(repr.change_of_basis)
    
    representations = []
    current_position = 0
    for block, (irreps_indices, row_indices) in enumerate(blocks):
        
        irreps_indices = sorted(irreps_indices)
        row_indices = sorted(row_indices)
        
        total_size = len(row_indices)
        
        assert sum([irreps[irr].size for irr in irreps_indices]) == total_size
        
        col_indices = []
        for irr in irreps_indices:
            col_indices += list(range(irreps_pos[irr], irreps_pos[irr]+irreps[irr].size))
        
        assert len(col_indices) == total_size
        
        new_cob = repr.change_of_basis[np.ix_(row_indices, col_indices)]
        
        field_repr = Representation(repr.group,
                                      f"{repr.name}_{block}",
                                      [irreps[id].id for id in irreps_indices],
                                      new_cob,
                                      repr.supported_nonlinearities)
        representations.append(field_repr)

        next_position = current_position + len(row_indices)
        change_of_basis[current_position:next_position, row_indices] = np.eye(len(row_indices))
        
        current_position = next_position
        
    return change_of_basis, representations


def change_basis(repr: Representation,
                 change_of_basis: np.ndarray,
                 name: str,
                 supported_nonlinearities: List[str] = None
                 ) -> Representation:
    r"""
    Build a new representation from an already existing one by applying a change of basis.
    In other words, if :math:`\rho(\cdot)` is the representation and :math:`Q` the change of basis in input, the
    resulting representation will evaluate to :math:`Q \rho(\cdot) Q^{-1}`.
    
    Notice that the change of basis :math:`Q` has to be invertible.
    
    
    Args:
        repr (Representation): the input representation
        change_of_basis (~numpy.ndarray): the change of basis to apply
        name (str, optional): the name to use to identify the new representation
        supported_nonlinearities (list, optional): a list containing the ids of the supported non-linearities
            for the new representation

    Returns:
        the new representation

    """
    assert len(change_of_basis.shape) == 2
    assert change_of_basis.shape[0] == change_of_basis.shape[1]
    assert change_of_basis.shape[0] == repr.size
    
    if supported_nonlinearities is None:
        # by default, no non-linearities are supported
        supported_nonlinearities = []
    
    # compute the new change of basis
    new_cob = change_of_basis @ repr.change_of_basis
    new_cob_inv = repr.change_of_basis_inv @ sp.linalg.inv(change_of_basis)
    
    return Representation(repr.group, name, repr.irreps, new_cob,
                          supported_nonlinearities=supported_nonlinearities,
                          change_of_basis_inv=new_cob_inv)


def build_regular_representation(group: escnn.group.Group) -> Tuple[List[escnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""
    
    Build the regular representation of the input ``group``.
    As the regular representation has size equal to the number of elements in the group, only
    finite groups are accepted.
    
    Args:
        group (Group): the group whose representations has to be built

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the regular representation

    """
    assert group.order() > 0
    assert group.elements is not None and len(group.elements) > 0
    
    size = group.order()

    index = {e: i for i, e in enumerate(group.elements)}
    
    representation = {}
    character = {}
    
    for e in group.elements:
        # print(index[e], e)
        r = np.zeros((size, size), dtype=float)
        for g in group.elements:
            
            eg = e @ g

            i = index[g]
            j = index[eg]
            
            r[j, i] = 1.0
        
        representation[e] = r
        # the character maps an element to the trace of its representation
        character[e] = np.trace(r)

    # compute the multiplicities of the irreps from the dot product between
    # their characters and the character of the representation
    irreps = []
    multiplicities = []
    for irrep in group.irreps():
        # for each irrep
        multiplicity = 0.0
    
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(~element)
    
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
    
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep [%s] is not an integer: %f" % (str(irrep.id), multiplicity)
        # print(irrep_name, multiplicity)

        multiplicity = int(round(multiplicity))
        irreps += [irrep]*multiplicity
        multiplicities += [(irrep, multiplicity)]
    
    P = directsum(irreps, name="irreps")
    
    v = np.zeros((size, 1), dtype=float)
    
    p = 0
    for irr, m in multiplicities:
        assert irr.size >= m
        s = irr.size
        v[p:p+m*s, 0] = np.eye(m, s).reshape(-1) * np.sqrt(s)
        p += m*s
        
    change_of_basis = np.zeros((size, size))
    
    # np.set_printoptions(precision=4, threshold=10*size**2, suppress=False, linewidth=25*size + 5)
    
    for e in group.elements:
        ev = P(e) @ v
        change_of_basis[index[e], :] = ev.T
    
    change_of_basis /= np.sqrt(size)
    
    # the computed change of basis is an orthonormal matrix
    
    # change_of_basis_inv = sp.linalg.inv(change_of_basis)
    change_of_basis_inv = change_of_basis.T
    
    return irreps, change_of_basis, change_of_basis_inv
    
    # return Representation(group,
    #                       "regular",
    #                       [r.name for r in irreps],
    #                       change_of_basis,
    #                       ['pointwise', 'norm', 'gated', 'concatenated'],
    #                       representation=representation,
    #                       change_of_basis_inv=change_of_basis_inv)


def build_quotient_representation(group: escnn.group.Group,
                                  subgroup_id
                                  ) -> Tuple[List[escnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""

    Build the quotient representation of the input ``group`` with respect to the subgroup identified by ``subgroup_id``.
    
    .. seealso::
        See the :class:`~escnn.group.Group` instance's implementation of the method :meth:`~escnn.group.Group.subgroup`
        for more details on ``subgroup_id``.
    
    .. warning ::
        Only finite groups are supported
    
    Args:
        group (Group): the group whose representation has to be built
        subgroup_id: identifier of the subgroup

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the quotient representation

    """
    subgroup, _, _ = group.subgroup(subgroup_id)
    return build_induced_representation(group,
                                        subgroup_id,
                                        subgroup.trivial_representation)


def build_induced_representation(group: escnn.group.Group,
                                 subgroup_id,
                                 repr: escnn.group.IrreducibleRepresentation,
                                 representatives: List[GroupElement] = None,
                                 ) -> Tuple[List[escnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""

    Build the induced representation of the input ``group`` from the representation ``repr`` of the subgroup
    identified by ``subgroup_id``.

    .. seealso::
        See the :class:`~escnn.group.Group` instance's implementation of the method :meth:`~escnn.group.Group.subgroup`
        for more details on ``subgroup_id``.

    .. warning ::
        Only irreducible representations are supported as the subgroup representation.
        
    .. warning ::
        It is not possible to compute the index of [G:H] when they are not finite groups.
        Therefore, it is not possible to check that `representatives` contains sufficient elements.
        In case it does not, the construction of the final representation will probably fail.

    # TODO add note about the fact all irreps of `group` which contain `repr` need to be inside `group.irreps`
    
    Args:
        group (Group): the group whose representation has to be built
        subgroup_id: identifier of the subgroup
        repr (IrreducibleRepresentation): the representation of the subgroup
        representatives (list, optional): list of coset representatives used to define the induced representation

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the induced representation

    """
    
    assert repr.irreducible, "Induction from general representations is not supported yet"
    
    subgroup, parent, child = group.subgroup(subgroup_id)
    
    assert repr.group == subgroup
    homspace = group.homspace(subgroup_id)

    if group.order() > 0:
        quotient_size = int(group.order() / subgroup.order())
        
        if representatives is None:
            representatives = []
            # the coset each element belongs to
            cosets = {}
            for e in group.elements:
                if e not in cosets:
                    representatives.append(e)
                    for g in subgroup.elements:
                        eg = e @ homspace._inclusion(g)
                        cosets[eg] = e
                        
        assert quotient_size == len(representatives)
        
    else:
        assert representatives is not None
        quotient_size = len(representatives)
            
    size = repr.size * quotient_size
    
    # check that all representatives belong to different cosets
    for i, r1 in enumerate(representatives):
        for j, r2 in enumerate(representatives):
            if i != j:
                assert not homspace.same_coset(r1, r2), (r1, r2)

    change_of_basis = []
    irreps = []
    
    for rho in group.irreps():
        basis_rho = homspace._dirac_kernel_ft(rho.id, repr.id)
    
        m_rho = basis_rho.shape[1]
    
        irreps += [rho] * m_rho
    
        if m_rho > 0:
            ift = np.zeros((len(representatives), repr.size, m_rho, rho.size))
            for i, r in enumerate(representatives):
                ift[i, ...] = np.einsum('oi, imp->pmo', rho(r), basis_rho)
        
            ift /= np.sqrt(len(representatives))
            ift /= np.sqrt(repr.size)
            ift *= np.sqrt(rho.size)
        
            assert np.allclose(
                    ift.reshape(size, -1).T @ ift.reshape(size, -1),
                    np.eye(rho.size * m_rho)
            ), ift.reshape(size, -1).T @ ift.reshape(size, -1)
        
            ift = ift.reshape(size, rho.size * m_rho)
            change_of_basis.append(ift)

    change_of_basis = np.concatenate(change_of_basis, axis=1)
    
    assert change_of_basis.shape[0] <= change_of_basis.shape[1], f'Error! Induced representation not complete because group {group} is missing some irreps'
    assert change_of_basis.shape[0] >= change_of_basis.shape[1], f'Error! Induced representation not complete because the set of representatives passed is not complete'
    
    assert np.allclose(change_of_basis @ change_of_basis.T, np.eye(size))
    assert np.allclose(change_of_basis.T @ change_of_basis, np.eye(size))

    return irreps, change_of_basis, change_of_basis.T


def direct_sum_factory(irreps: List[escnn.group.IrreducibleRepresentation],
                       change_of_basis: np.ndarray,
                       change_of_basis_inv: np.ndarray = None
                       ) -> Callable[[GroupElement], np.ndarray]:
    """
    The method builds and returns a function implementing the direct sum of the "irreps" transformed by the given
    "change_of_basis" matrix.

    More precisely, the built method will take as input a value accepted by all the irreps, evaluate the irreps on that
    input and return the direct sum of the produced matrices left and right multiplied respectively by the
    change_of_basis matrix and its inverse.

    Args:
        irreps (list): list of irreps
        change_of_basis: the matrix transforming the direct sum of the irreps
        change_of_basis_inv: the inverse of the change of basis matrix

    Returns:
        function taking an input accepted by the irreps and returning the direct sum of the irreps evaluated
        on that input
    """
    size = sum(irr.size for irr in irreps)

    if change_of_basis is not None:
        shape = change_of_basis.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert shape[0] == size

        if change_of_basis_inv is None:
            # pre-compute the inverse of the change-of-_bases matrix
            change_of_basis_inv = linalg.inv(change_of_basis)
        else:
            assert len(change_of_basis_inv.shape) == 2
            assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
            assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
            assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
            assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
    
    unique_irreps = list({irr.id: irr for irr in irreps}.items())
    irreps_ids = [irr.id for irr in irreps]
    
    def direct_sum(element: GroupElement,
                   irreps_ids=irreps_ids, change_of_basis=change_of_basis,
                   change_of_basis_inv=change_of_basis_inv, unique_irreps=unique_irreps):
        reprs = {}
        for n, irr in unique_irreps:
            reprs[n] = irr(element)
        
        blocks = []
        for irrep_id in irreps_ids:
            repr = reprs[irrep_id]
            blocks.append(repr)
        
        P = sparse.block_diag(blocks, format='csc')

        if change_of_basis is None:
            return np.asarray(P.todense())
        else:
            return change_of_basis @ P @ change_of_basis_inv
    
    return direct_sum


def homomorphism_space(rho1: Representation, rho2: Representation) -> np.ndarray:
    r"""
        Compute a basis for the space of homomorphisms from `rho1` to `rho2`.

        .. warning::
            This basis might become quite big for large representations and, therefore, this method might get
            computationally expensive.
            In practice, we recommend using this method only to generate the basis for small representations (e.g.
            for irreps).
            This basis is generated and stored more efficiently in the :doc:`escnn.nn` package, when parameterizing
            neural network modules; e.g. see :class:`~escnn.nn.BlocksBasisExpansion`.
            
    """
    assert rho1.group == rho2.group
    G = rho1.group
    
    dim = 0
    for in_irrep in rho1.irreps:
        in_irrep = G.irrep(*in_irrep)
        for out_irrep in rho2.irreps:
            out_irrep = G.irrep(*out_irrep)
            if in_irrep == out_irrep:
                dim += in_irrep.sum_of_squares_constituents
    
    basis = np.zeros((dim, rho2.size, rho1.size))
    
    p = 0
    i = 0
    for in_irrep in rho1.irreps:
        in_irrep = G.irrep(*in_irrep)
        
        j = 0
        for out_irrep in rho2.irreps:
            out_irrep = G.irrep(*out_irrep)
            
            if out_irrep == in_irrep:
                basis[
                    p:p + in_irrep.sum_of_squares_constituents,
                    j:j + in_irrep.size,
                    i:i + in_irrep.size
                ] = in_irrep.endomorphism_basis()
                
                p += in_irrep.sum_of_squares_constituents
            
            j += out_irrep.size
        i += in_irrep.size
    
    basis = np.einsum('Mm,kmn,nN->kMN', rho2.change_of_basis, basis, rho1.change_of_basis_inv)
    return basis
