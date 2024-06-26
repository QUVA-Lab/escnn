
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Iterable, List, Any, Dict

import escnn.group
import numpy as np
from scipy import sparse


__all__ = ["Group", "GroupElement"]


class Group(ABC):
    
    @property
    @abstractmethod
    def PARAM(self) -> str:
        f"""
            Default parametrization used for storing the elements of the group.
        """
        pass

    @property
    @abstractmethod
    def PARAMETRIZATIONS(self) -> List[str]:
        f"""
            List of all supported parametrizations of the group.
        """
        pass

    def __init__(self, name: str, continuous: bool, abelian: bool):
        r"""
        Abstract class defining the interface of a group.
        
        A group is a set of *group elements* together with a binary operation satisfying a number of axioms.
        
        In this library, this is implemented using this class :class:`~escnn.group.Group` and the class
        :class:`~escnn.group.GroupElement`.
        
        One can retrieve or generate elements of a group by using, for instance, the properties or methods
        :meth:`~escnn.group.Group.identity` , :meth:`~escnn.group.Group.elements` or :meth:`~escnn.group.Group.sample`.
        Each group may also have additional methods to generate its group elements.
        Additionally, one can use the method :meth:`~escnn.group.Group.element` to generate a new group element.
        
        The group algebra is directly implemented inside :class:`~escnn.group.GroupElement` such that one can combine
        group elements in a way that resamples mathematical expressions.
        In particular, the ``@`` implements the binary product while ``~`` implements the group inverse.
        See :class:`~escnn.group.GroupElement` for more details.
        
        
        Args:
            name (str): name identifying the group
            continuous (bool): whether the group is non-finite or finite
            abelian (bool): whether the group is *abelian* (commutative)
            
        Attributes:
            ~.name (str): Name identifying the group
            ~.continuous (bool): Whether it is a non-finite or a finite group
            ~.abelian (bool): Whether it is an *abelian* group (i.e. if the group law is commutative)

        """
        
        self.name = name
        
        self.continuous = continuous
        
        self.abelian = abelian
        
        self._irreps = {}
        
        self._representations = {}
        
        self._subgroups = {}
        
        self._homspaces = {}
        
    def order(self) -> int:
        r"""
        Returns the number of elements in this group if it is a finite group, otherwise -1 is returned
        
        Returns:
            the size of the group or ``-1`` if it is a continuous group

        """
        if self.elements is not None:
            return len(self.elements)
        else:
            return -1

    def element(self, element, param: str = None) -> GroupElement:
        r"""
            Generate the element of the current group parametrized by ``element`` according to
            the parametrization ``param``.
            
            Each group supports a different set of parametrizations.
            By default, the parametrization :attr:`escnn.group.Group.PARAM` is used.
            The list of all available parametrizations of a group can be accessed through the property
            :attr:`escnn.group.Group.PARAMETRIZATIONS`.
            
        Args:
            element: values parametrizing a group element.
            param (str): string identifying the parametrization to be used.

        Returns:
            an instance of :class:`~escnn.group.GroupElement`

        """
        if param is None:
            param = self.PARAM
            
        return GroupElement(element, self, param)
    
    @property
    @abstractmethod
    def identity(self) -> GroupElement:
        r"""
            The identity element of the group.
            
            The identity element :math:`e` satisfies the following property
            :math:`\forall\ g \in G,\ g \cdot e = e \cdot g= g` .
            
        """
        pass

    @property
    @abstractmethod
    def elements(self) -> List[GroupElement]:
        r"""
            If the group is finite (i.e. ``self.continuous = False``), it is a list of all group elements.
            If the group is not finite, this property is `None`.
            
            Returns:
                a list of :class:`~escnn.group.GroupElement` instances
        """
        pass

    @property
    @abstractmethod
    def subgroup_trivial_id(self):
        r"""
            The subgroup `id` associated with the trivial subgroup containing only the identity element :math:`{e}`.
            The id can be used in the method :meth:`~escnn.group.Group.subgroup` to generate the subgroup.
        """
        pass
    
    @property
    @abstractmethod
    def subgroup_self_id(self):
        r"""
            The subgroup `id` associated with the group itself.
            The id can be used in the method :meth:`~escnn.group.Group.subgroup` to generate the subgroup.
        """
        pass

    @property
    @abstractmethod
    def _keys(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def generators(self) -> List[GroupElement]:
        r"""
            If the group is finite (``self.continuous = False``), a list of group elements which can generate this group.
            Should raise a `ValueError` if the group is not finite.
            
            Returns:
                a list of :class:`~escnn.group.GroupElement` instances
        """
        pass
    
    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################

    @abstractmethod
    def _combine(self, e1, e2,
                 param: str,
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
        pass

    @abstractmethod
    def _inverse(self, element, param: str):
        r"""
        Method that returns the inverse in the group of the element given as input

        Args:
            element: an element of the group

        Returns:
            its inverse
        """
        pass

    @abstractmethod
    def _is_element(self,
                    element,
                    param: str,
                    verbose: bool = False
                    ) -> bool:
        r"""
        Check whether the input is an element of this group or not.

        Args:
            element: input object to test

        Returns:
            if the input is an element of the group

        """
        pass

    @abstractmethod
    def _equal(self, e1, e2,
               param: str,
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
        pass

    @abstractmethod
    def _change_param(self, element, p_from: str, p_to: str):
        pass

    @abstractmethod
    def _hash_element(self, element, param: str):
        r"""
        Method that returns a unique hash for a group element given in input

        Args:
            element: an element of the group

        Returns:
            a unique hash
        """
        pass
    
    @abstractmethod
    def _repr_element(self, element, param: str):
        r"""
        Method that returns a representative string for a group element given in input

        Args:
            element: an element of the group

        Returns:
            a unique hash
        """
        pass

    ###########################################################################

    def __repr__(self):
        return self.name
    
    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def sample(self) -> GroupElement:
        r"""
            Sample a random element of the group from a uniform distribution over the group.
           
            Returns:
                :class:`~escnn.group.GroupElement`: the element sampled
            
        """
        pass

    def grid(self, *args, **kwargs) -> List[GroupElement]:
        r"""
        Method to generate collections fo points over the group.
        Each group should implement its own set of collections.
        Check the individual groups' documentations for details about the supported arguments.

        Returns:
            a list of :class:`~escnn.group.GroupElement` instances

        """
        raise NotImplementedError()

    def _process_subgroup_id(self, id):
        return id

    def subgroup(self, id) -> Tuple[
        escnn.group.Group,
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement],
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement]
    ]:
        r"""
        Restrict the current group to the subgroup identified by the input ``id``.

        Args:
            id: the identifier of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)

        """
        id = self._process_subgroup_id(id)
        
        if id not in self._subgroups:
            subgroup, parent_mapping, child_mapping = self._subgroup(id)
            self._subgroups[id] = subgroup, parent_mapping, child_mapping

        return self._subgroups[id]

    @abstractmethod
    def _subgroup(self, id) -> Tuple[
        escnn.group.Group,
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement],
        Callable[[escnn.group.GroupElement], escnn.group.GroupElement]
    ]:
        r"""
        Restrict the current group to the subgroup identified by the input ``id``.

        Args:
            id: the identifier of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                -a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)

        """
        pass

    def _combine_subgroups(self, sg_id1, sg_id2):
        raise NotImplementedError
    
    def irreps(self) -> List[escnn.group.IrreducibleRepresentation]:
        r"""
        List containing all irreducible representations (:class:`~escnn.group.IrreducibleRepresentation`)
        currently instantiated for this group.

        Returns:
            a list containing all irreducible representations built

        """
        return list(self._irreps.values())

    @property
    def representations(self) -> Dict[str, escnn.group.Representation]:
        r"""
        Dictionary containing all representations (:class:`~escnn.group.Representation`)
        instantiated for this group.

        Returns:
            a dictionary containing all representations built

        """
        return self._representations
    
    @property
    @abstractmethod
    def trivial_representation(self) -> escnn.group.IrreducibleRepresentation:
        r"""
        Builds the trivial representation of the group.
        The trivial representation is a 1-dimensional representation which maps any element to 1,
        i.e. :math:`\forall g \in G,\ \rho(g) = 1`.
        
        Returns:
            the trivial representation of the group

        """
        pass

    @abstractmethod
    def irrep(self, *id) -> escnn.group.IrreducibleRepresentation:
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
        # TODO implement memoization here and let subclasses define an _irrep(*id) module
        pass

    @property
    def regular_representation(self) -> escnn.group.Representation:
        r"""
        Builds the regular representation of the group if the group has a *finite* number of elements;
        returns ``None`` otherwise.
        
        The regular representation of a finite group :math:`G` acts on a vector space :math:`\R^{|G|}` by permuting its
        axes.
        Specifically, associating each axis :math:`e_g` of :math:`\R^{|G|}` to an element :math:`g \in G`, the
        representation of an element :math:`\tilde{g}\in G` is a permutation matrix which maps :math:`e_g` to
        :math:`e_{\tilde{g}g}`.
        For instance, the regular representation of the group :math:`C_4` with elements
        :math:`\{r^k | k=0,\dots,3 \}` is instantiated by:
        
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        |    :math:`g`                      |          :math:`e`                                                                                         |          :math:`r`                                                                                         |        :math:`r^2`                                                                                         |        :math:`r^3`                                                                                         |
        +===================================+============================================================================================================+============================================================================================================+============================================================================================================+============================================================================================================+
        |  :math:`\rho_\text{reg}^{C_4}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\  0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\  0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ \end{bmatrix}` |
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        
        A vector :math:`v=\sum_g v_g e_g` in :math:`\R^{|G|}` can be interpreted as a scalar function
        :math:`v:G \to \R,\, g \mapsto v_g` on :math:`G`.
        
        Returns:
            the regular representation of the group

        """
        if self.order() < 0:
            raise ValueError(f"Regular representation is supported only for finite groups but "
                             f"the group {self.name} has an infinite number of elements")
        else:
            if "regular" not in self.representations:
                irreps, change_of_basis, change_of_basis_inv = escnn.group.representation.build_regular_representation(self)
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'concatenated']
                self.representations["regular"] = escnn.group.Representation(self,
                                                                             "regular",
                                                                             [r.id for r in irreps],
                                                                             change_of_basis,
                                                                             supported_nonlinearities,
                                                                             change_of_basis_inv=change_of_basis_inv,
                                                                             )
            return self.representations["regular"]

    def quotient_representation(self,
                                subgroup_id,
                                representatives: List[GroupElement] = None,
                                name: str = None,
                                ) -> escnn.group.Representation:
        r"""
        Builds the quotient representation of the group with respect to the subgroup identified by the
        input ``subgroup_id``.
        
        Similar to :meth:`~escnn.group.Group.regular_representation`, the quotient representation
        :math:`\rho_\text{quot}^{G/H}` of :math:`G` w.r.t. a subgroup :math:`H` acts on :math:`\R^{|G|/|H|}` by
        permuting its axes.
        Labeling the axes by the cosets :math:`gH` in the quotient space :math:`G/H`, it can be defined via its action
        :math:`\rho_\text{quot}^{G/H}(\tilde{g})e_{gH}=e_{\tilde{g}gH}`.

        Regular and trivial representations are two specific cases of quotient representations obtained by choosing
        :math:`H=\{e\}` or :math:`H=G`, respectively.
        Vectors in the representation space :math:`\R^{|G|/|H|}` can be viewed as scalar functions on the quotient
        space :math:`G/H`.
        
        The quotient representation :math:`\rho_\text{quot}^{G/H}` can also be defined as the
        :meth:`~escnn.group.Group.induced_representation` from the trivial representation of the subgroup :math:`H`.
        
        .. todo ::
            docs for `representatives`
        
        
        Args:
            subgroup_id: identifier of the subgroup
            representatives (list, optional):
            name (str, optional): optionally, specify a custom name for this representation
        
        Returns:
            the quotient representation of the group

        """

        if name is None:
            name = f"quotient[{subgroup_id}]"
        
        if name not in self.representations:
            subgroup, _, _ = self.subgroup(subgroup_id)
            
            supported_nonlinearities = _induced_nonlinearities(subgroup.trivial_representation)
            
            irreps, change_of_basis, change_of_basis_inv = self._induced_from_irrep(
                subgroup_id,
                subgroup.trivial_representation,
                representatives
            )
            self.representations[name] = escnn.group.Representation(self,
                                                                    name,
                                                                    [r.id for r in irreps],
                                                                    change_of_basis,
                                                                    supported_nonlinearities,
                                                                    change_of_basis_inv=change_of_basis_inv,
                                                                    )

        return self.representations[name]

    def induced_representation(self,
                               subgroup_id,
                               repr: escnn.group.IrreducibleRepresentation,
                               representatives: List[GroupElement] = None,
                               name: str = None
                               ) -> escnn.group.Representation:
        r"""
        Builds the induced representation from the input representation ``repr`` of the subgroup identified by
        the input ``subgroup_id``.
        
        .. todo ::
            docs for `representatives`
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup
            representatives (list, optional):
            name (str, optional): optionally, specify a custom name for this representation

        Returns:
            the induced representation of the group

        """
        
        assert repr.irreducible, "Induction from general representations is not supported yet"

        if name is None:
            name = f"induced[{subgroup_id}][{repr.name}]"

        if name not in self.representations:

            supported_nonlinearities = _induced_nonlinearities(repr)

            irreps, change_of_basis, change_of_basis_inv = self._induced_from_irrep(
                subgroup_id,
                repr,
                representatives
            )
            self.representations[name] = escnn.group.Representation(self,
                                                                    name,
                                                                    [r.id for r in irreps],
                                                                    change_of_basis,
                                                                    supported_nonlinearities,
                                                                    change_of_basis_inv=change_of_basis_inv,
                                                                    )

        return self.representations[name]

    def _induced_from_irrep(self, subgroup_id: Tuple[float, int],
                            repr: escnn.group.IrreducibleRepresentation,
                            representatives: List[GroupElement] = None,
                            ) -> Tuple[List[escnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    
        r"""
        Builds the induced representation from the input *irreducible* representation ``repr`` of the subgroup
        identified by the input ``subgroup_id``.
        
        .. todo ::
            docs for `representatives`
        
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup
            

        Returns:
            a tuple containing the list of irreps, the change of basis and the inverse change of basis of
            the induced representation

        """
    
        assert repr.irreducible
        return escnn.group.representation.build_induced_representation(
            self,
            subgroup_id,
            repr,
            representatives
        )

    def spectral_regular_representation(self, *irreps, name: str = None) -> 'Representation':
        r"""
        Finite dimensional invariant subspace of the regular representation containing only the irreps passed in input.
        The regular representation is expressed in the spectral basis, i.e. as a direct sum of irreps.

        The optional parameter ``name`` is also used for caching purpose.
        Consecutive calls of this method using the same ``name`` will ignore the argument  ``irreps``
        and return the same instance of representation.

        .. seealso::
            :meth:`escnn.group.HomSpace.induced_representation`

        """

        if name is None:
            irreps_names = '|'.join(str(i) for i in irreps)
            name = f'regular_[{irreps_names}]'

        return self.spectral_quotient_representation(self.subgroup_trivial_id, *irreps, name=name)

    def spectral_quotient_representation(self, subgroup_id: Tuple, *irreps, name: str = None) -> 'Representation':
        r"""
        Finite dimensional invariant subspace of the quotient representation containing only the irreps passed in input.
        The quotient representation is expressed in the spectral basis, i.e. as a direct sum of irreps.

        The optional parameter ``name`` is also used for caching purpose.
        Consecutive calls of this method using the same ``name`` will ignore the arguments ``subgroup_id`` and ``irreps``
        and return the same instance of representation.

        .. seealso::
            :meth:`escnn.group.HomSpace.induced_representation`

        """

        if name is None:
            irreps_names = '|'.join(str(i) for i in irreps)
            name = f'quotient[{subgroup_id}]_[{irreps_names}]'

        if name not in self._representations:
            homspace = self.homspace(subgroup_id)
            self._representations[name] = homspace.induced_representation(homspace.H.trivial_representation.id, irreps, name)

        return self._representations[name]

    def restrict_representation(self, id, repr: escnn.group.Representation) -> escnn.group.Representation:
        r"""

        Restrict the input :class:`~escnn.group.Representation` to the subgroup identified by ``id``.
        
        Any representation :math:`\rho : G \to \GL{\R^n}` can be uniquely restricted to a representation
        of a subgroup :math:`H < G` by restricting its domain of definition:

        .. math ::

            \Res{H}{G}(\rho): H \to \GL{{\R}^n},\ h \mapsto \rho\big|_H(h)
        
        We recommend directly using the method :meth:`escnn.group.Representation.restrict`.
        
        .. seealso ::

            Check the documentation of the method :meth:`~escnn.group.Group.subgroup()` of the group used to see
            the available subgroups and accepted ids.

        Args:
            id: identifier of the subgroup
            repr (Representation): the representation to restrict

        Returns:
            the restricted representation

        """
    
        assert repr.group == self
    
        sg, _, _ = self.subgroup(id)
        id = self._process_subgroup_id(id)

        # First, restrict each irrep in the representation
    
        irreps_changes_of_basis = []
        irreps = []
    
        for irr in repr.irreps:
            irrep_cob, reduced_irreps = self._restrict_irrep(irr, id)
            size = self.irrep(*irr).size
            assert irrep_cob.shape == (size, size)
        
            irreps_changes_of_basis.append(irrep_cob)
            irreps += reduced_irreps
    
        # concatenate the restricted irreps and merge the representation's change of basis with the
        # restricted irreps' change of basis matrices
        irreps_changes_of_basis = sparse.block_diag(irreps_changes_of_basis, format='csc')
        change_of_basis = repr.change_of_basis @ irreps_changes_of_basis
    
        name = f"{self.name}:{repr.name}"
    
        resr = escnn.group.Representation(sg,
                                          name,
                                          irreps,
                                          change_of_basis,
                                          repr.supported_nonlinearities)
    
        if resr.is_trivial() and 'pointwise' not in repr.supported_nonlinearities:
            resr.supported_nonlinearities.add("pointwise")
    
        return resr

    def homspace(self, id) -> escnn.group.HomSpace:
        r"""
            If :math:`G` is the current group and ``id`` identifies the subgroup :math:`H`
            (see :meth:`~escnn.group.Group.subgroup`), this method generates the homogeneous space
            :class:`~escnn.group.HomSpace` :math:`X = G / H`.
            
            .. note ::

                The generated instance of :class:`~escnn.group.HomSpace` is cached inside the instance of the current
                group such that repeated calls of this method using the same ``id`` return the same instance of
                :class:`~escnn.group.HomSpace` and no additional computations are required.
            
            
        Returns:
            an instance of :class:`~escnn.group.HomSpace`

        """
        id = self._process_subgroup_id(id)
        
        if id not in self._homspaces:
            self._homspaces[id] = escnn.group.HomSpace(self, self._process_subgroup_id(id))
            
        return self._homspaces[id]
    
    @abstractmethod
    def _restrict_irrep(self, irrep: Tuple, id) -> Tuple[np.matrix, List[Tuple]]:
        pass

    def _clebsh_gordan_coeff(self, m, n, j) -> np.ndarray:
        group_keys = self._keys
        m = self.get_irrep_id(m)
        n = self.get_irrep_id(n)
        j = self.get_irrep_id(j)
        return escnn.group._clebsh_gordan._clebsh_gordan_tensor(m, n, j, self.__class__.__name__, **group_keys)

    def _tensor_product_irreps(self, m, n) -> List[Tuple[Tuple, int]]:
        group_keys = self._keys
        m = self.get_irrep_id(m)
        n = self.get_irrep_id(n)
        return escnn.group._clebsh_gordan._find_tensor_decomposition(m, n, self.__class__.__name__, **group_keys)

    def _tensor_product(self, rho1: escnn.group.Representation, rho2: escnn.group.Representation) -> escnn.group.Representation:
        assert rho1.group == self
        assert rho2.group == self
        
        D1 = rho1.size
        D2 = rho2.size
        D = D1 * D2
        
        change_of_basis = np.zeros((D, D))
        irreps = []
        
        p = 0
        for irr1 in rho1.irreps:
            irr1 = self.irrep(*irr1)
            
            permutation = np.zeros((irr1.size * rho2.size, irr1.size * rho2.size))
            q = 0
            
            for irr2 in rho2.irreps:
                irr2 = self.irrep(*irr2)

                irr1_tensor_irr2 = self._tensor_product_irreps(irr1.id, irr2.id)
                size = 0
                for irr_id, S in irr1_tensor_irr2:
                    irr = self.irrep(*irr_id)
                    size += irr.size*S
                    irreps += [irr.id]*S
                
                assert size == irr1.size * irr2.size, (size, irr1.size, irr2.size)

                i = 0
                for irr_j, S in irr1_tensor_irr2:
                    irr_j = self.irrep(*irr_j)
                    change_of_basis[
                        p:p+size,
                        p+i:p+i+irr_j.size*S
                    ] = self._clebsh_gordan_coeff(irr1.id, irr2.id, irr_j.id).reshape(-1, irr_j.size*S)
                    i += irr_j.size * S
                
                assert i == size
                
                for i in range(irr1.size):
                    permutation[
                        q + i*rho2.size:q + i*rho2.size + irr2.size,
                        q*irr1.size + i * irr2.size:q*irr1.size + (i+1) * irr2.size
                    ] = np.eye(irr2.size)
                
                q += irr2.size
                p += size
            
            assert np.allclose(permutation @ permutation.T, np.eye(permutation.shape[0]))
            assert np.allclose(permutation.T @ permutation, np.eye(permutation.shape[0]))

            change_of_basis[
                p - irr1.size*rho2.size:p,
                p - irr1.size*rho2.size:p
            ] = permutation @ change_of_basis[p - irr1.size*rho2.size:p, p - irr1.size*rho2.size:p]
            
        change_of_basis = np.kron(rho1.change_of_basis, rho2.change_of_basis) @ change_of_basis

        assert p == sum(self.irrep(*irr).size for irr in irreps), (p, rho1.size, rho2.size)
        assert p == rho1.size * rho2.size, (p, rho1.size, rho2.size)
        assert p == change_of_basis.shape[0]
        assert p == change_of_basis.shape[1]

        supported_nonlinearities = _tensor_nonlinearities(rho1, rho2)

        character = _tensor_product_character(rho1, rho2)
        
        if len(irreps) > 1:
            return escnn.group.Representation(self,
                                              f'{rho1.name} X {rho2.name}',
                                              irreps,
                                              change_of_basis,
                                              character=character,
                                              supported_nonlinearities=supported_nonlinearities
                                              )
        else:
            return escnn.group.change_basis(
                self.irrep(*irreps[0]),
                change_of_basis,
                name=f'{rho1.name} X {rho2.name}'
            )

    @abstractmethod
    def testing_elements(self) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        pass

    def get_irrep_id(self, psi):
        if isinstance(psi, escnn.group.IrreducibleRepresentation):
            assert psi.group == self
            return psi.id
        elif isinstance(psi, str):
            psi = self.representations[psi]
            assert isinstance(psi, escnn.group.IrreducibleRepresentation)
            return psi
        elif isinstance(psi, tuple):
            return self.irrep(*psi).id
        else:
            return self.irrep(psi).id

    def _decode_subgroup_id_pickleable(self, id: Tuple) -> Tuple:

        if isinstance(id, tuple):
            if id[0] == 'GROUPELEMENT':
                id = self.element(id[1], id[2])
            else:
                id = list(id)
                for i in range(len(id)):
                    id[i] = self._decode_subgroup_id_pickleable(id[i])
                id = tuple(id)

        return id

    def _encode_subgroup_id_pickleable(self, id: Tuple) -> Tuple:
        if isinstance(id, GroupElement):
            id = 'GROUPELEMENT', id.value, id.param
        elif isinstance(id, tuple):
            id = list(id)
            for i in range(len(id)):
                id[i] = self._encode_subgroup_id_pickleable(id[i])
            id = tuple(id)
        return id

    @classmethod
    @abstractmethod
    def _generator(cls, *args, **kwargs) -> 'Group':
        # TODO solve the singleton problem!!!
        pass


def _tensor_product_character(rho1: 'Representation', rho2: 'Representation'):
    
    def character(e: GroupElement, rho1=rho1, rho2=rho2) -> float:
        return rho1.character(e) * rho2.character(e)
    
    return character


def _induced_nonlinearities(repr: escnn.group.Representation):
    
    supported_nonlinearities = []
    
    if 'pointwise' in repr.supported_nonlinearities:
        supported_nonlinearities.append('pointwise')
    if 'concatenated' in repr.supported_nonlinearities:
        supported_nonlinearities.append('concatenated')
    if 'gated' in repr.supported_nonlinearities:
        supported_nonlinearities.append('gated')
        for nl in repr.supported_nonlinearities:
            if nl.startswith('induced_gated'):
                supported_nonlinearities.append(nl)
                break
        else:
            supported_nonlinearities.append(f'induced_gated_{repr.size}')
    if 'norm' in repr.supported_nonlinearities:
        supported_nonlinearities.append('norm')
        for nl in repr.supported_nonlinearities:
            if nl.startswith('induced_norm'):
                supported_nonlinearities.append(nl)
                break
        else:
            supported_nonlinearities.append(f'induced_norm_{repr.size}')
    if 'gate' in repr.supported_nonlinearities or 'induced_gate' in repr.supported_nonlinearities:
        supported_nonlinearities.append('induced_gate')

    # 'vectorfield' not always supported by the induced representation so they are not added

    return supported_nonlinearities


def _tensor_nonlinearities(repr1: escnn.group.Representation, repr2: escnn.group.Representation):
    supported_nonlinearities = []
    
    if 'pointwise' in repr1.supported_nonlinearities and 'pointwise' in repr2.supported_nonlinearities:
        supported_nonlinearities.append('pointwise')
        
    supported_nonlinearities.append('gated')
    supported_nonlinearities.append('norm')

    if 'gate' in repr1.supported_nonlinearities and 'gate' in repr2.supported_nonlinearities:
        supported_nonlinearities.append('gate')
        
    # TODO - check for induced non-linearities
    
    return supported_nonlinearities


class GroupElement(ABC):
    
    def __init__(self, g, group: Group, param: str = None):
        r"""
        
        Class implementing an element of a group.
        
        Group elements can be combined the group operations like the *group law* or the *inverse*.
        In particular, one can combine two group elements through the group law using the
        operator ``@`` or compute the inverse of an element using ``~``.
        For example ::
            
            G = so3_group()
            a = G.sample()
            b = G.sample()
            
            c = a @ b
            a_ = ~a
            
            e = G.identity
            
            assert e == ~e
            assert a == a @ e
            
            
        Args:
            g: values parametrizing the group element
            group (Group): the group this element belongs to
            param (str): the type of parametrization of ``g``
            
        Attributes:
            ~.group (Group): the group it belongs to
            
        """

        if param is None:
            param = group.PARAM
        
        # TODO - create "ParametrizationError"?
        if param not in group.PARAMETRIZATIONS:
            raise ValueError(f'Error! {param} not recognized. Expected one of {group.PARAMETRIZATIONS}')
        
        if not group._is_element(g, param): #, verbose=True):
            raise ValueError(f'Error! {g} is not an element of {group} under the parametrization [{param}]')
        
        # Group: the group this element belongs to
        self.group = group
        
        # TODO: do lazy conversion. Keep always input parametrization.
        #  Convert to self.group.PARAM only when performing operations and creating new group elements
        self._element = group._change_param(g, param, group.PARAM)
    
    def __eq__(self, other: GroupElement):
        if not isinstance(other, GroupElement) or other.group != self.group:
            return False
        return self.group._equal(self._element, other._element, param1=self.param, param2=other.param)

    def __matmul__(self, other: GroupElement):
        if isinstance(other, GroupElement):
            if other.group == self.group:
                return GroupElement(
                    self.group._combine(self._element, other._element, param1=self.param, param2=other.param),
                    self.group,
                    self.param
                )
            else:
                raise NotImplementedError(f'Multiplication of group elements which belong to different groups is not supported.')
        else:
            return NotImplemented

    def __invert__(self):
        return GroupElement(
            self.group._inverse(self._element, param=self.param),
            self.group,
            self.param
        )

    def __hash__(self):
        return self.group._hash_element(self._element, self.param)

    def __repr__(self):
        return self.group._repr_element(self._element, self.param)

    @property
    def value(self):
        r"""
            Returns the values of the internal parametrization of the group element.
            These values parametrize the group element according to the parametrization
            :attr:`escnn.group.GroupElement.param`.

        """
        return self._element

    @property
    def param(self) -> str:
        r"""
            The type parametrization used internally to store this group element.
        """
        return self.group.PARAM

    def to(self, param: str):
        r"""
            Converts the current group element to the input parametrization ``param`` and returns the corresponding
            values.

            .. note ::
                This method does *not* return an instance of :class:`~escnn.group.GroupElement`.
                This method does *not* affect the internal representation of the element, but just converts it to the
                input ``param`` and returns the converted values.

        """
        return self.group._change_param(self._element, self.param, param)

