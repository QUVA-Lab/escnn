
from typing import List, Dict, Tuple, Union, Optional

from collections import defaultdict
from itertools import groupby

import escnn.nn
from escnn.group import Group, GroupElement
from escnn.group import Representation
from escnn.gspaces import GSpace
from escnn.group import directsum

import numpy as np
from scipy import sparse

import torch

__all__ = ["FieldType"]

# TODO:
# I think the band-limit frequency should be the argument to this class, not 
# the irreps.  I can get the irreps from the frequency, because I have the 
# group.  And it's nice to be able to compare against the frequency, e.g. for 
# determining how many grid points to use.

class FieldType:
    
    def __init__(self,
                 gspace: GSpace,
                 representations: Union[Tuple[Representation], List[Representation]]):
        r"""
        
        An ``FieldType`` can be interpreted as the *data type* of a feature space. It describes:
        
        - the base space on which a feature field is living and its symmetries considered
        
        - the transformation law of feature fields under the action of the fiber group
        
        The former is formalize by a choice of ``gspace`` while the latter is determined by a choice of group
        representations (``representations``), passed as a list of :class:`~escnn.group.Representation` instances.
        Each single representation in this list corresponds to one independent feature field contained in the feature
        space.
        The input ``representations`` need to belong to ``gspace``'s fiber group
        (:attr:`escnn.gspaces.GSpace.fibergroup`).
        
        .. note ::
            
            Mathematically, this class describes a *(trivial) vector bundle*, *associated* to the symmetry group
            :math:`(\R^D, +) \rtimes G`.
            
            Given a *principal bundle* :math:`\pi: (\R^D, +) \rtimes G \to \R^D, tg \mapsto tG`
            with fiber group :math:`G`, an *associated vector bundle* has the same base space
            :math:`\R^D` but its fibers are vector spaces like :math:`\mathbb{R}^c`.
            Moreover, these vector spaces are associated to a :math:`c`-dimensional representation :math:`\rho` of the
            fiber group :math:`G` and transform accordingly.
            
            The representation :math:`\rho` is defined as the *direct sum* of the representations :math:`\{\rho_i\}_i`
            in ``representations``. See also :func:`~escnn.group.directsum`.
            
        
        Args:
            gspace (GSpace): the space where the feature fields live and its symmetries
            representations (tuple, list): a list or tuple of :class:`~escnn.group.Representation` s of the ``gspace``'s fiber group,
                            determining the transformation laws of the feature fields
        
        Attributes:
            ~.gspace (GSpace)
            ~.representations (tuple)
            ~.size (int): dimensionality of the feature space described by the :class:`~escnn.nn.FieldType`.
                          It corresponds to the sum of the dimensionalities of the individual feature fields or
                          group representations (:attr:`escnn.group.Representation.size`).
 
            
        """
        assert isinstance(gspace, GSpace)
        assert len(representations) > 0

        assert isinstance(representations, tuple) or isinstance(representations, list)

        for repr in representations:
            assert repr.group == gspace.fibergroup
        
        # GSpace: Space where data lives and its (abstract) symmetries
        self.gspace = gspace

        if not isinstance(representations, tuple):
            representations = tuple(representations)

        # tuple: tuple containing the list of representations of each feature field composing the feature space of this type
        self.representations = representations
        
        # int: size of the field associated to this type.
        # as the representation associated to the field is the direct sum of the representations
        # in :attr:`escnn.nn.fieldtype.representations`, its size is the sum of each of these
        # representations' size
        self.size = sum([repr.size for repr in representations])

        self._unique_representations = set(self.representations)
        
        self._representation = None
        
        self._field_start = None
        self._field_end = None

        self._hash = hash(self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}')

        self._uniform = True
        rho = self.representations[0]
        for psi in self.representations:
            if psi != rho:
                self._uniform = False
                break

    @property
    def fibergroup(self) -> Group:
        r"""
        The fiber group of :attr:`~escnn.nn.FieldType.gspace`.

        Returns:
            the fiber group

        """
        return self.gspace.fibergroup

    @property
    def representation(self) -> Representation:
        r"""
        The (combined) representations of this field type.
        They describe how the feature vectors transform under the fiber group action, that is, how the channels mix.
 
        It is the direct sum (:func:`~escnn.group.directsum`) of the representations in
        :attr:`escnn.nn.FieldType.representations`.
        
        Because a feature space can contain a very large number of feature fields, computing this representation as
        the direct sum of many small representations can be expensive.
        Hence, this representation is only built the first time it is explicitly used, in order to avoid unnecessary
        overhead when not needed.
        
        Returns:
            the :class:`~escnn.group.Representation` describing the whole feature space
            
        """
        if self._representation is None:
            uniques_fields_names = sorted([r.name for r in self._unique_representations])
            self._representation = directsum(list(self.representations), name=f"FiberRepresentation:[{self.size}], [{uniques_fields_names}]")

        return self._representation

    @property
    def irreps(self) -> List[Tuple]:
        r"""
        Ordered list of irreps contained in the :attr:`~escnn.nn.FieldType.representation` of the field type.
        It is the concatenation of the irreps in each representation in :attr:`escnn.nn.FieldType.representations`.

        Returns:
            list of irreps

        """
        irreps = []
        for repr in self.representations:
            irreps += repr.irreps
        return irreps

    @property
    def change_of_basis(self) -> sparse.coo_matrix:
        r"""
        
        The change of basis matrix which decomposes the field types representation into irreps, given as a sparse
        (block diagonal) matrix (:class:`scipy.sparse.coo_matrix`).
        
        It is the direct sum of the change of basis matrices of each representation in
        :attr:`escnn.nn.FieldType.representations`.
        
        .. seealso ::
            :attr:`escnn.group.Representation.change_of_basis`
 
        
        Returns:
            the change of basis
        
        """
        change_of_basis = []
        for repr in self.representations:
            change_of_basis.append(repr.change_of_basis)
        return sparse.block_diag(change_of_basis)

    @property
    def change_of_basis_inv(self) -> sparse.coo_matrix:
        r"""
        Inverse of the (sparse) change of basis matrix. See :attr:`escnn.nn.FieldType.change_of_basis` for more details.
        
        Returns:
            the inverted change of basis

        """
        change_of_basis_inv = []
        for repr in self.representations:
            change_of_basis_inv.append(repr.change_of_basis_inv)
        return sparse.block_diag(change_of_basis_inv)

    def get_dense_change_of_basis(self) -> torch.FloatTensor:
        """
        The method returns a dense :class:`torch.Tensor` containing a copy of the change-of-basis matrix.
        
        .. seealso ::
            See :attr:`escnn.nn.FieldType.change_of_basis` for more details.

        """
        return torch.FloatTensor(self.change_of_basis.todense())

    def get_dense_change_of_basis_inv(self) -> torch.FloatTensor:
        """
        The method returns a dense :class:`torch.Tensor` containing a copy of the inverse of the
        change-of-basis matrix.
        
        .. seealso ::
            See :attr:`escnn.nn.FieldType.change_of_basis` for more details.

        """
        return torch.FloatTensor(self.change_of_basis_inv.todense())
    
    def fiber_representation(self, element: GroupElement) -> torch.Tensor:
    
        assert element.group == self.fibergroup
    
        representation = []
        for repr in self.representations:
            representation.append(repr(element))
        representation = sparse.block_diag(representation).todense()
    
        representation = torch.tensor(representation)
        return representation

    def transform_fibers(self, input: torch.Tensor, element: GroupElement) -> torch.Tensor:
        r"""

        Transform the feature vectors of the input tensor according to the group representation associated to
        the input element.

        Interpreting the tensor as a vector-valued signal :math:`f: X \to \mathbb{R}^c` over a base space :math:`X`
        (where :math:`c` is the number of channels of the tensor), given the input ``element`` :math:`g \in G`
        (:math:`G` fiber group) the method returns the new signal :math:`f'`:

        .. math ::
            f'(x) := \rho(g) f(x)

        for :math:`x \in X` point in the base space and :math:`\rho` the representation of :math:`G` in the
        field type of this tensor.


        Notice that the input element has to be an element of the fiber group of this tensor's field type.

        .. seealso ::

            See :meth:`escnn.nn.FieldType.transform` to transform the whole tensor.

        Args:
            input (torch.Tensor): the tensor to transform
            element (GroupElement): an element of the group of symmetries of the fiber.

        Returns:
            the transformed tensor

        """
        representation = self.fiber_representation(element).to(dtype=input.dtype, device=input.device)
        # .contiguous() seems necessary here; if the array is not contiguous some operations have unexpected behaviors
        return torch.einsum("oi,bi...->bo...", representation, input).contiguous()

    def transform(self, input: torch.Tensor, element: GroupElement, coords: torch.Tensor = None, order: int = 2) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""

        The method takes a PyTorch's tensor, compatible with this type (i.e. whose spatial dimensions are
        supported by the base space and whose number of channels equals the :attr:`escnn.nn.FieldType.size`
        of this type), and an element of the fiber group of this type.
        
        Transform the input tensor according to the group representation associated with the input element
        and its (induced) action on the base space.
        
        This transformation includes both an action over the basespace (e.g. a rotation of the points on the plane)
        and a transformation of the channels by left-multiplying them with a representation of the fiber group.

        The method takes as input a tensor (``input``) and an ``element`` of the fiber
        group. The tensor ``input`` is the feature field to be transformed and needs to be compatible with the G-space
        and the representation (i.e. its number of channels equals the size of that representation).
        ``element`` needs to belong to the fiber group: check :meth:`escnn.group.GroupElement.group`.
        This method returns a transformed tensor through the action of ``element``.
        
        In addition, the method accepts an optional `coords` tensor.
        If the argument is not passed, the `input` tensor is assumed to have shape
        `(batchsize, channels, *spatial_grid_shape)` and to represent features sampled on a grid of shape
        `spatial_grid_shape`; in this case,  the action on the base space resamples the transformed features on this
        grid (using interpolation, if necessary).
        If `coords` is not `None`, `input` is assumed to be a `(#points, channels)` tensor containing an unstructured
        set of points living on the base space; then, `coords` should contain the coordinates of these points.
        The base space action will then transform these coordinates (no interpolation required).
        In that case, the method returns a pair containing both the transformed features (according to the action on the
        fibers) and the transformed coordinates (according to the action on the basespace).


        More precisely, given an input tensor, interpreted as an :math:`c`-dimensional signal
        :math:`f: \R^D \to \mathbb{R}^c` defined over the base space :math:`\R^D`, a representation
        :math:`\rho: G \to \mathbb{R}^{c \times c}` of :math:`G` and an element :math:`g \in G` of the fiber group,
        the method returns the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := \rho(g) f(g^{-1} x)

        .. note ::

            Mathematically, this method transforms the input with the **induced representation** from the input ``repr``
            (:math:`\rho`) of the symmetry group (:math:`G`) to the *total space* (:math:`P`), i.e.
            with :math:`Ind_{G}^{P} \rho`.
            For more details on this, see
            `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_ or
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.

        .. warning ::
            In case `coords` is not passed and, therefore, the resampling of the grid is performed, the input tensor is
            detached before the transformation, therefore no gradient is propagated back through this operation.

        .. seealso ::

            See :meth:`escnn.nn.GeometricTensor.transform_fibers` to transform only the fibers, i.e. not transform
            the base space.

            See :meth:`escnn.gspaces.GSpace._interpolate_transform_basespace` for more details on the action on the
            base space.


        Args:
            input (torch.Tensor): input tensor
            element (GroupElement): element of the fiber group
            coords (torch.Tensor, optional): coordinates of the points in `input`. If `None` (by default), it assumes
                                             the points `input` are arranged in a grid and it transforms the grid by
                                             interpolation. Otherwise, it transforms the coordinates in `coords` using
                                             `self.gspace.basespace_action()`. In the last case, the method returns a
                                             tuple `(transformed_input, transformed_coords)`.

        Returns:
            transformed tensor and, optionally, the transformed coordinates

        """
        
        assert element.group == self.fibergroup

        output = self.transform_fibers(input, element)

        if coords is None:
            output_np = output.detach().to('cpu').numpy()
            transformed = self.gspace._interpolate_transform_basespace(output_np, element, order=order)
            transformed = np.ascontiguousarray(transformed).astype(output_np.dtype)
            return torch.from_numpy(transformed).to(device=input.device)
        
        else:
            assert len(coords.shape) == 2, coords.shape
            assert len(input.shape) == 2, input.shape
            assert coords.shape[1] == self.gspace.dimensionality, \
                f"Error! `coords` tensor with shape {coords.shape} does not match the dimensionality of the field type {self.gspace.dimensionality}."
            assert input.shape[0] == coords.shape[0], \
                f"Error! Points in the `input` and `coords` tensors do not match: {input.shape[0]} != {coords.shape[0]}."
            
            representation = torch.tensor(
                self.gspace.basespace_action(element)
            ).to(dtype=coords.dtype, device=coords.device)
            
            # .contiguous() seems necessary here; if the array is not contiguous some operations have unexpected behaviors
            transformed_coords = torch.einsum("oi,pi->po", representation, coords).contiguous()
            
            return output, transformed_coords

    def restrict(self, id) -> 'FieldType':
        r"""
        
        Reduce the symmetries modeled by the :class:`~escnn.nn.FieldType` by choosing a subgroup of its fiber group as
        specified by ``id``. This implies a restriction of each representation in
        :attr:`escnn.nn.FieldType.representations` to this subgroup.
 
        .. seealso ::
        
            Check the documentation of the :meth:`~escnn.gspaces.GSpace.restrict` method in the subclass of
            :class:`~escnn.gspaces.GSpace` used for a description of the parameter ``id``.

        Args:
            id: identifier of the subgroup to which the :class:`~escnn.nn.FieldType` and its
                :attr:`escnn.nn.FieldType.representations` should be restricted

        Returns:
            the restricted type

        """
    
        # build the subgroup
        subspace, _, _ = self.gspace.restrict(id)
    
        # restrict each different base representation in the fiber representation
        restricted_reprs = {}
        for r in self._unique_representations:
            restricted_reprs[r.name] = self.gspace.fibergroup.restrict_representation(id, r)
    
        # for each field, retrieve the corresponding restricted representation
        fields = [restricted_reprs[r.name] for r in self.representations]
    
        # build the restricted fiber representation
        rrepr = subspace.type(*fields)
    
        return rrepr

    def sorted(self) -> 'FieldType':
        r"""

        Return a new field type containing the fields of the current one sorted by their dimensionalities.
        It is built from the :attr:`escnn.nn.FieldType.representations` of this field type sorted.

        Returns:
            the sorted field type

        """
        keys = [(r.size, i) for i, r in enumerate(self.representations)]
    
        keys = sorted(keys)
    
        permutation = [k[1] for k in keys]
    
        return self.index_select(permutation)

    def __add__(self, other: 'FieldType') -> 'FieldType':
        r"""

        Returns a field type associate with the *direct sum* :math:`\rho = \rho_1 \oplus \rho_2` of the representations
        :math:`\rho_1` and :math:`\rho_2` of two field types.
        
        In practice, the method builds a new :class:`~escnn.nn.FieldType` using the concatenation of the lists
        :attr:`escnn.nn.FieldType.representations` of the two field types.
        
        The two field types need to be associated with the same :class:`~escnn.gspaces.GSpace`.

        Args:
            other (FieldType): the other addend

        Returns:
            the direct sum

        """
    
        assert self.gspace == other.gspace
    
        return FieldType(self.gspace, self.representations + other.representations)

    def __len__(self) -> int:
        r"""

        Return the number of feature fields in this :class:`~escnn.nn.FieldType`, i.e. the length of
        :attr:`escnn.nn.FieldType.representations`.
        
        .. note ::
            This is in general different from :attr:`escnn.nn.FieldType.size`.

        Returns:
            the number of fields in this type

        """
        return len(self.representations)

    def fields_names(self) -> List[str]:
        r"""
        Return an ordered list containing the names of the representation associated with each field.

        Returns:
            the list of fields' representations' names

        """
        return [r.name for r in self.representations]

    def index_select(self, index: List[int]) -> 'FieldType':
        r"""
        
        Build a new :class:`~escnn.nn.FieldType` from the current one by taking the
        :class:`~escnn.group.Representation` s selected by the input ``index``.
        
        Args:
            index (list): a list of integers in the range ``{0, ..., N-1}``, where ``N`` is the number of representations
                          in the current field type

        Returns:
            the new field type
            
            
        """
        assert max(index) < len(self.representations)
        assert min(index) >= 0

        # retrieve the fields in the input representation to build the output representation
        representations = [self.representations[i] for i in index]
        return FieldType(self.gspace, representations)

    @property
    def fields_end(self) -> np.ndarray:
        r"""
        
            Array containing the index of the first channel following each field.
            More precisely, the integer in the :math:`i`-th position is equal to the index of the last channel of
            the :math:`i`-th field plus :math:`1`.
        
        """
        if self._field_end is None:
            field_idx = []
            p = 0
            for r in self.representations:
                p += r.size
                field_idx.append(p)
            self._field_end = np.array(field_idx, dtype=np.uint64)
    
        return self._field_end

    @property
    def fields_start(self) -> np.ndarray:
        r"""

            Array containing the index of the first channel of each field.
            More precisely, the integer in the :math:`i`-th position is equal to the index of the first channel of
            the :math:`i`-th field.

        """
        if self._field_start is None:
            field_idx = []
            p = 0
            for r in self.representations:
                field_idx.append(p)
                p += r.size
            self._field_start = np.array(field_idx, dtype=np.uint64)
            
        return self._field_start

    def group_by_labels(self, labels: List[str]) -> Dict[str, 'FieldType']:
        r"""
        
        Associate a label to each feature field (or representation in :attr:`escnn.nn.FieldType.representations`)
        and group them accordingly into new :class:`~escnn.nn.FieldType` s.
 
        Args:
            labels (list): a list of strings with length equal to the number of representations in
                           :attr:`escnn.nn.FieldType.representations`

        Returns:
            a dictionary mapping each different input label to a new field type

        """
        assert len(labels) == len(self)
        
        fields = defaultdict(lambda: [])
        
        for c, l in enumerate(labels):
            # append the index of the current field to the list of fields belonging to this label
            fields[l].append(c)
        
        # for each label, build the field type of the sub-fiber on which it acts
        types = {}
    
        for l in labels:
            # retrieve the sub-fiber corresponding to this label
            types[l] = self.index_select(fields[l])
        
        return types

    @property
    def uniform(self) -> bool:
        r"""
            Whether this FieldType contains only copies of the same representation, i.e. if all the elements of
            :attr:`~escnn.nn.FieldType.representations` are the same :class:`escnn.group.Representation`.
        """
        return self._uniform

    def __iter__(self):
        r"""
        
        It is possible to iterate over all :attr:`~escnn.nn.FieldType.representations` in a field type by using
        :class:`~escnn.nn.FieldType` as an *iterable* object.

        """
        return iter(self.representations)

    def __eq__(self, other):
        if isinstance(other, FieldType):
            return self.gspace == other.gspace and self.representations == other.representations
        else:
            return False
    
    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        summarized_representations = [
            (k, len(list(g)))
            for k, g in groupby([r.name for r in self.representations])
        ]

        return '[' + self.gspace.name + ': {' + ', '.join([f'{k} (x{n})' for k, n in summarized_representations]) + '}' + f'({self.size})]'
        # return '[' + self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}]'

    @property
    def testing_elements(self):
        r"""
        Alias for ``self.gspace.testing_elements``.
        
        .. seealso::
            :attr:`escnn.gspaces.GSpace.testing_elements` and
            :attr:`escnn.group.Group.testing_elements`
        
        """
        return self.gspace.testing_elements

    def __call__(self, tensor: torch.Tensor, coords: torch.Tensor = None) -> 'escnn.nn.GeometricTensor':
        return escnn.nn.GeometricTensor(tensor, self, coords)

class FourierFieldType(FieldType):
    """
    A field type that is compatible with Fourier transforms.
    """

    def __init__(
            self,
            gspace: GSpace,
            channels: int,
            bl_irreps: List,
            *,
            subgroup_id: Optional[Tuple] = None,
            unpack=False
    ):
        r"""
        A ``FieldType`` that is compatible with the Fourier transform modules.

        More specifically, this is a field type that is guaranteed to use only
        spectral regular representations.  Feature vectors transformed by such 
        representations can be interpreted as the coefficients of a 
        band-limited set of Fourier basis vectors.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): the number of band-limited spectral regular representations that comprise each fiber.
            irreps (list): list of irreps' ids to construct the band-limited representation
            subgroup_id (tuple): ...
            unpack (bool): Whether to treat the representation as a single entity (True) or as an set of irreps (False).  This affect nonlinearities like `GatedNonLinearity1`.
        
        Attributes:
            ~.gspace (GSpace)
            ~.representations (tuple)
            ~.size (int): dimensionality of the feature space described by the :class:`~escnn.nn.FieldType`.
                          It corresponds to the sum of the dimensionalities of the individual feature fields or
                          group representations (:attr:`escnn.group.Representation.size`).

        Example:

            >>> gspace = rot3DonR3()
            >>> so3 = gspace.fibergroup
            >>> in_type = FourierFieldType(gspace, 10, so3.bl_irreps(2))
        """
        self.channels = channels
        self.bl_irreps = bl_irreps
        self.subgroup_id = subgroup_id
        self.rho = make_fourier_representation(
                gspace.fibergroup,
                bl_irreps,
                subgroup_id,
        )

        if unpack:
            rho = [gspace.fibergroup.irrep(*n) for n in self.rho.irreps]
        else:
            rho = [self.rho]

        super().__init__(gspace, rho * channels)


def make_fourier_representation(group, bl_irreps, subgroup_id=None):
    if subgroup_id is None:
        return group.spectral_regular_representation(*bl_irreps, name=None)
    else:
        return group.spectral_quotient_representation(subgroup_id, *bl_irreps, name=None)


