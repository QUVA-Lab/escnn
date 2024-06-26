
import torch

from torch import Tensor

from escnn.group import GroupElement
from .field_type import FieldType

from typing import List, Union

import itertools
from collections.abc import Iterable


__all__ = ["GeometricTensor", "tensor_directsum"]


class GeometricTensor:
    
    def __init__(self, tensor: Tensor, type: FieldType, coords: Tensor = None):
        r"""

        A GeometricTensor can be interpreted as a *typed* tensor.
        It is wrapping a common :class:`torch.Tensor` and endows it with a (compatible) :class:`~escnn.nn.FieldType` as
        *transformation law*.
        
        The :class:`~escnn.nn.FieldType` describes the action of a group :math:`G` on the tensor.
        This action includes both a transformation of the base space and a transformation of the channels according to
        a :math:`G`-representation :math:`\rho`.
        
        All *escnn* neural network operations have :class:`~escnn.nn.GeometricTensor` s as inputs and outputs.
        They perform a dynamic typechecking, ensuring that the transformation laws of the data and the operation match.
        See also :class:`~escnn.nn.EquivariantModule`.
 
        As usual, the first dimension of the tensor is interpreted as the batch dimension. The second is the fiber
        (or channel) dimension, which is associated with a group representation by the field type. The following
        dimensions are the spatial dimensions (like in a conventional CNN).
        
        In addition, the method accepts an optional ``coords`` tensor.
        If the argument ``coords`` is not passed, the `input` tensor is assumed to have shape
        ``(batchsize, channels, *spatial_grid_shape)`` and to represent features sampled on a grid of shape
        ``spatial_grid_shape``; in this case,  the action on the base space resamples the transformed features on this
        grid (using interpolation, if necessary).
        If ``coords`` is not ``None``, ``input`` is assumed to be a ``(#points, channels)`` tensor containing an unstructured
        set of points living on the base space; then, ``coords`` should contain the coordinates of these points.
        The base space action will then transform these coordinates (no interpolation required).
        In that case, the method returns a pair containing both the transformed features (according to the action on the
        fibers) and the transformed coordinates (according to the action on the basespace).

        In addition, this class accepts an optional argument ``coords``.
        If the argument is not passed, the input ``tensor`` is assumed to have shape
        ``(batchsize, channels, *spatial_grid_shape)`` and to represent features sampled on a grid of shape
        ``spatial_grid_shape``; in this case,  the action on the base space resamples the transformed features on this
        grid (using interpolation, if necessary).
        If ``coords`` is not ``None``, ``tensor`` is assumed to be a ``(#points, channels)`` tensor containing an unstructured
        set of points living on the base space; then, `coords` should contain the coordinates of these points.
        The tensor ``coords`` must have shape ``(#points, type.gspace.dimensionality)``, where ``#points = tensor.shape[0]``.
        The base space action will then transform these coordinates (no interpolation required).
        See also the method :meth:`escnn.nn.FieldType.transform` for more details.
        
        The operations of **addition** and **scalar multiplication** are supported.
        For example::
        
            gs = escnn.gspaces.rot2dOnR2(8)
            type = escnn.nn.FieldType(gs, [gs.regular_repr]*3)
            t1 = escnn.nn.GeometricTensor(torch.randn(1, 24, 3, 3), type)
            t2 = escnn.nn.GeometricTensor(torch.randn(1, 24, 3, 3), type)
            
            # addition
            t3 = t1 + t2
            
            # scalar product
            t3 = t1 * 3.
            
            # scalar product also supports tensors containing only one scalar
            t3 = t1 * torch.tensor(3.)
            
            # inplace operations are also supported
            t1 += t2
            t2 *= 3.
        
        .. warning ::
            The multiplication of a PyTorch tensor containing only a scalar with a GeometricTensor is only supported
            when using PyTorch 1.4 or higher (see this `issue <https://github.com/pytorch/pytorch/issues/26333>`_ )
            
        .. warning ::
            These operations are only supported for *compatible* tensors, i.e. tensors which share the same ``type`` and,
            if set, ``coords``.
            
        A GeometricTensor can also be right-multiplied with a :class:`~escnn.group.GroupElement` with the ``@`` operator.
        The group element must belong to the :attr:`~escnn.nn.FieldType.fibergroup` of the :class:`~escnn.nn.FieldType`
        ``type`` of this tensor.
        The right-multiplication through ``@`` *only transforms the channels* using the group representation
        ``type.representation`` but *does not transform the base space*.
        This operation is equivalent to :meth:`~escnn.nn.GeometricTensor.transform_fibers`.
        Check its documentation for more details.
        
        
        A GeometricTensor supports **slicing** in a similar way to PyTorch's :class:`torch.Tensor`.
        More precisely, slicing along the batch (1st) and the spatial (3rd, 4th, ...) dimensions works as usual.
        However, slicing the fiber (2nd) dimension would break equivariance when splitting channels belonging to
        the same field.
        To prevent this, slicing on the second dimension is defined over *fields* instead of channels.
        Note that, if ``coords`` is not `None`, slicing over the first dimension also slices the ``coords`` tensor over its
        first dimension.
        Moreover, a GeometricTensor also partially supports **advanced indexing** (see NumPy's
        documentation about
        `indexing <https://numpy.org/doc/stable/user/basics.indexing.html#indexing-on-ndarrays>`_
        for more details).

        .. warning ::
            In contrast to NumPy and PyTorch, an index containing a single integer value **does not** reduce
            the dimensionality of the tensor.
            In this way, the resulting tensor can always be interpreted as a GeometricTensor.
            
        
        We give few examples to illustrate this behavior::
        
            # Example of GeometricTensor slicing
            space = escnn.gspaces.rot2dOnR2(4)
            type = escnn.nn.FieldType(space, [
                # field type            # index # size
                space.regular_repr,     #   0   #  4
                space.regular_repr,     #   1   #  4
                space.irrep(1),         #   2   #  2
                space.irrep(1),         #   3   #  2
                space.trivial_repr,     #   4   #  1
                space.trivial_repr,     #   5   #  1
                space.trivial_repr,     #   6   #  1
            ])                          #   sum = 15
            
            # this FieldType contains 7 fields
            len(type)
            >> 7
            
            # the size of this FieldType is equal to the sum of the sizes of each of its fields
            type.size
            >> 15
            
            geom_tensor = escnn.nn.GeometricTensor(torch.randn(10, type.size, 9, 9), type)

            # or, equivalently:
            geom_tensor = type(torch.randn(10, type.size, 9, 9))

            geom_tensor.shape
            >> torch.Size([10, 15, 9, 9])
            
            geom_tensor[1:3, :, 2:5, 2:5].shape
            >> torch.Size([2, 15, 3, 3])
            
            geom_tensor[..., 2:5].shape
            >> torch.Size([10, 15, 9, 3])
            
            # the tensor contains the fields 1:4, i.e 1, 2 and 3
            # these fields have size, respectively, 4, 2 and 2
            # so the resulting tensor has 8 channels
            geom_tensor[:, 1:4, ...].shape
            >> torch.Size([10, 8, 9, 9])
            
            # the tensor contains the fields 0:6:2, i.e 0, 2 and 4
            # these fields have size, respectively, 4, 2 and 1
            # so the resulting tensor has 7 channels
            geom_tensor[:, 0:6:2].shape
            >> torch.Size([10, 7, 9, 9])
            
            # the tensor contains only the field 2, which has size 2
            # note, also, that even though a single index is used for the batch dimension, the resulting tensor
            # still has 4 dimensions
            geom_tensor[3, 2].shape
            >> torch.Size(1, 2, 9, 9)

            # we can use a boolean tensor to perform advanced indexing over the fields of a Geometric Tensor.
            # the tensor contains only the two fields of size 2, i.e. the third and the fourth.
            # moreover, we also slice the batch dimension.
            idx = torch.tensor([type.representations[i].size == 2 for i in range(len(type))])
            geom_tensor[0:4, idx].shape
            >> torch.Size(4, 4, 9, 9)

            # it is also possible to use an integer tensor to perform advanced indexing.
            # Here, we select the second and the sixth fields.
            idx = torch.tensor([1, 5])
            geom_tensor[0:4, idx].shape
            >> torch.Size(4, 5, 9, 9)

            # advanced indexing over the other dimensions works as usual.
            idx = torch.tensor([1, 2, 2, 8, 4])
            geom_tensor[idx, 1:3].shape
            >> torch.Size(5, 6, 9, 9)

        
        .. warning ::
        
            *Slicing* over the fiber (2nd) dimension with ``step > 1`` or with a negative step is converted
            into *indexing* over the channels.
            This means that, in these cases, slicing behaves like *advanced indexing* in PyTorch and NumPy
            **returning a copy instead of a view**.
            For more details, see the *note* `here <https://pytorch.org/docs/stable/tensor_view.html>`_ and
            *NumPy*'s `docs <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_ .
        
        .. note ::
    
            Slicing is not supported for setting values inside the tensor
            (i.e. :meth:`~object.__setitem__` is not implemented).
            Indeed, depending on the values which are assigned, this operation can break the symmetry of the tensor
            which may not transform anymore according to its transformation law (specified by ``type``).
            In case this feature is necessary, one can directly access the underlying :class:`torch.Tensor`, e.g.
            ``geom_tensor.tensor[:3, :, 2:5, 2:5] = torch.randn(3, 4, 3, 3)``, although this is not recommended.

        .. warning ::
            Using advanced indexing over multiple axes simultaneously drops the corresponding dimensions (except for the
            first one). This means that the resulting tensor will likely not have a shape compatible with a Geometric
            Tensor (e.g. the spatial dimensions are dropped), causing an error.
            
            
        Args:
            tensor (torch.Tensor): the tensor data
            type (FieldType): the type of the tensor, modeling its transformation law
            coords (torch.Tensor, optional): a tensor containing the coordinates of the datapoints

        Attributes:
            ~.tensor (torch.Tensor)
            ~.type (FieldType)
            ~.coords (torch.Tensor)

        """
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(type, FieldType)
        
        if coords is None:
            assert len(tensor.shape) == 2 + type.gspace.dimensionality, (tensor.shape, type.gspace.dimensionality)
        else:
            assert len(coords.shape) == 2, coords.shape

            assert coords.shape[1] == type.gspace.dimensionality, \
                f"Error! `coords` tensor with shape {coords.shape} does not match the dimensionality of the field type {type.gspace.dimensionality}."

            assert len(tensor.shape) == 2, \
                f"Error! If `coords` is not `None`, the tensor must be a 2D tensor of shape (#points, filedtype size) but the tensor has shape {tensor.shape}."

            assert tensor.shape[0] == coords.shape[0],\
                f"Error! Points in the `tensor` and `coords` tensors do not match: {tensor.shape[0]} != {coords.shape[0]}."

        assert tensor.shape[1] == type.size, \
            f"Error! The size of the tensor {tensor.shape} does not match the size of the field type {type.size}."

        # torch.Tensor: PyTorch tensor containing the data
        self.tensor = tensor

        # torch.Tensor: PyTorch tensor containing the coordinates of the data points
        self.coords = coords

        # FieldType: field type of the signal
        self.type = type
    
    def restrict(self, id) -> 'GeometricTensor':
        r"""
        Restrict the field type of this tensor.
        
        The method returns a new :class:`~escnn.nn.GeometricTensor` whose :attr:`~escnn.nn.GeometricTensor.type`
        is equal to this tensor's :attr:`~escnn.nn.GeometricTensor.type`
        restricted to a subgroup :math:`H<G` (see :meth:`escnn.nn.FieldType.restrict`).
        The restricted :attr:`~escnn.nn.GeometricTensor.type` is associated with the restricted representation
        :math:`\Res{H}{G}\rho` of the :math:`G`-representation :math:`\rho` associated to this tensor's
        :attr:`~escnn.nn.GeometricTensor.type`.
        The input ``id`` specifies the subgroup :math:`H < G`.
        
        Notice that the underlying :attr:`~escnn.nn.GeometricTensor.tensor` instance will be shared between
        the current tensor and the returned one.
        
        .. warning ::
        
            The method builds the new representation on the fly; hence, if this operation is needed at run time,
            we suggest to use :class:`escnn.nn.RestrictionModule` which pre-computes the new representation offline.
        
        .. seealso ::
        
            Check the documentation of the :meth:`~escnn.gspaces.GSpace.restrict` method in the
            :class:`~escnn.gspaces.GSpace` instance used for a description of the parameter ``id``.
        
        
        Args:
            id: the id identifying the subgroup :math:`H` the representations are restricted to

        Returns:
            the geometric tensor with the restricted representations
            
        """
        new_class = self.type.restrict(id)
        return GeometricTensor(self.tensor, new_class, self.coords)
    
    def split(self, breaks: List[int]):
        r"""
        
        Split this tensor on the channel dimension in a list of smaller tensors.
        The original tensor is split at the *fields* specified by the index list ``breaks``.
        
        If the tensor is associated with the list of fields :math:`\{\rho_i\}_i`
        (see :attr:`escnn.nn.FieldType.representations`), the :math:`j`-th output tensor will contain the fields
        :math:`\rho_{\text{breaks}[j-1]}, \dots, \rho_{\text{breaks}[j]-1}` of the original tensor.
        
        If `breaks = None`, the tensor is split at each field.
        This is equivalent to using `breaks = list(range(len(self.type)))`.
        
        Example ::
        
            space = escnn.gspaces.rot2dOnR2(4)
            type = escnn.nn.FieldType(space, [
                space.regular_repr,     # size = 4
                space.regular_repr,     # size = 4
                space.irrep(1),         # size = 2
                space.irrep(1),         # size = 2
                space.trivial_repr,     # size = 1
                space.trivial_repr,     # size = 1
                space.trivial_repr,     # size = 1
            ])                          #  sum = 15
            
            type.size
            >> 15
            
            geom_tensor = escnn.nn.GeometricTensor(torch.randn(10, type.size, 7, 7), type)
            
            geom_tensor.shape
            >> torch.Size([10, 15, 7, 7])
            
            # split the tensor in 3 parts
            len(geom_tensor.split([0, 4, 6]))
            >> 3
            
            # the first contains
            # - the first 2 regular fields (2*4 = 8 channels)
            # - 2 vector fields (irrep(1)) (2*2 = 4 channels)
            # and, therefore, contains 12 channels
            geom_tensor.split([0, 4, 6])[0].shape
            >> torch.Size([10, 12, 7, 7])
            
            # the second contains only 2 scalar (trivial) fields (2*1 = 2 channels)
            geom_tensor.split([0, 4, 6])[1].shape
            >> torch.Size([10, 2, 7, 7])
            
            # the last contains only 1 scalar (trivial) field (1*1 = 1 channels)
            geom_tensor.split([0, 4, 6])[2].shape
            >> torch.Size([10, 1, 7, 7])
            
        
        Args:
            breaks (list): indices of the fields where to split the tensor

        Returns:
            list of :class:`~escnn.nn.GeometricTensor` s into which the original tensor is chunked
            
        """
        if breaks is None:
            breaks = list(range(len(self.type)))
            
        breaks.append(len(self.type))
        
        # final list of tensors
        tensors = []
        
        # list containing the index of the channels separating consecutive fields in this tensor
        positions = []
        last = 0
        for repr in self.type.representations:
            positions.append(last)
            last += repr.size
        positions.append(last)
        
        last_field = 0
        # for each break point
        for b in breaks:
            assert b > last_field, 'Error! "breaks" must be an increasing list of positive indexes'
            
            # compute the sub-class of the new sub-tensor
            repr = FieldType(self.type.gspace, self.type.representations[last_field: b])
            
            # retrieve the sub-tensor
            data = self.tensor[:, positions[last_field]:positions[b], ...]
            
            tensors.append(GeometricTensor(data, repr, self.coords))
            
            last_field = b
        
        return tensors

    def transform(self, element: GroupElement) -> 'GeometricTensor':
        r"""
        Transform the current tensor according to the group representation associated to the input element
        and its induced action on the base space

        .. warning ::
            The input tensor is detached before the transformation therefore no gradient is backpropagated
            through this operation

            See :meth:`escnn.nn.GeometricTensor.transform_fibers` to transform only the fibers, i.e. not transform
            the base space.

        Args:
            element (GroupElement): an element of the group of symmetries of the fiber.

        Returns:
            the transformed tensor

        """

        if self.coords is None:
            transformed = self.type.transform(self.tensor, element, coords=None)
            coords = None
        else:
            transformed, coords = self.type.transform(self.tensor, element, self.coords)

        return GeometricTensor(transformed, self.type, coords)

    def transform_fibers(self, element: GroupElement) -> 'GeometricTensor':
        r"""
        
        Transform the feature vectors of the underlying tensor according to the group representation associated to
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
        
            See :meth:`escnn.nn.GeometricTensor.transform` to transform the whole tensor.
        
        Args:
            element (GroupElement): an element of the group of symmetries of the fiber.

        Returns:
            the transformed tensor

        """
        
        transformed_tensor = self.type.transform_fibers(self.tensor, element)
        return GeometricTensor(transformed_tensor, self.type, self.coords)
    
    @property
    def shape(self):
        r"""
        Alias for ``self.tensor.shape``

        """
        return self.tensor.shape

    def size(self):
        r"""
        Alias for ``self.tensor.size()``

        .. seealso ::
            :meth:`torch.Tensor.size`

        """
        return self.tensor.size()

    def to(self, *args, **kwargs):
        r"""
        
        Alias for ``self.tensor.to(*args, **kwargs)``.
        
        Applies :meth:`torch.Tensor.to` to the underlying tensor and wraps the resulting tensor in a new
        :class:`~escnn.nn.GeometricTensor` with the same type.
        
        .. warning ::
            This method does not affect ``self.coords``.

        """
        tensor = self.tensor.to(*args, **kwargs)
        return GeometricTensor(tensor, self.type, self.coords)

    def __getitem__(self, slices) -> 'GeometricTensor':
        r'''
        
        A GeometricTensor supports *slicing* in a similar way to PyTorch's :class:`torch.Tensor`.
        More precisely, slicing along the batch (1st) and the spatial (3rd, 4th, ...) dimensions works as usual.
        However, slicing along the channel dimension could break equivariance by splitting the channels belonging to the
        same field.
        For this reason, slicing on the second dimension is not defined over the channels but over fields.
        
        When a continuous (step=1) slice is used over the fields/channels dimension (the 2nd axis), it is converted
        into a continuous slice over the channels.
        This is not possible when the step is greater than 1 or negative.
        In such cases, the slice over the fields needs to be converted into an index over the channels.
        
        Moreover, when a single integer is used to index an axis, that axis is not discarded as in PyTorch but is
        preserved with size 1.

        If slicing is performed over the batch dimension and `coords` is not `None`, also the `coords` tensor is sliced.
        
        Slicing is not supported for setting values inside the tensor (i.e. :meth:`object.__setitem__`
        is not implemented).
        
        '''
        
        # Slicing is not supported on the channel dimension.
        if isinstance(slices, tuple):
            if len(slices) > len(self.tensor.shape):
                raise TypeError(
                    f'''
                        Error! Too many slicing indices for GeometricTensor.
                        At most {len(self.tensor.shape)} indices expected but {len(slices)} indices passed.
                    '''
                )
        else:
            slices = (slices,)
            
        for i, idx in enumerate(slices):
            if (isinstance(idx, slice) or isinstance(idx, int) or idx == Ellipsis):
                continue
            elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                continue
            elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
                continue
            else:
                raise TypeError(f'''
                        Error! Advanced Indexing over a GeometricTensor is not fully supported yet.
                        Currently, only indexing with boolean or long tensors and basic slicing are supported.
                ''')
        
        naxes = len(self.tensor.shape)
        
        # count the number of indexes passed
        indexed_axes = 0
        for idx in slices:
            indexed_axes += 1 - (idx == Ellipsis)
        
        # number of axes which are missing an index
        missing_axes = naxes - indexed_axes
        
        # expand the first ellipsis with a number of full slices (i.e. [::]) equal to the number
        # of axes not indexed. Discard all other ellipses
        expanded_idxs = []
        expanded_ellipsis = False
        for s in slices:
            if s == Ellipsis:
                # expand only the first ellipsis
                if not expanded_ellipsis:
                    expanded_idxs += [slice(None)]*missing_axes
                    expanded_ellipsis = True
            else:
                # other indices are preserved
                expanded_idxs.append(s)
        
        # maximum index per dimension
        idx_max = list(self.tensor.shape)
        idx_max[1] = len(self.type)
        
        # If an index containing a single integer is passed, it is converted into a slice
        # which starts at that index and ends at the following one.
        # In this way, when passing a single integer to index a dimension, the resulting tensor will still have that
        # dimension with size 1
        for i in range(len(expanded_idxs)):
            if isinstance(expanded_idxs[i], int):
                idx = expanded_idxs[i]
                if idx < 0:
                    # convert a negative index into a positive index
                    idx = idx_max[i] + idx
                expanded_idxs[i] = slice(idx, idx+1, 1)
        
        if len(expanded_idxs) == 1:
            # if only the first dimension is indexed, there is no need to do anything
            # the resulting tensor will have the same type of the original as the indexing does not affect the
            # channels/fields dimension
            type = self.type
            
        elif isinstance(expanded_idxs[1], slice) and (expanded_idxs[1].step is None or expanded_idxs[1].step == 1):
            # If the index over the fields is a slice and it is contiguous, we can convert it into a
            # contiguous slice over the channels.
            # The slice will start from the first channel of the first field and will stop at the last channel
            # of the last field
            start = expanded_idxs[1].start if expanded_idxs[1].start is not None else 0
            stop = expanded_idxs[1].stop if expanded_idxs[1].stop is not None else len(self.type)
            channel_idxs = slice(
                self.type.fields_start[start],
                self.type.fields_end[stop-1],
                1
            )
            
            if start == 0 and stop == len(self.type):
                # if all the fields are retrieved by this index, the resulting tensor has the same field
                # types of the original one
                type = self.type
            else:
                # otherwise, only a subset of the fields are preserved
                type = FieldType(self.type.gspace, self.type.representations[expanded_idxs[1]])
                
            expanded_idxs[1] = channel_idxs

        else:
            # If the index over the fields is not a slice or it is not a contiguous slice, we need to convert it
            # into an index over the channels. We first use the index provided to retrieve the list of fields
            # and then add the index of their channels in a list of indexes
            idxs = []

            # convert the indices into iterable and retrieve the subset of field representations
            if isinstance(expanded_idxs[1], slice):
                fields = range(len(self.type))[expanded_idxs[1]]
                representations = self.type.representations[expanded_idxs[1]]
            elif isinstance(expanded_idxs[1], int):
                fields = [expanded_idxs[1]]
                representations = self.type.representations[expanded_idxs[1]]
            elif isinstance(expanded_idxs[1], torch.Tensor) and expanded_idxs[1].dtype == torch.bool:
                fields = [i for i in range(len(self.type)) if expanded_idxs[1][i]]
                representations = [self.type.representations[f] for f in fields]
            elif isinstance(expanded_idxs[1], torch.Tensor) and expanded_idxs[1].dtype == torch.long:
                fields = expanded_idxs[1].tolist()
                representations = [self.type.representations[f] for f in fields]
            elif isinstance(expanded_idxs[1], Iterable):
                fields = expanded_idxs[1]
                representations = [self.type.representations[f] for f in fields]
            else:
                raise ValueError('Index over the fiber (2nd) dimension not recognized.')

            # iterate over all fields indexed by the user
            for field in fields:
                # append the indexes of the channels in the field
                idxs.append(list(
                    range(
                        self.type.fields_start[field],
                        self.type.fields_end[field],
                        1
                    )
                ))

            # only a subset of the fields are preserved by this index
            type = FieldType(self.type.gspace, representations)
            
            # concatenate all the channel indexes
            channel_idxs = list(itertools.chain(*idxs))
            expanded_idxs[1] = channel_idxs

        idxs = tuple(expanded_idxs)
        
        sliced_tensor = self.tensor[idxs]

        if self.coords is not None:
            coords = self.coords[idxs[0], :]
        else:
            coords = None

        return GeometricTensor(sliced_tensor, type, coords)
    
    def has_same_coords(self, other: 'GeometricTensor') -> bool:
        if self.coords is other.coords:
            # this includes also the case both are None
            return True
        
        if (self.coords is None) != (other.coords is None):
            return False
        elif self.coords is not None:
            return torch.allclose(self.coords, other.coords)
        return True

    def is_compatible(self, other: 'GeometricTensor') -> bool:
        if self.type != other.type:
            return False
        return self.has_same_coords(other)

    def __add__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Add two compatible :class:`~escnn.nn.GeometricTensor` using pointwise addition.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            the sum

        """
        if isinstance(other, GeometricTensor):
            assert self.is_compatible(other), 'The two geometric tensor must have the same `type` and `coords`'
            return GeometricTensor(self.tensor + other.tensor, self.type, self.coords)
        else:
            return NotImplemented

    def __sub__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Subtract two compatible :class:`~escnn.nn.GeometricTensor` using pointwise subtraction.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            their difference

        """
        if isinstance(other, GeometricTensor):
            assert self.is_compatible(other), 'The two geometric tensor must have the same `type` and `coords`'
            return GeometricTensor(self.tensor - other.tensor, self.type, self.coords)
        else:
            return NotImplemented

    def __iadd__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Add a compatible :class:`~escnn.nn.GeometricTensor` to this tensor inplace.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            this tensor

        """
        if isinstance(other, GeometricTensor):
            assert self.is_compatible(other), 'The two geometric tensor must have the same `type` and `coords`'
            self.tensor += other.tensor
            return self
        else:
            return NotImplemented

    def __isub__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Subtract a compatible :class:`~escnn.nn.GeometricTensor` to this tensor inplace.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            this tensor

        """
        if isinstance(other, GeometricTensor):
            assert self.is_compatible(other), 'The two geometric tensor must have the same `type` and `coords`'
            self.tensor -= other.tensor
            return self
        else:
            return NotImplemented

    def __mul__(self, other: Union[float, torch.Tensor]) -> 'GeometricTensor':
        r"""
        Scalar product of this :class:`~escnn.nn.GeometricTensor` with a scalar.
        
        The scalar can be a float number or a :class:`torch.Tensor` containing only
        one scalar (i.e. :func:`torch.numel` should return `1`).

        Args:
            other : a scalar

        Returns:
            the scalar product

        """
        if isinstance(other, float) or (isinstance(other, torch.Tensor) and other.numel() == 1):
            # Only multiplication with a scalar is allowed
            return GeometricTensor(self.tensor * other, self.type, self.coords)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: Union[float, torch.Tensor]) -> 'GeometricTensor':
        r"""
        Scalar product of this :class:`~escnn.nn.GeometricTensor` with a scalar.
        The operation is done inplace.

        The scalar can be a float number of a :class:`torch.Tensor` containing only
        one scalar (i.e. :func:`torch.numel` should return `1`).

        Args:
            other : a scalar

        Returns:
            the scalar product

        """
        if isinstance(other, float) or (isinstance(other, torch.Tensor) and other.numel() == 1):
            self.tensor *= other
            return self
        else:
    
            return NotImplemented

    def __rmatmul__(self, other: GroupElement):
        r"""
        
        Equivalent to :meth:`escnn.nn.GeometricTensor.transform_fibers`.
        
        .. warning::
            This only transforms the fibers, not the basespace.
        
        .. todo ::
            should this be like `.transform(element)` ?

        """
        
        if isinstance(other, GroupElement):
            assert other.group == self.type.fibergroup
            return GeometricTensor(
                self.type.transform_fibers(self.tensor, other),
                self.type,
                self.coords
            )
        else:
            return NotImplemented

    def __repr__(self):
        t = repr(self.tensor)[:-1]
        t = t.replace('\n', '\n  ')
        r = 'g_' + t + ', ' + repr(self.type) + ')'

        return r


def tensor_directsum(tensors: List['GeometricTensor']) -> 'GeometricTensor':
    r"""
    Concatenate a list of :class:`~escnn.nn.GeometricTensor` s on the channels dimension (``dim=1``).
    The input tensors have to be compatible: they need to have the same shape except for the channels
    dimension (``dim=1``); additionally, if their `coords` attribute is not `None`, they also need to share the same
    value.
    
    In the resulting :class:`~escnn.nn.GeometricTensor`, the channels dimension will be associated with the direct sum
    representation of the representations of the input tensors.
    
    .. seealso::
        
        :func:`escnn.group.directsum`
    
    Args:
        tensors (list): a list of :class:`~escnn.nn.GeometricTensor` s

    Returns:
        the direct sum of the inputs
        
    """
    # assert len(tensors) > 1
    
    for i in range(1, len(tensors)):
        assert tensors[0].type.gspace == tensors[i].type.gspace
        assert tensors[0].tensor.ndimension() == tensors[i].tensor.ndimension()
        assert tensors[0].tensor.shape[0] == tensors[i].tensor.shape[0]
        assert tensors[0].tensor.shape[2:] == tensors[i].tensor.shape[2:]
        assert tensors[0].has_same_coords(tensors[i])
    
    # concatenate all representations from all field types
    reprs = []
    for t in tensors:
        reprs += t.type.representations
    
    # build the new field type
    cls = FieldType(tensors[0].type.gspace, reprs)
    
    # concatenate the underlying tensors
    data = torch.cat([t.tensor for t in tensors], dim=1)
    
    # build the new Geometric Tensor
    return GeometricTensor(data, cls, tensors[0].coords)

