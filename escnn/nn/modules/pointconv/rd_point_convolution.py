
from abc import ABC, abstractmethod

from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.group import Representation
from escnn.kernels import KernelBasis, EmptyBasisException
from escnn.gspaces import *

from ..equivariant_module import EquivariantModule

from escnn.nn.modules.basismanager import BlocksBasisSampler

from typing import Callable, Tuple, Dict, Union

import torch

import torch_geometric

from torch.nn import Parameter
import numpy as np
import math


__all__ = ["_RdPointConv"]


class _RdPointConv(torch_geometric.nn.MessagePassing, EquivariantModule, ABC):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 d: int,
                 groups: int = 1,
                 bias: bool = True,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 ):

        r"""

        Abstract class which implements a general G-steerable convolution, mapping between the input and output
        :class:`~escnn.nn.FieldType` s specified by the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^d\rtimes G` where :math:`G` is the
        :attr:`escnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.

        This class implements convolution with steerable filters over sparse planar geometric graphs.
        Instead, :class:`~escnn.nn.modules.conv._RdConv` implements an equivalent convolution layer over a pixel/voxel
        grid. See the documentation of :class:`~escnn.nn.modules.conv._RdConv` for more details about equivariance and
        steerable convolution.

        The input of this module is a geometric graph, i.e. a graph whose nodes are associated with ``d``-dimensional
        coordinates in :math:`\R^d`.
        The nodes' coordinates should be stored in the ``coords`` attribute of the input
        :class:`~escnn.nn.GeometricTensor`.
        The adjacency of the graph should be passed as a second input tensor ``edge_index``, like commonly done in
        :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`.
        See :meth:`~escnn.nn.modules.pointconv._RdPointConv.forward`.

        In each forward pass, the module computes the relative coordinates of the points on the edges and samples each
        filter in the basis of G-steerable kernels at these relative locations.
        The basis filters are expanded using the learnable weights and used to perform convolution over the graph in the
        message passing framework.
        Optionally, the relative coordinates can be pre-computed and passed in the input ``edge_delta`` tensor.

        .. note ::
            In practice, we first apply the basis filters on the input features and then combine the responses via the
            learnable weights. See also :meth:`~escnn.nn.modules.basismanager.BlocksBasisSampler.compute_messages`.

        .. warning ::

            When :meth:`~torch.nn.Module.eval()` is called, the bias is built with the current trained weights and stored
            for future reuse such that no overhead of expanding the bias remains.

            When :meth:`~torch.nn.Module.train()` is called, the attribute
            :attr:`~escnn.nn.modules.pointconv._RdPointConv.expanded_bias` is discarded to avoid situations of mismatch
            with the learnable expansion coefficients. See also :meth:`escnn.nn.modules.pointconv._RdPointConv.train`.

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        .. warning ::
            We don't support ``groups > 1`` yet.
            We include this parameter for future compatibility.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.

        Attributes:

            ~.weights (torch.Tensor): the learnable parameters which are used to expand the kernel
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~escnn.nn.modules.pointconv._RdPointConv.bias`

        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GSpace)
        assert d >= in_type.gspace.dimensionality

        super(_RdPointConv, self).__init__(aggr='mean')

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type

        assert self.space.dimensionality == self.d
        
        self.groups = groups

        if groups > 1:
            # Check the input and output classes can be split in `groups` groups, all equal to each other
            # first, check that the number of fields is divisible by `groups`
            assert len(in_type) % groups == 0
            assert len(out_type) % groups == 0
            in_size = len(in_type) // groups
            out_size = len(out_type) // groups
            
            # then, check that all groups are equal to each other, i.e. have the same types in the same order
            assert all(in_type.representations[i] == in_type.representations[i % in_size] for i in range(len(in_type)))
            assert all(out_type.representations[i] == out_type.representations[i % out_size] for i in range(len(out_type)))
            
            # finally, retrieve the type associated to a single group in input.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from one input
            # group to all output groups. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            in_type = in_type.index_select(list(range(in_size)))
        
        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis
            
            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irrep(*irr).is_trivial():
                        trivials += 1
            
            # if there is at least 1 trivial irrep
            if trivials > 0:
                
                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = torch.zeros(self.out_type.size, trivials)
                
                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irrep(*irr)
                        if irr.is_trivial():
                            bias_expansion[p:p+r.size, c] = torch.tensor(r.change_of_basis[:, pi])
                            c += 1
                        pi += irr.size
                    p += r.size
                
                self.register_buffer("bias_expansion", bias_expansion)
                self.bias = Parameter(torch.zeros(trivials), requires_grad=True)
                self.register_buffer("expanded_bias", torch.zeros(out_type.size))
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        # TODO support `groups` arg for conv
        # if groups != 1:
        #     raise NotImplementedError(f'`groups !=1` not supported yet!')

        # BlocksBasisSampler: submodule which takes care of building the filter
        self._basissampler = BlocksBasisSampler(in_type.representations, out_type.representations,
                                                    self._build_kernel_basis,
                                                    basis_filter=basis_filter,
                                                    recompute=recompute)

        if self.basissampler.dimension() == 0:
            raise ValueError('''
                The basis for the steerable filter is empty!
                Tune the `frequencies_cutoff`, `kernel_size`, `rings` or `basis_filter` parameters to allow
                for a larger basis.
            ''')

        self.weights = Parameter(torch.zeros(self.basissampler.dimension()), requires_grad=True)

    @abstractmethod
    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        raise NotImplementedError

    @property
    def basissampler(self) -> BlocksBasisSampler:
        r"""

        Submodule which takes care of sampling the steerable filters.

        It is used to sample the G-steerable basis on the relative coordinates along the edges of a geometric graph and,
        then, expand the kernel in the sampled basis using the learned ``weights``.
        See also :meth:`~escnn.nn.modules.basismanager.BlockBasisSampler.forward`.

        In practice, this submodule is also used to directly compute the messages via
        :meth:`~escnn.nn.modules.basismanager.BlockBasisSampler.compute_messages`: first, the basis filters are applied
        on the input features and, then, the responses are combined using the learnable weights.

        """
        return self._basissampler

    def expand_bias(self) -> torch.Tensor:
        r"""

        Expand the bias in terms of :class:`~escnn.nn._RdPointConv.bias`.

        Returns:
            the expanded bias

        """
        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias

        return _bias

    def expand_filter(self, points: Union[torch.Tensor, Dict[Tuple, torch.Tensor]]) -> torch.Tensor:
        r"""

        Expand the filter in terms of :class:`~escnn.nn._RdPointConv.weights`.

        Returns:
            the expanded filter sampled on the input points

        """

        return self.basissampler(self.weights, points)

    def expand_parameters(self, points: Union[torch.Tensor, Dict[Tuple, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Expand the filter in terms of the :attr:`~escnn.nn._RdConv.weights` and the
        expanded bias in terms of :class:`~escnn.nn._RdConv.bias`.

        Returns:
            the expanded filter and bias

        """

        _filter = self.expand_filter(points)
        _bias = self.expand_bias()

        return _filter, _bias

    def forward(self, x: GeometricTensor, edge_index: torch.Tensor, edge_delta: torch.Tensor = None):
        r"""
        Convolve the input with the expanded filter and bias.

        This method is based on PyTorch Geometric's :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`,
        i.e. it uses :meth:`~torch_geometric.nn.conv.message_passing.MessagePassing.propagate` to send the messages
        computed in :meth:`~escnn.nn.modules.pointconv._RdPointConv.message`.

        The input tensor ``input`` represents a feature field over the nodes of a geometric graph.
        Hence, the ``coords`` attribute of ``input`` should contain the ``d``-dimensional coordinates of each node (see
        :class:`~escnn.nn.GeometricTensor`).

        The tensor ``edge_index`` must be a :class:`torch.LongTensor` of shape ``(2, m)``, representing ``m`` edges.

        Mini-batches containing multiple graphs can be constructed as in
        `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html>`_ by merging the
        graphs in a unique, disconnected, graph.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``.
            edge_index (torch.Tensor): tensor representing the connectivity of the graph.
            edge_delta (torch.Tensor, optional): the relative coordinates of the nodes on each edge. If not passed, it
                    is automatically computed using ``input.coords`` and ``edge_index``.

        Returns:
            output feature field transforming according to ``out_type``

        """
        assert isinstance(x, GeometricTensor)
        assert x.type == self.in_type

        assert len(edge_index.shape) == 2
        assert edge_index.shape[0] == 2

        if edge_delta is None:
            pos = x.coords
            row, cols = edge_index
            edge_delta = pos[row] - pos[cols]

        out = self.propagate(edge_index, x=x.tensor, edge_delta=edge_delta)

        if not self.training:
            _bias = self.expanded_bias
        else:
            # retrieve the bias
            _bias = self.expand_bias()

        if _bias is not None:
            out += _bias

        out = GeometricTensor(out, self.out_type, coords=x.coords)
        
        return out

    def message(self, x_j: torch.Tensor, edge_delta: torch.Tensor=None) -> torch.Tensor:
        r"""

        This methods computes the message from the input node ``j`` to the output node ``i`` of each edge in
        ``edge_index``.

        The message is equal to the product of the filter evaluated on the relative coordinate along an edge with
        the feature vector on the input node of the edge.

        """
        return self.basissampler.compute_messages(self.weights, x_j, edge_delta, conv_first=True, groups=self.groups)

    def train(self, mode=True):
        r"""

        If ``mode=True``, the method sets the module in training mode and discards the
        :attr:`~escnn.nn._RdPointConv.expanded_bias` attribute.

        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the bias
        using the current values of the trainable parameters and store it :attr:`~escnn.nn._RdConv.expanded_bias`
        such that it is not recomputed at each forward pass.

        .. warning ::

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.

        """

        if mode:
            if hasattr(self, "expanded_bias"):
                del self.expanded_bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`

            _bias = self.expand_bias()

            if _bias is not None:
                self.register_buffer("expanded_bias", _bias)
            else:
                self.expanded_bias = None

        return super(_RdPointConv, self).train(mode)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 3
        assert input_shape[1] == self.in_type.size

        return input_shape[0], self.out_type.size, input_shape[2]

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        main_str = self._get_name() + '('
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(extra_lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self):
        s = ('{in_type}, {out_type}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def check_equivariance(self, atol: float = 1e-5, rtol: float = 5e-6, assertion: bool = True, verbose: bool = True):
    
        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)
    
        P = 30
    
        pos = torch.randn(P, self.d, device=self.weights.device)
        x = torch.randn(P, self.in_type.size, device=self.weights.device)
        x = GeometricTensor(x, self.in_type, pos)
    
        distance = torch.norm(pos.unsqueeze(1) - pos, dim=2, keepdim=False)
    
        thr = sorted(distance.view(-1).tolist())[int(P**2//16)]
        edge_index = torch.nonzero(distance < thr).T.contiguous()

        errors = []
    
        for el in self.space.testing_elements:
        
            out1 = self(x, edge_index).transform(el).tensor.cpu().detach().numpy()
            out2 = self(x.transform(el), edge_index).tensor.cpu().detach().numpy()
        
            errs = np.abs(out1 - out2)
        
            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1
        
            relerr = errs / esum
        
            # if verbose:
            #     print(el)
            #     print(relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())
        
            tol = rtol * esum + atol
        
            if np.any(errs > tol) and verbose:
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])
        
            if assertion:
                assert np.all(
                    errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                    el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors


