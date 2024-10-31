
from collections import defaultdict


from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.utils import unique_ever_seen

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import Parameter
from typing import List, Tuple, Any, Union
import numpy as np

from abc import ABC, abstractmethod

__all__ = [
    "_IIDBatchNorm",
    "IIDBatchNorm1d",
    "IIDBatchNorm2d",
    "IIDBatchNorm3d",
]


class _IIDBatchNorm(EquivariantModule, ABC):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ):
    
        assert isinstance(in_type.gspace, GSpace)
        
        super(_IIDBatchNorm, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        self.track_running_stats = track_running_stats

        self._nfields = None
        
        # group fields by their type and
        #   - check if fields of the same type are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each type
        self._nfields = defaultdict(int)
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        ntrivials = 0
        position = 0
        last_field = None
        for i, r in enumerate(self.in_type.representations):
            
            for irr in r.irreps:
                if self.in_type.fibergroup.irrep(*irr).is_trivial():
                    ntrivials += 1
            
            if r.name != last_field:
                if not r.name in self._contiguous:
                    self._contiguous[r.name] = True
                else:
                    self._contiguous[r.name] = False

            last_field = r.name
            _indices[r.name] += list(range(position, position + r.size))
            self._nfields[r.name] += 1
            position += r.size
        
        for name, contiguous in self._contiguous.items():
            
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[name] = [min(_indices[name]), max(_indices[name])+1]
                setattr(self, f"{self._escape_name(name)}_indices", _indices[name])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[name] = torch.LongTensor(_indices[name])
                
                # register the indices tensors as parameters of this module
                self.register_buffer(f"{self._escape_name(name)}_indices", _indices[name])
        
        # store the size of each field type
        self._sizes = []
        
        # store for each field type the indices of the trivial irreps in it
        self._trivial_idxs = {}

        # store for each field type the sizes and the indices of all its irreps, grouped by their size
        self._irreps_sizes = {}
        
        self._has_trivial = {}

        # for each different representation in the input type.

        # It's important to ensure that we iterate through the representations
        # in the same order every time the program runs.  This order becomes
        # the order that the various batch norm parameters are passed to the
        # optimizer, and if that order changes between runs, then it becomes
        # impossible to resume training from checkpoints [1].
        
        # Practically, this means that we can't use a set to (more succinctly)
        # eliminate duplicate representations.  Set iteration order is not only
        # arbitrary, but also non-deterministic, because python salts the hash
        # values of some common types to protect against DOS attacks [2].
        #
        # [1]: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
        # [2]: https://stackoverflow.com/questions/3848091/set-iteration-order-varies-from-run-to-run 
        
        for r in unique_ever_seen(self.in_type.representations):
            p = 0
            trivials = []
            
            # mask containing the location of the trivial irreps in the irrep decomposition of the representation
            S = np.zeros((r.size, r.size))
            
            # find all trivial irreps occurring in the representation
            for i, irr in enumerate(r.irreps):
                irr = self.in_type.fibergroup.irrep(*irr)
                if irr.is_trivial():
                    trivials.append(p)
                    S[p, p] = 1.
                
                p += irr.size
            
            name = r.name
            self._sizes.append((name, r.size))
            
            self._has_trivial[name] = len(trivials) > 0
            
            if self._has_trivial[name]:
                # averaging matrix which computes the expectation of a input vector, i.e. projects it in the trivial
                # subspace by masking out all non-trivial irreps
                P = r.change_of_basis @ S @ r.change_of_basis_inv
                self.register_buffer(f'{self._escape_name(name)}_avg', torch.tensor(P, dtype=torch.float))
            
                Q = torch.tensor(r.change_of_basis, dtype=torch.float)[:, trivials]
                self.register_buffer(f'{self._escape_name(name)}_change_of_basis', Q)
                
                running_mean = torch.zeros((self._nfields[r.name], r.size), dtype=torch.float)
                self.register_buffer(f'{self._escape_name(name)}_running_mean', running_mean)

            # assume all dimensions have same variance, i.e. the covariance matrix is a scalar multiple of the identity
            running_var = torch.ones((self._nfields[r.name], 1), dtype=torch.float)
            self.register_buffer(f'{self._escape_name(name)}_running_var', running_var)
            
            if self.affine:
                # scale all dimensions of the same field by the same weight
                weight = Parameter(torch.ones((self._nfields[r.name], 1)), requires_grad=True)
                self.register_parameter(f'{self._escape_name(name)}_weight', weight)
                if self._has_trivial[name]:
                    # the bias is applied only to the trivial channels
                    bias = Parameter(torch.zeros((self._nfields[r.name], len(trivials))), requires_grad=True)
                    self.register_parameter(f'{self._escape_name(name)}_bias', bias)
            
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.eps = eps
        self.momentum = momentum

    def reset_running_stats(self):
        for name, size in self._sizes:
            if hasattr(self, f"{self._escape_name(name)}_running_var"):
                running_var = getattr(self, f"{self._escape_name(name)}_running_var")
                running_var.fill_(1)
                
            if hasattr(self, f"{self._escape_name(name)}_running_mean"):
                running_mean = getattr(self, f"{self._escape_name(name)}_running_mean")
                running_mean.fill_(0)
                
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for name, size in self._sizes:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
                weight.data.fill_(1)
                if hasattr(self, f"{self._escape_name(name)}_bias"):
                    bias = getattr(self, f"{self._escape_name(name)}_bias")
                    bias.data.fill_(0)
    
    def _estimate_stats(self, slice, name: str):

        agg_axes = (0,) + tuple(range(3, len(slice.shape)))

        if self._has_trivial[name]:
            P = getattr(self, f'{self._escape_name(name)}_avg')

            # compute the mean
            means = torch.einsum(
                'ij,bcj...->bci...',
                P,
                slice.mean(dim=agg_axes, keepdim=True).detach()
            )
            centered = slice - means
            means = means.reshape(means.shape[1], means.shape[2])
        else:
            means = None
            centered = slice
    
        # Center the data and compute the variance
        # N.B.: we implicitly assume the dimensions to be iid,
        # i.e. the covariance matrix is a scalar multiple of the identity
        # TODO: this should compute the mean squared norm, not the var since we have already subtracted the theoretically invariant mean
        vars = centered.var(dim=agg_axes, unbiased=True, keepdim=False).mean(dim=1, keepdim=True).detach()

        return means, vars
    
    def _get_running_stats(self, name: str):
        vars = getattr(self, f"{self._escape_name(name)}_running_var")
        if self._has_trivial[name]:
            means = getattr(self, f"{self._escape_name(name)}_running_mean")
        else:
            means = None
            
        return means, vars

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Apply norm non-linearities to the input feature map
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_dims = input.shape[2:]

        output = torch.empty_like(input)
        
        # iterate through all field types
        for name, size in self._sizes:
            indices = getattr(self, f"{self._escape_name(name)}_indices")
            
            if self._contiguous[name]:
                slice = input[:, indices[0]:indices[1], ...]
            else:
                slice = input[:, indices, ...]

            slice = slice.view(b, -1, size, *spatial_dims)
            
            if not self.track_running_stats:
                means, vars = self._estimate_stats(slice, name)
                
            elif self.training:
                means, vars = self._estimate_stats(slice, name)
                running_mean, running_var = self._get_running_stats(name)

                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars
                assert torch.allclose(running_var, getattr(self, f"{self._escape_name(name)}_running_var")), name

                if self._has_trivial[name]:
                    running_mean *= 1 - exponential_average_factor
                    running_mean += exponential_average_factor * means
                    assert torch.allclose(running_mean, getattr(self, f"{self._escape_name(name)}_running_mean")), name
                
            else:
                means, vars = self._get_running_stats(name)

            if self._has_trivial[name]:
                # center data by subtracting the mean
                slice = slice - means.view(1, means.shape[0], means.shape[1], *(1,)*len(spatial_dims))

            # normalize dividing by the std and multiply by the new scale
            if self.affine:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
            else:
                weight = 1.

            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()
            # scale features
            slice = slice * scales.view(1, scales.shape[0], scales.shape[1], *(1,)*len(spatial_dims))
            
            # shift the features with the learnable bias
            if self.affine and self._has_trivial[name]:
                bias = getattr(self, f"{self._escape_name(name)}_bias")
                Q = getattr(self, f'{self._escape_name(name)}_change_of_basis')
                slice = slice + torch.einsum(
                    'ij,cj->ci',
                    Q,
                    bias
                ).view(1, bias.shape[0], Q.shape[0], *(1,)*len(spatial_dims))

            if not self._contiguous[name]:
                output[:, indices, ...] = slice.view(b, -1, *spatial_dims)
            else:
                output[:, indices[0]:indices[1], ...] = slice.view(b, -1, *spatial_dims)

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) > 1, input_shape
        assert input_shape[1] == self.in_type.size, input_shape
    
        return (input_shape[0], self.out_type.size, *input_shape[2:])

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def _escape_name(self, name: str):
        return name.replace('.', '^')

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
        return '{in_type}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}' \
            .format(**self.__dict__)

    def _export(self, d: int):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNormNd` module and set to "eval" mode.

        """

        if not self.track_running_stats:
            raise ValueError('''
                Equivariant Batch Normalization can not be converted into conventional batch normalization when
                "track_running_stats" is False because the statistics contained in a single batch are generally
                not symmetric
            ''')

        self.eval()

        if d == 1:
            batchnorm_class = torch.nn.BatchNorm1d
        elif d == 2:
            batchnorm_class = torch.nn.BatchNorm2d
        elif d == 3:
            batchnorm_class = torch.nn.BatchNorm3d
        else:
            raise ValueError

        batchnorm = batchnorm_class(
            self.in_type.size,
            self.eps,
            self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats
        )

        batchnorm.num_batches_tracked.data = self.num_batches_tracked.data

        for name, size in self._sizes:
            indices = getattr(self, f'{self._escape_name(name)}_indices')
            running_mean, running_var = self._get_running_stats(name)
            n = self._nfields[name]

            if self._contiguous[name]:
                start, end = indices[0], indices[1]

                batchnorm.running_var.data[start:end] = running_var.data.view(n, 1).expand(n, size).reshape(-1)
                if self.affine:
                    weight = getattr(self, f'{self._escape_name(name)}_weight')
                    batchnorm.weight.data[start:end] = weight.data.view(n, 1).expand(n, size).reshape(-1)

                if self._has_trivial[name]:
                    batchnorm.running_mean.data[start:end] = running_mean.data.view(n, size).reshape(-1)
                    if self.affine:
                        bias = getattr(self, f'{self._escape_name(name)}_bias')
                        Q = getattr(self, f'{self._escape_name(name)}_change_of_basis')
                        bias = torch.einsum(
                            'ij,cj->ci',
                            Q,
                            bias
                        )
                        batchnorm.bias.data[start:end] = bias.data.view(n, size).reshape(-1)
                else:
                    batchnorm.running_mean.data[start:end] = 0.
                    if self.affine:
                        batchnorm.bias.data[start:end] = 0.

            else:

                batchnorm.running_var.data[indices] = running_var.data.view(n, 1).expand(n, size).reshape(-1)
                if self.affine:
                    weight = getattr(self, f'{self._escape_name(name)}_weight')
                    batchnorm.weight.data[indices] = weight.data.view(n, 1).expand(n, size).reshape(-1)

                if self._has_trivial[name]:
                    batchnorm.running_mean.data[indices] = running_mean.data.view(n, size).reshape(-1)
                    if self.affine:
                        bias = getattr(self, f'{self._escape_name(name)}_bias')
                        Q = getattr(self, f'{self._escape_name(name)}_change_of_basis')
                        bias = torch.einsum(
                            'ij,cj->ci',
                            Q,
                            bias
                        )
                        batchnorm.bias.data[indices] = bias.data.view(n, size).reshape(-1)
                else:
                    batchnorm.running_mean.data[indices] = 0.
                    if self.affine:
                        batchnorm.bias.data[indices] = 0.

        batchnorm.eval()

        return batchnorm

    @abstractmethod
    def export(self):
        pass

    @abstractmethod
    def _check_input_shape(self, shape: Tuple[int, ...]):
        pass


class IIDBatchNorm1d(_IIDBatchNorm):
    r"""

    Batch normalization for generic representations for 1D or 0D data (i.e. 3D or 2D inputs).

    This batch normalization assumes that all dimensions within the same field have the same variance, i.e. that
    the covariance matrix of each field in `in_type` is a scalar multiple of the identity.
    Moreover, the mean is only computed over the trivial irreps occurring in the input representations (the input
    representation does not need to be decomposed into a direct sum of irreps since this module can deal with the
    change of basis).

    Similarly, if ``affine = True``, a single scale is learnt per input field and the bias is applied only to the
    trivial irreps.

    This assumption is equivalent to the usual Batch Normalization in a Group Convolution NN (GCNN), where
    statistics are shared over the group dimension.
    See Chapter 4.2 at `https://gabri95.github.io/Thesis/thesis.pdf <https://gabri95.github.io/Thesis/thesis.pdf>`_ .

    Args:
        in_type (FieldType): the input field type
        eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
        momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
        affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                              when set to ``False``, it does not track such statistics but uses
                                              batch statistics in both training and eval modes.
                                              Default: ``True``

    """
    
    def _check_input_shape(self, shape: Tuple[int, ...]):
        if len(shape) != 2 or len(shape) != 3:
            raise ValueError('Error, expected a 2D or 3D tensor but a {} one was found'.format(len(shape)))
    
    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNorm2d` module and set to "eval" mode.

        """

        return self._export(d=1)


class IIDBatchNorm2d(_IIDBatchNorm):
    r"""

    Batch normalization for generic representations for 2D data (i.e. 4D inputs).
    
    This batch normalization assumes that all dimensions within the same field have the same variance, i.e. that
    the covariance matrix of each field in `in_type` is a scalar multiple of the identity.
    Moreover, the mean is only computed over the trivial irreps occurring in the input representations (the input
    representation does not need to be decomposed into a direct sum of irreps since this module can deal with the
    change of basis).
    
    Similarly, if ``affine = True``, a single scale is learnt per input field and the bias is applied only to the
    trivial irreps.
    
    This assumption is equivalent to the usual Batch Normalization in a Group Convolution NN (GCNN), where
    statistics are shared over the group dimension.
    See Chapter 4.2 at `https://gabri95.github.io/Thesis/thesis.pdf <https://gabri95.github.io/Thesis/thesis.pdf>`_ .

    Args:
        in_type (FieldType): the input field type
        eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
        momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
        affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                              when set to ``False``, it does not track such statistics but uses
                                              batch statistics in both training and eval modes.
                                              Default: ``True``

    """

    def _check_input_shape(self, shape: Tuple[int, ...]):
        if len(shape) != 4:
            raise ValueError('Error, expected a 4D tensor but a {} one was found'.format(len(shape)))

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNorm2d` module and set to "eval" mode.

        """

        return self._export(d=2)


class IIDBatchNorm3d(_IIDBatchNorm):
    r"""

    Batch normalization for generic representations for 3D data (i.e. 5D inputs).
    
    This batch normalization assumes that all dimensions within the same field have the same variance, i.e. that
    the covariance matrix of each field in `in_type` is a scalar multiple of the identity.
    Moreover, the mean is only computed over the trivial irreps occurring in the input representations (the input
    representation does not need to be decomposed into a direct sum of irreps since this module can deal with the
    change of basis).
    
    Similarly, if ``affine = True``, a single scale is learnt per input field and the bias is applied only to the
    trivial irreps.
    
    This assumption is equivalent to the usual Batch Normalization in a Group Convolution NN (GCNN), where
    statistics are shared over the group dimension.
    See Chapter 4.2 at `https://gabri95.github.io/Thesis/thesis.pdf <https://gabri95.github.io/Thesis/thesis.pdf>`_ .

    Args:
        in_type (FieldType): the input field type
        eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
        momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
        affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                              when set to ``False``, it does not track such statistics but uses
                                              batch statistics in both training and eval modes.
                                              Default: ``True``

    """

    def _check_input_shape(self, shape: Tuple[int, ...]):
        if len(shape) != 5:
            raise ValueError('Error, expected a 5D tensor but a {} one was found'.format(len(shape)))

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNorm2d` module and set to "eval" mode.

        """

        return self._export(d=3)

