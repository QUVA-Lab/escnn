
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

__all__ = ["GNormBatchNorm"]


class GNormBatchNorm(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ):
        r"""

        Batch normalization for generic representations.

        This batch normalization assumes that the covariance matrix of a subset of channels in `in_type` transforming
        under an irreducible representation is a scalar multiple of the identity.
        Moreover, the mean is only computed over the trivial irreps occurring in the input representations.
        These assumptions are necessary and sufficient conditions for the equivariance in expectation of this module,
        see Chapter 4.2 at `https://gabri95.github.io/Thesis/thesis.pdf <https://gabri95.github.io/Thesis/thesis.pdf>`_ .

        Similarly, if ``affine = True``, a single scale is learnt per input irrep and the bias is applied only to the
        trivial irreps.

        Note that the representations in the input field type do not need to be already decomposed into direct sums of
        irreps since this module can deal with changes of basis.

        .. warning::
            However, because the irreps in the input representations rarely appear in a contiguous way, this module might
            internally use advanced indexing, leading to some computational overhead.
            Modules like :class:`~escnn.nn.IIDBatchNorm2d` or :class:`~escnn.nn.IIDBatchNorm3d`, instead, share the same
            variance with all channels within the same field (and, therefore, over multiple irreps).
            This can be more efficient if the input field type contains multiple copies of a larger,
            reducible representation.


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

        assert isinstance(in_type.gspace, GSpace)
        
        super(GNormBatchNorm, self).__init__()

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
                if self.in_type.fibergroup._irreps[irr].is_trivial():
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

        for r in unique_ever_seen(self.in_type.representations):
            p = 0
            irreps = defaultdict(lambda: [])
            trivials = []
            aggregator = torch.zeros(r.size, len(r.irreps))
            
            for i, irr in enumerate(r.irreps):
                irr = self.in_type.fibergroup._irreps[irr]
                if irr.is_trivial():
                    trivials.append(p)
                
                aggregator[p:p+irr.size, i] = 1. / irr.size
                
                irreps[irr.size] += list(range(p, p+irr.size))
                p += irr.size
            
            propagator = (aggregator > 0).clone().to(dtype=torch.float)
            
            name = r.name
            
            self._trivial_idxs[name] = torch.tensor(trivials, dtype=torch.long)
            self._irreps_sizes[name] = [(s, idxs) for s, idxs in irreps.items()]
            self._sizes.append((name, r.size))
            
            if not np.allclose(r.change_of_basis, np.eye(r.size)):
                self.register_buffer(f'{self._escape_name(name)}_change_of_basis', torch.tensor(r.change_of_basis, dtype=torch.float))
                self.register_buffer(f'{self._escape_name(name)}_change_of_basis_inv', torch.tensor(r.change_of_basis_inv, dtype=torch.float))
            
            self.register_buffer(f'vars_aggregator_{self._escape_name(name)}', aggregator)
            self.register_buffer(f'vars_propagator_{self._escape_name(name)}', propagator)
        
            running_var = torch.ones((self._nfields[r.name], len(r.irreps)), dtype=torch.float)
            running_mean = torch.zeros((self._nfields[r.name], len(trivials)), dtype=torch.float)
            self.register_buffer(f'{self._escape_name(name)}_running_var', running_var)
            self.register_buffer(f'{self._escape_name(name)}_running_mean', running_mean)
            
            if self.affine:
                weight = Parameter(torch.ones((self._nfields[r.name], len(r.irreps))), requires_grad=True)
                bias = Parameter(torch.zeros((self._nfields[r.name], len(trivials))), requires_grad=True)
                self.register_parameter(f'{self._escape_name(name)}_weight', weight)
                self.register_parameter(f'{self._escape_name(name)}_bias', bias)
            
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.eps = eps
        self.momentum = momentum

    def reset_running_stats(self):
        for name, size in self._sizes:
            running_var = getattr(self, f"{self._escape_name(name)}_running_var")
            running_mean = getattr(self, f"{self._escape_name(name)}_running_mean")
            running_var.fill_(1)
            running_mean.fill_(0)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for name, size in self._sizes:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
                bias = getattr(self, f"{self._escape_name(name)}_bias")
                weight.data.fill_(1)
                bias.data.fill_(0)

    def _get_running_stats(self, name: str):
        vars = getattr(self, f"{self._escape_name(name)}_running_var")
        means = getattr(self, f"{self._escape_name(name)}_running_mean")
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
        shape = input.shape[2:]
        b, c = input.shape[:2]
        
        output = torch.empty_like(input)
        
        # iterate through all field types
        for name, size in self._sizes:
            indices = getattr(self, f"{self._escape_name(name)}_indices")
            
            if self._contiguous[name]:
                slice = input[:, indices[0]:indices[1], ...]
            else:
                slice = input[:, indices, ...]

            slice = slice.view(b, -1, size, *shape)
            
            if hasattr(self, f"{self._escape_name(name)}_change_of_basis_inv"):
                cob_inv = getattr(self, f"{self._escape_name(name)}_change_of_basis_inv")
                slice = torch.einsum("ds,bcsxy->bcdxy", (cob_inv, slice))

            if not self.track_running_stats:
                means, vars = self._compute_statistics(slice, name)
            elif self.training:
                
                # compute the mean and variance of the fields
                means, vars = self._compute_statistics(slice, name)
                
                running_mean, running_var = self._get_running_stats(name)

                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars
                
                running_mean *= 1 - exponential_average_factor
                running_mean += exponential_average_factor * means

                assert not torch.isnan(running_mean).any()
                assert not torch.isnan(running_var).any()
                assert torch.allclose(running_mean, getattr(self, f"{self._escape_name(name)}_running_mean"))
                assert torch.allclose(running_var, getattr(self, f"{self._escape_name(name)}_running_var"))
                
            else:
                means, vars = self._get_running_stats(name)

            if self.affine:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
            else:
                weight = 1.

            # helps in case mixed precision is used
            vars[vars < 0.] = 0.

            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()

            # compute the point shifts
            # shifts = bias - self._scale(means, scales, name=name)
            centered = self._shift(slice, -1*means, name=name, out=None)
            normalized = self._scale(centered, scales, name=name, out=None)
            
            if self.affine:
                bias = getattr(self, f"{self._escape_name(name)}_bias")
                normalized = self._shift(normalized, bias, name=name, out=None)
            
            if hasattr(self, f"{self._escape_name(name)}_change_of_basis"):
                cob = getattr(self, f"{self._escape_name(name)}_change_of_basis")
                normalized = torch.einsum("ds,bcsxy->bcdxy", (cob, normalized))
                
            if not self._contiguous[name]:
                output[:, indices, ...] = normalized.view(b, -1, *shape)
            else:
                output[:, indices[0]:indices[1], ...] = normalized.view(b, -1, *shape)

            # if self._contiguous[name]:
            #     slice2 = output[:, indices[0]:indices[1], ...]
            # else:
            #     slice2 = output[:, indices, ...]
            # assert torch.allclose(slice2.view(b, -1, size, h, w), slice), name
            
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def _compute_statistics(self, t: torch.Tensor, name: str):
        
        trivial_idxs = self._trivial_idxs[name]
        vars_aggregator = getattr(self, f"vars_aggregator_{self._escape_name(name)}")
        
        b, c, s = t.shape[:3]
        shape = t.shape[3:]

        l = trivial_idxs.numel()
        
        # number of samples in the tensor used to estimate the statistics
        M = (np.prod(shape) if shape else 1)
        N = b * M

        # compute the mean of the trivial fields
        trivial_means = t[:, :, trivial_idxs, ...].view(b, c, l, M).mean(dim=(0, 3), keepdim=False).detach()
        
        # compute the mean of squares of all channels
        vars = (t ** 2).view(b, c, s, M).mean(dim=(0, 3), keepdim=False).detach()
        
        # For the non-trivial fields the mean of the fields is 0, so we can compute the variance as the mean of the
        # norms squared.
        # For trivial channels, we need to subtract the squared mean
        vars[:, trivial_idxs] -= trivial_means**2

        # helps in case mixed precision is used
        vars[vars < 0.] = 0.

        # aggregate the squared means of the channels which belong to the same irrep
        vars = torch.einsum("io,ci->co", (vars_aggregator, vars))

        # helps in case mixed precision is used
        vars[vars < 0.] = 0.

        # Correct the estimation of the variance with Bessel's correction
        correction = N/(N-1) if N > 1 else 1.
        vars *= correction
        
        return trivial_means, vars

    def _scale(self, t: torch.Tensor, scales: torch.Tensor, name: str, out: torch.Tensor = None):
        
        if out is None:
            out = torch.empty_like(t)
        
        vars_aggregator = getattr(self, f"vars_propagator_{self._escape_name(name)}")
        
        ndims = len(t.shape[3:])
        scale_shape = (1, scales.shape[0], vars_aggregator.shape[0]) + (1,)*ndims
        # scale all fields
        out[...] = t * torch.einsum("oi,ci->co", (vars_aggregator, scales)).reshape(scale_shape)
        
        return out
    
    def _shift(self, t: torch.Tensor, trivial_bias: torch.Tensor, name: str, out: torch.Tensor = None):
    
        if out is None:
            out = t.clone()
        else:
            out[:] = t
            
        trivial_idxs = self._trivial_idxs[name]
        
        bias_shape = (1,) + trivial_bias.shape + (1,)*(len(t.shape) - 3)
        
        # add bias to the trivial fields
        out[:, :, trivial_idxs, ...] += trivial_bias.view(bias_shape)

        return out

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

    def export(self):
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

        batchnorm = torch.nn.BatchNorm3d(
            self.in_type.size,
            self.eps,
            self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats
        )

        batchnorm.num_batches_tracked.data = self.num_batches_tracked.data

        for name, size in self._sizes:
            contiguous = self._contiguous[name]
            if not contiguous:
                raise NotImplementedError(
                    '''Non-contiguous indices not supported yet when converting
                    inner-batch normalization into conventional BatchNorm2d'''
                )

            n = self._nfields[name]

            start, end = getattr(self, f"{self._escape_name(name)}_indices")

            running_mean, running_var = self._get_running_stats(name)

            batchnorm.running_mean.data[start:end] = self._shift(torch.zeros(
                1, n, size
            ), running_mean, name=name, out=None).view(-1)
            batchnorm.running_var.data[start:end] = self._scale(torch.ones(
                1, n, size
            ), running_var, name=name, out=None).view(-1)

            if self.affine:
                weight = getattr(self, f'{self._escape_name(name)}_weight')
                batchnorm.weight.data[start:end] = self._scale(torch.ones(
                    1, n, size
                ), weight, name=name, out=None).view(-1)

                bias = getattr(self, f"{self._escape_name(name)}_bias")
                batchnorm.bias.data[start:end] = self._shift(torch.zeros(
                    1, n, size
                ), bias, name=name, out=None).view(-1)

            if hasattr(self, f"{self._escape_name(name)}_change_of_basis"):
                cob = getattr(self, f"{self._escape_name(name)}_change_of_basis")

                batchnorm.running_var.data[start:end] = torch.einsum(
                    "ds,cs->cd",
                    cob,
                    batchnorm.running_var.data[start:end].view(n, size)
                ).view(-1)

        batchnorm.eval()

        return batchnorm


