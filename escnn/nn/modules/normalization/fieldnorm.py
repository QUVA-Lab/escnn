from collections import defaultdict

from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import Parameter
from typing import List, Tuple, Any, Union
import numpy as np

__all__ = [
    "FieldNorm",
]


class FieldNorm(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 affine: bool = True,
                 ):
        r"""

        Normalization module which normalizes each field individually.
        The statistics are only computed over the channels within a single field (not over the batch dimension or
        the spatial dimensions).
        Moreover, this layer does not track running statistics and uses only the current input, so it behaves similarly
        at train and eval time.

        For each individual field, the mean is given by the projection on the subspaces transforming under the trivial
        representation while the variance is the squared norm of the field, after the mean has been subtracted.

        If ``affine = True``, a single scale is learnt per input field and the bias is applied only to the
        trivial irreps (this scale and bias are shared over the spatial dimensions in order to preserve equivariance).

        .. warning::
            If a field is only containing trivial irreps, this layer will just set its values to zero and, possibly,
            replace them with a learnable bias if ``affine = True``.


        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``

        """

        assert isinstance(in_type.gspace, GSpace)

        super(FieldNorm, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.affine = affine

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
                _indices[name] = [min(_indices[name]), max(_indices[name]) + 1]
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

        # for each different representation in the input type
        for r in self.in_type._unique_representations:
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

            if self.affine:
                # scale all dimensions of the same field by the same weight
                weight = Parameter(torch.ones((self._nfields[r.name], 1)), requires_grad=True)
                self.register_parameter(f'{self._escape_name(name)}_weight', weight)
                if self._has_trivial[name]:
                    # the bias is applied only to the trivial channels
                    bias = Parameter(torch.zeros((self._nfields[r.name], len(trivials))), requires_grad=True)
                    self.register_parameter(f'{self._escape_name(name)}_bias', bias)

        self.eps = eps

    def reset_parameters(self):
        if self.affine:
            for name, size in self._sizes:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
                weight.data.fill_(1)
                if hasattr(self, f"{self._escape_name(name)}_bias"):
                    bias = getattr(self, f"{self._escape_name(name)}_bias")
                    bias.data.fill_(0)

    def reset_running_stats(self):
        pass

    def _estimate_stats(self, slice, name: str):

        if self._has_trivial[name]:
            P = getattr(self, f'{self._escape_name(name)}_avg')

            # compute the mean
            means = torch.einsum(
                'ij,bcj...->bci...',
                P,
                slice.detach()
            )
            centered = slice - means
        else:
            means = None
            centered = slice

        # Center the data and compute the variance
        # N.B.: we implicitly assume the dimensions to be iid,
        # i.e. the covariance matrix is a scalar multiple of the identity
        # vars = centered.var(dim=2, unbiased=False, keepdim=True).detach()
        vars = (centered**2).mean(dim=2, keepdim=True).detach()

        return means, vars

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Normalize the input feature map

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

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

            means, vars = self._estimate_stats(slice, name)

            if self._has_trivial[name]:
                # center data by subtracting the mean
                slice = slice - means

            # normalize dividing by the std and multiply by the new scale
            if self.affine:
                weight = getattr(self, f"{self._escape_name(name)}_weight").view(1, self._nfields[name], 1, *(1,)*len(spatial_dims))
            else:
                weight = 1.

            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()
            # scales[vars < self.eps] = 0

            # print(name, size, indices, self._has_trivial[name])
            # print(slice.shape, scales.shape)
            # if not self.training:
            #     np.set_printoptions(precision=5, suppress=True, threshold=1000000, linewidth=3000)
            #     print(scales.detach().cpu().numpy().reshape(scales.shape[0], -1).T)

            # scale features
            slice = slice * scales

            # shift the features with the learnable bias
            if self.affine and self._has_trivial[name]:
                bias = getattr(self, f"{self._escape_name(name)}_bias")
                Q = getattr(self, f'{self._escape_name(name)}_change_of_basis')
                slice = slice + torch.einsum(
                    'ij,cj->ci',
                    Q,
                    bias
                ).view(1, bias.shape[0], Q.shape[0], *(1,) * len(spatial_dims))

            # needed for PyTorch's adaptive mixed precision
            slice = slice.to(output.dtype)

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
        return '{in_type}, eps={eps}, affine={affine}' \
            .format(**self.__dict__)

    def export(self):
        raise NotImplementedError()

