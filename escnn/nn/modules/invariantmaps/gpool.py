
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn.modules.utils import indexes_from_labels

import torch
from torch import nn

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np


__all__ = ["GroupPooling", "MaxPoolChannels"]


class GroupPooling(EquivariantModule):
    
    def __init__(self, in_type: FieldType, **kwargs):
        r"""
        
        Module that implements *group pooling*.
        This module only supports permutation representations such as regular representation,
        quotient representation or trivial representation (though, in the last case, this module
        acts as identity).
        For each input field, an output field is built by taking the maximum activation within that field; as a result,
        the output field transforms according to a trivial representation.
        
        .. seealso::
            :attr:`~escnn.group.Group.regular_representation`,
            :attr:`~escnn.group.Group.quotient_representation`
        
        Args:
            in_type (FieldType): the input field type
            
        """
        assert isinstance(in_type.gspace, GSpace)
        
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities,\
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        super(GroupPooling, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        
        # build the output representation substituting each input field with a trivial representation
        self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))

        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(lambda: [])
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(lambda: [])

        # whether each group of fields is contiguous or not
        self._contiguous = {}

        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields
        indeces = indexes_from_labels(in_type, [r.size for r in in_type.representations])

        for s, (contiguous, fields, idxs) in indeces.items():
            self._contiguous[s] = contiguous
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _in_indices[s] = torch.LongTensor([min(idxs), max(idxs)+1])
                _out_indices[s] = torch.LongTensor([min(fields), max(fields)+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _in_indices[s] = torch.LongTensor(idxs)
                _out_indices[s] = torch.LongTensor(fields)
        
            # register the indices tensors as parameters of this module
            self.register_buffer('in_indices_{}'.format(s), _in_indices[s])
            self.register_buffer('out_indices_{}'.format(s), _out_indices[s])

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply Group Pooling to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        output = torch.empty(self.evaluate_output_shape(input.shape), device=input.device, dtype=torch.float)
        
        for s, contiguous in self._contiguous.items():
            
            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))
            
            if contiguous:
                fm = input[:, in_indices[0]:in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]
                
            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            
            max_activations, _ = torch.max(fm, 2)
            
            if contiguous:
                output[:, out_indices[0]:out_indices[1], ...] = max_activations
            else:
                output[:, out_indices, ...] = max_activations
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
    
        x = torch.randn(3, c, 10, 10)
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors

    def export(self):
        r"""
        Export this module to the pure PyTorch module :class:`~escnn.nn.MaxPoolChannels`
        and set to "eval" mode.
        
        .. warning ::
        
                Currently, this method only supports group pooling with feature types containing only representations
                of the same size.
        
        .. note ::
            
            Because there is no native PyTorch module performing this operation, it is not possible to export this
            module without any dependency with this library.
            Indeed, the resulting module is dependent on this library through the class
            :class:`~escnn.nn.MaxPoolChannels`.
            In case PyTorch will introduce a similar module in a future release, we will update this method to remove
            this dependency.
            
            Nevertheless, the :class:`~escnn.nn.MaxPoolChannels` module is slightly lighter
            than :class:`~escnn.nn.GroupPooling` as it does not perform any automatic type checking and does not wrap
            each tensor in a :class:`~escnn.nn.GeometricTensor`.
            Furthermore, the :class:`~escnn.nn.MaxPoolChannels` class is very simple and
            one can easily reimplement it to remove any dependency with this library after training the model.
            
        """
        
        if len(self._contiguous) > 1:
            raise NotImplementedError("""
                Group pooling with feature types containing representations of different sizes is not supported yet.
            """)
    
        self.eval()

        size = int(list(self._contiguous.keys())[0])
        gpool = MaxPoolChannels(size)
        
        return gpool.eval()
    
    def extra_repr(self):
        return '{in_type}'.format(**self.__dict__)


class MaxPoolChannels(nn.Module):
    
    def __init__(self, kernel_size: int):
        r"""
        
        Module that computes the maximum activation within each group of ``kernel_size`` consecutive channels.
        
        Args:
            kernel_size (int): the size of the group of channels the max is computed over
            
        """
        super(MaxPoolChannels, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        assert input.shape[1] % self.kernel_size == 0, '''
            Error! The input number of channels ({}) is not divisible by the max pooling kernel size ({})
        '''.format(input.shape[1], self.kernel_size)
        
        b = input.shape[0]
        c = input.shape[1] // self.kernel_size
        s = input.shape[2:]
        
        shape = (b, c, self.kernel_size) + s
        
        return input.view(shape).max(2)[0]

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)
