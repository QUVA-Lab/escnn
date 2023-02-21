
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch.nn.functional as F
import torch

from typing import List, Tuple, Any, Union

import math

__all__ = ["PointwiseMaxPool", "PointwiseMaxPoolAntialiased"]


class PointwiseMaxPool(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.MaxPool2D`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.
        
        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when True, will use ceil instead of floor to compute the output shape

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality in [2, 3]

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                f"""Error! Representation "{r.name}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""
        
        super(PointwiseMaxPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.space.dimensionality
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride,) * self.space.dimensionality
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding,) * self.space.dimensionality
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation,) * self.space.dimensionality
        else:
            self.dilation = dilation
            
        self.ceil_mode = ceil_mode
            
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        # run the common max-pooling
        if self.space.dimensionality <= 2:
            output = F.max_pool2d(input.tensor,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
        elif self.space.dimensionality == 3:
            output = F.max_pool3d(input.tensor,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
                
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape

        # compute the output shape (see 'torch.nn.MaxPool2D')
        ho = (hi + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        wo = (wi + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        
        if self.ceil_mode:
            ho = math.ceil(ho)
            wo = math.ceil(wo)
        else:
            ho = math.floor(ho)
            wo = math.floor(wo)

        return b, self.out_type.size, ho, wo

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.MaxPool2d` module and set to "eval" mode.

        """
    
        self.eval()
    
        return torch.nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation).eval()


class PointwiseMaxPoolAntialiased(PointwiseMaxPool):
    
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 ceil_mode: bool = False,
                 sigma: float = 0.6,
                 ):
        r"""

        Anti-aliased version of channel-wise max-pooling (each channel is treated independently).
        
        The max over a neighborhood is performed pointwise withot downsampling.
        Then, convolution with a gaussian blurring filter is performed before downsampling the feature map.
        
        Based on `Making Convolutional Networks Shift-Invariant Again <https://arxiv.org/abs/1904.11486>`_.
        
        
        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape
            sigma (float): standard deviation for the Gaussian blur filter

        """
        
        if dilation != 1:
            raise NotImplementedError("Dilation larger than 1 is not supported yet")
        
        super(PointwiseMaxPoolAntialiased, self).__init__(in_type, kernel_size, stride, padding, dilation, ceil_mode)
        
        assert sigma > 0.
        
        filter_size = 2*int(round(4*sigma))+1

        # Build the Gaussian smoothing filter
        
        grid_x = torch.arange(filter_size).repeat(filter_size).view(filter_size, filter_size)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)

        mean = (filter_size - 1) / 2.
        variance = sigma ** 2.
        
        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        
        # Build the gaussian kernel
        _filter = torch.exp(r / (2 * variance))
        
        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in 2d depthwise convolution
        _filter = _filter.view(1, 1, filter_size, filter_size).repeat((in_type.size, 1, 1, 1))

        self.register_buffer('filter', _filter)
        self._pad = tuple(p + int((filter_size-1)//2) for p in self.padding)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """
        
        assert input.type == self.in_type
        
        # evaluate the max operation densely (stride = 1)
        output = F.max_pool2d(input.tensor,
                              self.kernel_size,
                              1,
                              self.padding,
                              self.dilation,
                              self.ceil_mode)
        
        output = F.conv2d(output, self.filter, stride=self.stride, padding=self._pad, groups=output.shape[1])
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)
