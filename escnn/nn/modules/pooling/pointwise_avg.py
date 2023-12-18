

from escnn.nn import FieldType

from .pointwise import _PointwisePoolND, _PointwiseAvgPoolAntialiasedND

import torch

from typing import Tuple, Union, Optional

__all__ = [
    "PointwiseAvgPool",
    "PointwiseAvgPoolAntialiased",
    "PointwiseAvgPool2D",
    "PointwiseAvgPoolAntialiased2D",
]


class PointwiseAvgPool2D(_PointwisePoolND):
    
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool2d`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a average over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape

        """
        super().__init__(
            in_type, 2,
            pool=torch.nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            ),
        )
    
    
class PointwiseAvgPool3D(_PointwisePoolND):
    
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool3d`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a average over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape

        """
        super().__init__(
            in_type, 3,
            pool=torch.nn.AvgPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            ),
        )
    
    
class PointwiseAdaptiveAvgPool2D(_PointwisePoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool2d`, wrapping it in
        the :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """
        super().__init__(
                in_type, 2,
                pool=torch.nn.AdaptiveAvgPool2d(
                    output_size=output_size,
                ),
        )


class PointwiseAdaptiveAvgPool3D(_PointwisePoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int, int]]
                 ):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool3d`, wrapping it in
        the :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the volume of the form H x W x D

        """
        super().__init__(
                in_type, 3,
                pool=torch.nn.AdaptiveAvgPool3d(
                    output_size=output_size,
                ),
        )

class PointwiseAvgPoolAntialiased2D(_PointwiseAvgPoolAntialiasedND):
    
    def __init__(self,
                 in_type: FieldType,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 padding: Optional[Union[int, Tuple[int, int]]] = None,
                 ):
        r"""

        Antialiased channel-wise average-pooling: each channel is treated independently.
        It performs strided convolution with a Gaussian blur filter.
        
        The size of the filter is computed as 3 standard deviations of the Gaussian curve.
        By default, padding is added such that input size is preserved if stride is 1.
        
        Based on `Making Convolutional Networks Shift-Invariant Again <https://arxiv.org/abs/1904.11486>`_.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            sigma (float): standard deviation for the Gaussian blur filter
            stride: the stride of the window.
            padding: additional zero padding to be added on both sides

        """
        super().__init__(
            in_type, 2,
            sigma=sigma,
            stride=stride,
            padding=padding,
        )

class PointwiseAvgPoolAntialiased3D(_PointwiseAvgPoolAntialiasedND):
    
    def __init__(self,
                 in_type: FieldType,
                 sigma: float,
                 stride: Union[int, Tuple[int, int, int]],
                 padding: Optional[Union[int, Tuple[int, int, int]]] = None,
                 ):
        r"""

        Antialiased channel-wise average-pooling: each channel is treated independently.
        It performs strided convolution with a Gaussian blur filter.
        
        The size of the filter is computed as 3 standard deviations of the Gaussian curve.
        By default, padding is added such that input size is preserved if stride is 1.
        
        Inspired by `Making Convolutional Networks Shift-Invariant Again <https://arxiv.org/abs/1904.11486>`_.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            sigma (float): standard deviation for the Gaussian blur filter
            stride: the stride of the window.
            padding: additional zero padding to be added on both sides

        """
        super().__init__(
            in_type, 3,
            sigma=sigma,
            stride=stride,
            padding=padding,
        )
        

# for backward compatibility
PointwiseAvgPool = PointwiseAvgPool2D
PointwiseAdaptiveAvgPool = PointwiseAdaptiveAvgPool2D
PointwiseAvgPoolAntialiased = PointwiseAvgPoolAntialiased2D
