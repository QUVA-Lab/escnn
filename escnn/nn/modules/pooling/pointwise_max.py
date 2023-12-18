
from escnn.nn import FieldType

from .pointwise import _PointwisePoolND, _PointwiseMaxPoolAntialiasedND

import torch

from typing import Tuple, Union, Optional

__all__ = [
    "PointwiseMaxPool2D", "PointwiseMaxPool",
    "PointwiseMaxPool3D",
    "PointwiseMaxPoolAntialiased2D", "PointwiseMaxPoolAntialiased",
    "PointwiseMaxPoolAntialiased3D",
]

class PointwiseMaxPool2D(_PointwisePoolND):

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
        This module works exactly as :class:`torch.nn.MaxPool2d`, wrapping it in the
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
        super().__init__(
            in_type, 2,
            pool=torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            ),
        )


class PointwiseMaxPool3D(_PointwisePoolND):

    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.MaxPool3d`, wrapping it in the
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
        super().__init__(
            in_type, 3,
            pool=torch.nn.MaxPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            ),
        )


class PointwiseAdaptiveMaxPool2D(_PointwisePoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Module that implements adaptive channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveMaxPool2d`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.

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
            pool=torch.nn.AdaptiveMaxPool2d(
                output_size=output_size,
            ),
        )


class PointwiseAdaptiveMaxPool3D(_PointwisePoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int, int]]
                 ):
        r"""

        Module that implements adaptive channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveMaxPool3d`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """
        super().__init__(
            in_type, 3,
            pool=torch.nn.AdaptiveMaxPool3d(
                output_size=output_size,
            ),
        )

class PointwiseMaxPoolAntialiased2D(_PointwiseMaxPoolAntialiasedND):

    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False,
                 sigma: float = 0.6,
                 ):
        r"""

        Anti-aliased version of channel-wise max-pooling (each channel is treated independently).

        The max over a neighborhood is performed pointwise without downsampling.
        Then, convolution with a Gaussian blurring filter is performed before downsampling the feature map.

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
        super().__init__(
            in_type, 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            sigma=sigma,
        )


class PointwiseMaxPoolAntialiased3D(_PointwiseMaxPoolAntialiasedND):

    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0,
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
        super().__init__(
            in_type, 3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            sigma=sigma,
        )


# for backward compatibility
PointwiseMaxPool = PointwiseMaxPool2D
PointwiseAdaptiveMaxPool = PointwiseAdaptiveMaxPool2D
PointwiseMaxPoolAntialiased = PointwiseMaxPoolAntialiased2D


