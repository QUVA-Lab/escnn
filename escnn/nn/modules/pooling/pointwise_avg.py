

from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch.nn.functional as F
import torch

from typing import List, Tuple, Any, Union

import math

__all__ = [
    "PointwiseAvgPool2D", "PointwiseAvgPool",
    "PointwiseAvgPool3D",
    "PointwiseAvgPoolAntialiased2D", "PointwiseAvgPoolAntialiased",
    "PointwiseAvgPoolAntialiased3D"
]


class _PointwiseAvgPoolND(EquivariantModule):
    def __init__(self,
                 in_type: FieldType,
                 d: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool2D` or :class:`torch.nn.AvgPool3D`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            kernel_size: the size of the window to take a average over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape

        """

        assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

        # for r in in_type.representations:
        #     assert 'pointwise' in r.supported_nonlinearities, \
        #         """Error! Representation "{}" does not support pointwise non-linearities
        #         so it is not possible to pool each channel independently"""

        super(_PointwiseAvgPoolND, self).__init__()

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.d
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride,) * self.d
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding,) * self.d
        else:
            self.padding = padding

        self.ceil_mode = ceil_mode

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        # run the common avg-pooling
        if self.d == 2:
            output = F.avg_pool2d(input.tensor,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  ceil_mode=self.ceil_mode)
        elif self.d == 3:
            output = F.avg_pool3d(input.tensor,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  ceil_mode=self.ceil_mode)

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2 + self.d, (input_shape, self.d)
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        shape_i = input_shape[2:]

        # compute the output shape (see 'torch.nn.AvgPool2D')
        shape_o = tuple(
            (shape_i[i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] + 1
            for i in range(self.d)
        )

        if self.ceil_mode:
            shape_o = tuple(
                math.ceil(shape_o[i])
                for i in range(self.d)
            )
        else:
            shape_o = tuple(
                math.floor(shape_o[i])
                for i in range(self.d)
            )

        return (b, self.out_type.size, *shape_o)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass

    def export(self) -> torch.nn.Module:
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.AvgPool2d` or :class:`torch.nn.AvgPool3d` module and set to "eval" mode.

        """

        self.eval()

        if self.d == 2:
            return torch.nn.AvgPool2d(self.kernel_size, self.stride, self.padding, self.ceil_mode).eval()
        elif self.d == 3:
            return torch.nn.AvgPool3d(self.kernel_size, self.stride, self.padding, self.ceil_mode).eval()
        else:
            raise NotImplementedError


class PointwiseAvgPool2D(_PointwiseAvgPoolND):

    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool2D`, wrapping it in the
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
        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        super(PointwiseAvgPool2D, self).__init__(
            in_type, 2, kernel_size, stride, padding, ceil_mode
        )


class PointwiseAvgPool3D(_PointwiseAvgPoolND):

    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool3D`, wrapping it in the
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
        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 3

        super(PointwiseAvgPool3D, self).__init__(
            in_type, 3, kernel_size, stride, padding, ceil_mode
        )


class _PointwiseAvgPoolAntialiasedND(_PointwiseAvgPoolND):

    def __init__(self,
                 in_type: FieldType,
                 d: int,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 # kernel_size: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = None,
                 # dilation: Union[int, Tuple[int, int]] = 1,
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
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            sigma (float): standard deviation for the Gaussian blur filter
            stride: the stride of the window.
            padding: additional zero padding to be added on both sides

        """
        assert sigma > 0.
        filter_size = 2*int(round(3*sigma))+1

        if stride is None:
            stride = filter_size

        if padding is None:
            padding = int((filter_size-1)//2)

        super(_PointwiseAvgPoolAntialiasedND, self).__init__(
            in_type, d, filter_size, stride, padding
        )

        # Build the Gaussian smoothing filter
        grid = torch.stack(torch.meshgrid(*[torch.arange(filter_size)] * self.d, indexing='ij'), dim=-1)

        mean = (filter_size - 1) / 2.
        variance = sigma ** 2.

        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())

        # Build the gaussian kernel
        _filter = torch.exp(r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in 2d depthwise convolution
        _filter = _filter.view(1, 1, *[filter_size]*self.d).repeat((in_type.size, 1, *[1]*self.d))

        self.register_buffer('filter', _filter)

        ################################################################################################################

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        if self.d == 2:
            output = F.conv2d(input.tensor, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        elif self.d == 3:
            output = F.conv3d(input.tensor, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        else:
            raise NotImplementedError

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def export(self):
        raise NotImplementedError


class PointwiseAvgPoolAntialiased2D(_PointwiseAvgPoolAntialiasedND):

    def __init__(self,
                 in_type: FieldType,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = None,
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

        super(PointwiseAvgPoolAntialiased2D, self).__init__(
            in_type, 2, sigma, stride, padding
        )


class PointwiseAvgPoolAntialiased3D(_PointwiseAvgPoolAntialiasedND):

    def __init__(self,
                 in_type: FieldType,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = None,
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

        super(PointwiseAvgPoolAntialiased3D, self).__init__(
            in_type, 3, sigma, stride, padding
        )


# for backward compatibility
PointwiseAvgPool = PointwiseAvgPool2D
PointwiseAvgPoolAntialiased = PointwiseAvgPoolAntialiased2D
