
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch.nn.functional as F
import torch

from typing import List, Tuple, Any, Union

import math

__all__ = [
    "PointwiseMaxPool2D", "PointwiseMaxPool",
    "PointwiseMaxPool3D",
    "PointwiseMaxPoolAntialiased2D", "PointwiseMaxPoolAntialiased",
    "PointwiseMaxPoolAntialiased3D",
]


class _PointwiseMaxPoolND(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 d: int,
                 kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = None,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.MaxPool2D` or :class:`torch.nn.MaxPool3D`, wrapping it in the
        :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when True, will use ceil instead of floor to compute the output shape

        """

        assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                f"""Error! Representation "{r.name}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""

        super(_PointwiseMaxPoolND, self).__init__()

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

        if isinstance(dilation, int):
            self.dilation = (dilation,) * self.d
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
        if self.d == 2:
            output = F.max_pool2d(input.tensor,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
        elif self.d == 3:
            output = F.max_pool3d(input.tensor,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
        else:
            raise NotImplementedError

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2 + self.d, (input_shape, self.d)
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        shape_i = input_shape[2:]

        # compute the output shape (see 'torch.nn.MaxPool2D')
        shape_o = tuple(
            (shape_i[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) / self.stride[i] + 1
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

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.MaxPool2d` module and set to "eval" mode.

        """

        self.eval()

        if self.d == 2:
            return torch.nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation).eval()
        elif self.d == 3:
            return torch.nn.MaxPool3d(self.kernel_size, self.stride, self.padding, self.dilation).eval()
        else:
            raise NotImplementedError


class _PointwiseMaxPoolAntialiasedND(_PointwiseMaxPoolND):

    def __init__(self,
                 in_type: FieldType,
                 d: int,
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
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape
            sigma (float): standard deviation for the Gaussian blur filter

        """

        if dilation != 1:
            raise NotImplementedError("Dilation larger than 1 is not supported yet")

        super(_PointwiseMaxPoolAntialiasedND, self).__init__(in_type, d, kernel_size, stride, padding, dilation, ceil_mode)

        assert sigma > 0.

        filter_size = 2 * int(round(4 * sigma)) + 1

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
        self._pad = tuple(p + int((filter_size - 1) // 2) for p in self.padding)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        if self.d == 2:
            # evaluate the max operation densely (stride = 1)
            output = F.max_pool2d(input.tensor,
                                  self.kernel_size,
                                  1,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
            output = F.conv2d(output, self.filter, stride=self.stride, padding=self._pad, groups=output.shape[1])
        elif self.d == 3:
            # evaluate the max operation densely (stride = 1)
            output = F.max_pool3d(input.tensor,
                                  self.kernel_size,
                                  1,
                                  self.padding,
                                  self.dilation,
                                  self.ceil_mode)
            output = F.conv3d(output, self.filter, stride=self.stride, padding=self._pad, groups=output.shape[1])
        else:
            raise NotImplementedError

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def export(self):
        raise NotImplementedError


class PointwiseMaxPool2D(_PointwiseMaxPoolND):

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
        assert in_type.gspace.dimensionality == 2

        super(PointwiseMaxPool2D, self).__init__(
            in_type, 2, kernel_size, stride, padding, dilation, ceil_mode
        )


class PointwiseMaxPool3D(_PointwiseMaxPoolND):

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
        This module works exactly as :class:`torch.nn.MaxPool3D`, wrapping it in the
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
        assert in_type.gspace.dimensionality == 3

        super(PointwiseMaxPool3D, self).__init__(
            in_type, 3, kernel_size, stride, padding, dilation, ceil_mode
        )


class PointwiseMaxPoolAntialiased2D(_PointwiseMaxPoolAntialiasedND):

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

        super(PointwiseMaxPoolAntialiased2D, self).__init__(
            in_type, 2, kernel_size, stride, padding, dilation, ceil_mode, sigma
        )


class PointwiseMaxPoolAntialiased3D(_PointwiseMaxPoolAntialiasedND):

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

        super(PointwiseMaxPoolAntialiased3D, self).__init__(
            in_type, 3, kernel_size, stride, padding, dilation, ceil_mode, sigma
        )


# for backward compatibility
PointwiseMaxPool = PointwiseMaxPool2D
PointwiseMaxPoolAntialiased = PointwiseMaxPoolAntialiased2D


