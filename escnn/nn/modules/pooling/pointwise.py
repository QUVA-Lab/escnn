
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.gspaces import GSpace

from ..equivariant_module import EquivariantModule
from .gaussian_blur import GaussianBlurND, kernel_size_from_radius
from .utils import get_nd_tuple

import torch
import math
from collections import OrderedDict

from typing import List, Tuple, Any, Union, Optional

_MAX_POOLS = {
        2: torch.nn.MaxPool2d,
        3: torch.nn.MaxPool3d,
}

class _PointwisePoolND(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            d: int,
            pool: torch.nn.Module,
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

        super().__init__()

        check_dimensions(in_type, d)
        check_pointwise_ok(in_type)

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.pool = pool

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        output = self.pool(input.tensor)

        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        return calc_pool_output_shape(self.d, self.pool, input_shape, self.out_type)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test 
        # equivariance
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.MaxPool2d` module and set to "eval" mode.

        """

        return self.pool.eval()

class _PointwiseAvgPoolAntialiasedND(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            d: int,
            *,
            sigma: float,
            stride: Union[int, Tuple[int, int]],
            padding: Optional[Union[int, Tuple[int, int]]],
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
        
        super().__init__()

        check_dimensions(in_type, d)
        # Don't need to check that the representation is compatible with 
        # pointwise nonlinearities, see #65.

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.blur = GaussianBlurND(
                sigma=sigma,
                kernel_size=kernel_size_from_radius(sigma * 3),
                stride=stride,
                padding=padding,
                d=d,
                channels=in_type.size,
        )
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """
        
        assert input.type == self.in_type

        output = self.blur(input.tensor)

        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return calc_pool_output_shape(self.d, self.blur, input_shape, self.out_type)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can't test 
        # equivariance
        pass

    def export(self):
        return self.blur.eval()

class _PointwiseMaxPoolAntialiasedND(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            d: int,
            *,
            kernel_size: Union[int, Tuple[int, ...]],
            stride: Optional[Union[int, Tuple[int, ...]]],
            padding: Union[int, Tuple[int, ...]],
            ceil_mode: bool,
            sigma: float,
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

        super().__init__()

        if ceil_mode:
            from warnings import warn
            warn("The `ceil_mode` argument doesn't do anything, because the stride for the pooling step is always 1.")

        check_dimensions(in_type, d)
        check_pointwise_ok(in_type)

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        pool = _MAX_POOLS[d](
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
        )
        blur = GaussianBlurND(
                sigma=sigma,
                kernel_size=kernel_size_from_radius(sigma * 4),
                stride=stride if stride is not None else kernel_size,
                rel_padding=padding,
                channels=in_type.size,
                d=d,
        )
        self.layers = torch.nn.Sequential(
                OrderedDict([('pool', pool), ('blur', blur)]),
        )

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type
        
        output = self.layers(input.tensor)

        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        for layer in self.layers:
            input_shape = calc_pool_output_shape(self.d, layer, input_shape, self.out_type)

        return input_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test 
        # equivariance
        pass

    def export(self):
        return self.layers.eval()

def check_dimensions(in_type: FieldType, d: int) -> None:
    assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

    assert isinstance(in_type.gspace, GSpace)
    assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

def check_pointwise_ok(in_type: FieldType) -> None:
    for r in in_type.representations:
        assert 'pointwise' in r.supported_nonlinearities, \
            f"""Error! Representation "{r.name}" does not support pointwise non-linearities
            so it is not possible to pool each channel independently"""

def calc_pool_output_shape(
        d: int,
        pool: Any,
        input_shape: Tuple[int, ...],
        out_type: FieldType,
) -> Tuple[int, ...]:

    assert len(input_shape) == 2 + d, (input_shape, d)

    if hasattr(pool, 'output_size'):
        output_size = get_nd_tuple(pool.output_size, d)
        return (input_shape[0], out_type.size, *output_size)

    b, c, *xyz_i = input_shape

    kernel_size = get_nd_tuple(pool.kernel_size, d)
    padding = get_nd_tuple(pool.padding, d)
    stride = get_nd_tuple(pool.stride, d)
    dilation = get_nd_tuple(getattr(pool, 'dilation', 1), d)

    # See online docs for `torch.nn.MaxPool2D`.
    xyz_o = tuple(
        (xyz_i[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1
        for i in range(d)
    )

    ceil_mode = getattr(pool, 'ceil_mode', False)
    round = math.ceil if ceil_mode else math.floor

    xyz_o = tuple(
        int(round(xyz_o[i]))
        for i in range(d)
    )

    return (b, out_type.size, *xyz_o)

