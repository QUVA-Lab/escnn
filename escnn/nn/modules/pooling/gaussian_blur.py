
from .utils import get_nd_tuple

import torch
import torch.nn.functional as F

from typing import Optional, Union, Tuple

_CONV = {
        2: F.conv2d,
        3: F.conv3d,
}

class GaussianBlurND(torch.nn.Module):

    def __init__(
            self,
            d: int,
            *, 
            sigma: float = 0.6,
            stride: Union[int, Tuple[int, ...]] = 1,
            rel_padding: Optional[Union[int, Tuple[int, ...]]] = None,
            abs_padding: Optional[Union[int, Tuple[int, ...]]] = None,
            channels: int,

            # 4 for max pooling, 3 for average pooling.
            _kernel_size_factor: float,
    ):
        """
        Apply a Gaussian blur to the input.

        This is equivalent to a depth-wise convolution with a Gaussian filter.

        Args:
            d (int): Dimensionality of the base space (2 for images, 3 for volumes)

            sigma (float): Standard deviation for the Gaussian blur filter.

            stride (int): The stride of the blur filter.

            abs_padding: Implicit zero padding to be added on all sides of the 
                input, without regard to the size of the filter.  Note that the 
                size of the filter depends on *sigma* and *_kernel_size_factor*,
                and in this padding mode, the shape of the output tensor 
                depends on the size of the filter.  It is an error to specify 
                *abs_padding* and *rel_padding*.

            rel_padding: Implicit zero padding to be added on all sides of the 
                input, treating the filter as if it were 1x1 (or 1x1x1), no 
                matter what size it really is.  This means that the shape of 
                the output tensor is independent of the filter size.  It is an 
                error to specify *abs_padding* and *rel_padding*.
            
            channels (int): The channel dimension of the input.

            _kernel_size_factor (float): How big the Gaussian blur filter 
                should be, in terms of *sigma*.  See the code for the exact 
                expression, but the basic idea is that larger filters are 
                needed for larger standard deviations.
        """
        super().__init__()

        assert sigma > 0.

        if rel_padding is not None and abs_padding is not None:
            raise ValueError("can't specify `rel_padding` and `max_padding`")

        self.d = d
        self.sigma = sigma
        self.stride = stride
        self.kernel_size = 2 * int(round(_kernel_size_factor * sigma)) + 1

        # Build the Gaussian smoothing filter

        grid = torch.meshgrid(
                *[torch.arange(self.kernel_size)] * d,
                indexing='ij',
        )
        grid = torch.stack(grid, dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = sigma ** 2.

        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())

        # Build the gaussian kernel
        _filter = torch.exp(-r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in depthwise convolution
        _filter = _filter\
                .view(1, 1, *[self.kernel_size]*d)\
                .repeat((channels, 1, *[1]*d))

        self.register_buffer('filter', _filter)

        if abs_padding is not None:
            self.padding = abs_padding
        else:
            half_kernel_size = (self.kernel_size - 1) // 2
            self.padding = tuple(
                    p + half_kernel_size
                    for p in get_nd_tuple(rel_padding or 0, d)
            )

    def __repr__(self):
        return f'{self.__class__.__name__}(d={self.d}, sigma={self.sigma}, stride={self.stride}, padding={self.padding}, channels={self.filter.shape[1]})'

    def forward(self, x):
        return _CONV[self.d](
                x,
                self.filter,
                stride=self.stride,
                padding=self.padding,
                groups=x.shape[1],
        )

