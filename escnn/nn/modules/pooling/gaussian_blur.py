
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
            *, 
            sigma: float,
            kernel_size: int,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Optional[Union[int, Tuple[int, ...]]] = None,
            rel_padding: Optional[Union[int, Tuple[int, ...]]] = None,
            d: int,
            channels: int,
    ):
        """
        Apply a Gaussian blur to the input.

        This is equivalent to a depth-wise convolution with a Gaussian filter.

        Args:
            sigma (float): Standard deviation of the Gaussian making up the 
                blur filter.

            kernel_size (int): Size of the convolutional filter used to apply 
                the blur.  Note that this should be related to the value of 
                *sigma*; larger standard deviations require larger kernels to 
                overlap the same density.  You can think of the Gaussian blur 
                as being truncated to zero beyond the bounds of the filter.

            stride (int): Stride of the convolutional filter used to apply the 
                blur.

            padding: Implicit zero padding to be added on all sides of the 
                input, without regard to the size of the filter.  It is an 
                error to specify *padding* and *rel_padding*.

            rel_padding: Implicit zero padding to be added on all sides of the 
                input, treating the filter as if it were 1x1 (or 1x1x1, etc.), 
                no matter what size it really is.  This means that the shape of 
                the output tensor is independent of the filter size.  This is 
                helpful when, for example, the *kernel_size* argument is 
                dynamically calculated as a function of *sigma*.  It is an 
                error to specify *padding* and *rel_padding*.
            
            d (int): Dimensionality of the base space (2 for images, 3 for 
                volumes)

            channels (int): Channel dimension of the input.
        """
        super().__init__()

        assert sigma > 0.

        if padding is not None and rel_padding is not None:
            raise ValueError("can't specify `padding` and `rel_padding`")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.d = d

        # Build the Gaussian smoothing filter

        grid = torch.meshgrid(
                *[torch.arange(kernel_size)] * d,
                indexing='ij',
        )
        grid = torch.stack(grid, dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())

        # Build the gaussian kernel
        _filter = torch.exp(-r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in depthwise convolution
        _filter = _filter\
                .view(1, 1, *[kernel_size]*d)\
                .repeat((channels, 1, *[1]*d))

        self.register_buffer('filter', _filter)

        if padding is not None:
            self.padding = padding
        else:
            half_kernel_size = (kernel_size - 1) // 2
            self.padding = tuple(
                    p + half_kernel_size
                    for p in get_nd_tuple(rel_padding or 0, d)
            )

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, d={self.d}, channels={self.filter.shape[1]})'

    def forward(self, x):
        return _CONV[self.d](
                x,
                self.filter,
                stride=self.stride,
                padding=self.padding,
                groups=x.shape[1],
        )

def kernel_size_from_radius(radius):
    return 2 * int(round(radius)) + 1
