

from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

__all__ = [
    "PointwiseAdaptiveAvgPool2D", "PointwiseAdaptiveAvgPool",
    "PointwiseAdaptiveAvgPool3D",
]


class _PointwiseAdaptiveAvgPoolND(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 d: int,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool2D`, wrapping it in
        the :class:`~escnn.nn.EquivariantModule` interface.
        
        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """

        assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                f"""Error! Representation "{r.name}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""
        
        super(_PointwiseAdaptiveAvgPoolND, self).__init__()

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(output_size, int):
            self.output_size = (output_size,) * self.d
        else:
            self.output_size = output_size

        assert isinstance(self.output_size, tuple) and len(self.output_size) == self.d, self.output_size

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
            output = F.adaptive_avg_pool2d(input.tensor, self.output_size)
        elif self.d == 3:
            output = F.adaptive_avg_pool3d(input.tensor, self.output_size)
        else:
            raise NotImplementedError

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2 + self.d
        assert input_shape[1] == self.in_type.size
        return (input_shape[0], self.out_type.size, *self.output_size)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.AdaptiveAvgPool2d` or
        :class:`torch.nn.AdaptiveAvgPool3d` module and set to "eval" mode.

        """
    
        self.eval()
    
        if self.d == 2:
            return torch.nn.AdaptiveAvgPool2d(self.output_size).eval()
        elif self.d == 3:
            return torch.nn.AdaptiveAvgPool3d(self.output_size).eval()
        else:
            raise NotImplementedError


class PointwiseAdaptiveAvgPool2D(_PointwiseAdaptiveAvgPoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool2D`, wrapping it in
        the :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        super(PointwiseAdaptiveAvgPool2D, self).__init__(in_type, 2, output_size)


class PointwiseAdaptiveAvgPool3D(_PointwiseAdaptiveAvgPoolND):

    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool3D`, wrapping it in
        the :class:`~escnn.nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the volume of the form H x W x D

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 3

        super(PointwiseAdaptiveAvgPool3D, self).__init__(in_type, 3, output_size)


# for backward compatibility
PointwiseAdaptiveAvgPool = PointwiseAdaptiveAvgPool2D
