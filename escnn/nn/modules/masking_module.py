
from escnn.nn import GeometricTensor
from escnn.nn import FieldType
from .equivariant_module import EquivariantModule

from itertools import product, repeat
from typing import Tuple

import torch

import math

__all__ = ["MaskModule"]


def build_mask(
        s,
        dim: int = 2,
        margin: float = 2.0,
        sigma: float = 2.0,
        dtype=torch.float32,
):
    mask = torch.zeros(1, 1, *repeat(s, dim), dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r > t:
            mask[(..., *k)] = math.exp((t - r) / sigma**2)
        else:
            mask[(..., *k)] = 1.
    return mask


class MaskModule(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            S: int,
            margin: float = 0.,
            sigma: float = 2.,
    ):
        r"""
        
        Performs an element-wise multiplication of the input with a *mask* of shape :math:`S^n`, where :math:`n` is the
        dimensionality of the underlying space.
        
        The mask has value :math:`1` in all pixels with distance smaller than
        :math:`\frac{S - 1}{2} \times (1 - \frac{\mathrm{margin}}{100})` from the center of the mask and :math:`0`
        elsewhere. Values change smoothly between the two regions.
        
        This operation is useful to remove from an input image or feature map all the part of the signal defined on the
        pixels which lay outside the circle inscribed in the grid.
        Because a rotation would move these pixels outside the grid, this information would anyways be
        discarded when rotating an image. However, allowing a model to use this information might break the guaranteed
        equivariance as rotated and non-rotated inputs have different information content.
        
        .. note::
            The input tensors provided to this module must have the following dimensions: :math:`B \times C \times S^n`,
            where :math:`B` is the minibatch dimension, :math:`C` is the channels dimension, and :math:`S^n` are the
            :math:`n` spatial dimensions (corresponding to the Euclidean basespace :math:`\R^n`) associated with the
            given input field type, i.e. ``in_type.gspace.dimensionality``.
            Each Euclidean dimension must be of size :math:`S`.

            For example, if :math:`S=10` and the ``in_type.gspace.dimensionality=2``, then the input tensors
            should be of size :math:`B \times C \times 10 \times 10`.  If ``in_type.gspace.dimensionality=3``
            instead, then the input tensors should be of size
            :math:`B \times C \times 10 \times 10 \times 10`.
        
        Args:
            in_type (FieldType): input field type
            S (int): the shape of the mask and the expected inputs
            margin (float, optional): margin around the mask in percentage with respect to the radius of the mask
            sigma (float, optional): how quickly masked pixels should approach 0.  This can be thought of a standard
                deviation in units of pixels/voxels.  For example, the default value of 2 means that only 5% of the
                original signal will remain 4 px into the masked region.
        
        """
        super(MaskModule, self).__init__()

        self.dim: int = in_type.gspace.dimensionality

        self.margin = margin
        self.mask = torch.nn.Parameter(
                build_mask(S, dim=self.dim, margin=margin, sigma=sigma),
                requires_grad=False,
        )

        self.in_type = self.out_type = in_type

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        assert input.tensor.shape[2:] == self.mask.shape[2:]

        out = input.tensor * self.mask
        return GeometricTensor(out, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

