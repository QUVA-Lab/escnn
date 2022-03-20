

from escnn.nn import FieldType, GeometricTensor
from escnn.group import Representation
from escnn.kernels import KernelBasis

from torch_geometric.data import Data

from .rd_point_convolution import _RdPointConv

from typing import Callable, Tuple, Dict, Union, List

import torch
import numpy as np


import math


__all__ = ["R2PointConv"]


class R2PointConv(_RdPointConv):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 width: float = None,
                 n_rings: int = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 ):

        basis_filter, self._rings, self._sigma, self._maximum_frequency = compute_basis_params(
            frequencies_cutoff, rings, sigma, width, n_rings, basis_filter
        )

        super(R2PointConv, self).__init__(
            in_type, out_type,
            d=2,
            groups=groups,
            bias=bias,
            basis_filter=basis_filter,
            recompute=recompute
        )

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        return self.space.build_kernel_basis(in_repr, out_repr,
                                             self._sigma, self._rings,
                                             maximum_frequency=self._maximum_frequency
                                             )


def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""

    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (true) or not (false)

    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. in thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.

    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    args:
        frequency_cutoff (float): factor controlling the bandlimiting

    returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff

    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["irrep:frequency"]) <= frequency_cutoff(attributes["radius"])

    return bl_filter


def compute_basis_params(
        frequencies_cutoff: Union[float, Callable[[float], float]] = None,
        rings: List[float] = None,
        sigma: List[float] = None,
        width: float = None,
        n_rings: int = None,
        custom_basis_filter: Callable[[dict], bool] = None,
):

    assert (width is not None and n_rings is not None) != (rings is not None)

    # by default, the number of rings equals half of the filter size
    if rings is None:
        assert width > 0.
        assert n_rings > 0
        rings = torch.linspace(0, width, n_rings)
        rings = rings.tolist()

    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.:
                sigma[i] = 0.005
    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)

    if frequencies_cutoff is None:
        frequencies_cutoff = 3.

    if isinstance(frequencies_cutoff, float):
        frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r

    # check if the object is a callable function
    assert callable(frequencies_cutoff)

    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

    fco_filter = bandlimiting_filter(frequencies_cutoff)

    if custom_basis_filter is not None:
        basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (
                custom_basis_filter(d) and fco_filter(d)
        )
    else:
        basis_filter = fco_filter

    return basis_filter, rings, sigma, maximum_frequency

