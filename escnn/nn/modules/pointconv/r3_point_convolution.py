

from escnn.nn import FieldType
from escnn.group import Representation
from escnn.kernels import KernelBasis

from .rd_point_convolution import _RdPointConv
from .r2_point_convolution import compute_basis_params

from typing import Callable, Tuple, Dict, Union, List

__all__ = ["R3PointConv"]


class R3PointConv(_RdPointConv):
    
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

        super(R3PointConv, self).__init__(
            in_type, out_type,
            d=3,
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


