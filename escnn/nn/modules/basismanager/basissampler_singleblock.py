
from escnn.kernels import SteerableKernelBasis, EmptyBasisException
from .basismanager import BasisManager

from typing import Callable, Dict, Iterable, Union, Tuple

import torch
import numpy as np

__all__ = ["SingleBlockBasisSampler", "block_basissampler"]


class SingleBlockBasisSampler(torch.nn.Module, BasisManager):
    
    def __init__(self,
                 basis: SteerableKernelBasis,
                 mask: np.ndarray = None
                 ):
        r"""
        
        Basis expansion method for a single contiguous block, i.e. for kernels whose input type and output type contain
        only fields of one type.
        
        Args:
            basis (SteerableKernelBasis): analytical basis to sample
            mask (np.ndarray, optional): binary mask to select only a subset of the basis elements.
                                         By default (``None``), all elements are kept.
            
        """

        super(SingleBlockBasisSampler, self).__init__()
        
        self.basis = basis
        
        if mask is None:
            mask = np.ones(len(basis), dtype=bool)
            
        assert mask.shape == (len(basis),) and mask.dtype == bool
        
        if not mask.any():
            raise EmptyBasisException

        self._mask = mask
        self.basis = basis

        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in basis:
            sizes.append(attr["shape"][0])

        sizes = torch.tensor(sizes, dtype=torch.float32).reshape(1, 1, 1, -1)
        sizes = sizes[..., self._mask]

        # to normalize the basis
        self.register_buffer(
            'sizes',
            sizes
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        assert len(points.shape) == 2
        basis = self.basis.sample(points)

        # basis has shape (p, k, o, i)
        # permute to (p, o, i, k)
        basis = basis.permute((0, 2, 3, 1))

        if self._mask is not None:
            basis = basis[:, :, :, self._mask]

        # return basis.to(device=device, dtype=points.dtype) * self.sizes
        return basis * self.sizes

    def get_element_info(self, id: int) -> Dict:
        idx = 0
        for i, attr in enumerate(self.basis):
            if self._mask[i]:
                if idx == id:
                    attr["id"] = idx
                    return attr
                else:
                    idx += 1

    def get_basis_info(self) -> Iterable[Dict]:
        idx = 0
        for i, attr in enumerate(self.basis):
            if self._mask[i]:
                attr["id"] = idx
                idx += 1
                yield attr

    def dimension(self) -> int:
        return self._mask.astype(int).sum()
    
    def __eq__(self, other):
        if isinstance(other, SingleBlockBasisSampler
                      ):
            return (
                    self.basis == other.basis and
                    np.all(self._mask == other._mask)
            )
        else:
            return False
    
    def __hash__(self):
        return 10000 * hash(self.basis) + hash(self._mask.tobytes())

# dictionary storing references to already built basis samplers
# when a new filter tensor is built, it is also stored here
# when the same basis is built again (eg. in another layer), the already existing filter tensor is retrieved
_stored_filters = {}


def block_basissampler(basis: SteerableKernelBasis,
                         basis_filter: Callable[[dict], bool] = None,
                         recompute: bool = False
                         ) -> SingleBlockBasisSampler:
    r"""


    Args:
        basis (SteerableKernelBasis): basis defining the space of kernels
        basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                           element's attributes and return whether to keep it or not.
        recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

    """

    if basis_filter is not None:
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
    else:
        mask = np.ones(len(basis), dtype=bool)

    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        key = (basis, mask.tobytes())
        if key not in _stored_filters:
            _stored_filters[key] = SingleBlockBasisSampler(basis, mask)

        return _stored_filters[key]
    
    else:
        return SingleBlockBasisSampler(basis, mask)


