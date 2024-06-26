
from escnn.kernels import KernelBasis, EmptyBasisException
from .basismanager import BasisManager

from typing import Callable, Dict, List, Iterable, Union

import torch
import numpy as np

__all__ = ["SingleBlockBasisExpansion", "block_basisexpansion"]


class SingleBlockBasisExpansion(torch.nn.Module, BasisManager):
    
    def __init__(self,
                 basis: KernelBasis,
                 points: np.ndarray,
                 mask: np.ndarray = None
                 ):
        r"""
        
        Basis expansion method for a single contiguous block, i.e. for kernels whose input type and output type contain
        only fields of one type.
        
        Args:
            basis (KernelBasis): analytical basis to sample
            points (np.ndarray): points where the analytical basis should be sampled
            mask (np.ndarray, optional): binary mask to select only a subset of the basis elements.
                                         By default (``None``), all elements are kept.
            
        """

        super(SingleBlockBasisExpansion, self).__init__()

        # This is a hack to prevent PyTorch to register basis as a submodule
        # This is needed since a KernelBasis is a torch.nn.Module, but we only need it in the __init__ to construct the
        # filter basis. We keep this basis mostly for debugging or inspection purpose (e.g. to generate the basis
        # attributes), but we don't want it to appear among the sub-modules of this class (e.g. to prevent it is loaded
        # on CUDA, wasting memory, when this module is moved).
        object.__setattr__(self, 'basis', basis)

        if mask is None:
            mask = np.ones(len(basis), dtype=bool)
            
        assert mask.shape == (len(basis),) and mask.dtype == bool
        
        if not mask.any():
            raise EmptyBasisException

        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in basis:
            sizes.append(attr["shape"][0])

        # sample the basis on the grid
        # basis has shape (p, k, o, i)
        # permute to (k, o, i, p)
        sampled_basis = basis.sample(torch.tensor(
            points, dtype=torch.float32
        )).permute(1, 2, 3, 0)

        # normalize the basis
        sizes = torch.tensor(sizes, dtype=sampled_basis.dtype)
        sampled_basis = normalize_basis(sampled_basis, sizes)

        # discard the basis which are close to zero everywhere
        norms = (sampled_basis ** 2).reshape(sampled_basis.shape[0], -1).sum(1) > 1e-2
        mask = torch.tensor(mask) & norms
        if not mask.any():
            raise EmptyBasisException
        sampled_basis = sampled_basis[mask, ...]
        self._mask = mask
        
        # register the bases tensors as parameters of this module
        self.register_buffer('sampled_basis', sampled_basis)
            
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
    
        assert len(weights.shape) == 2 and weights.shape[1] == self.dimension()
    
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        return torch.einsum('boi...,kb->koi...', self.sampled_basis, weights)

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
        return self.sampled_basis.shape[0]
    
    def __eq__(self, other):
        if isinstance(other, SingleBlockBasisExpansion):
            return (
                    self.basis == other.basis and
                    torch.allclose(self.sampled_basis, other.sampled_basis) and
                    (self._mask == other._mask).all()
            )
        else:
            return False
    
    def __hash__(self):
        return 10000 * hash(self.basis) + 100 * hash(self.sampled_basis) + hash(self._mask)

# dictionary storing references to already built basis tensors.
# when a new filter tensor is built, it is also stored here
# when the same basis is built again (eg. in another layer), the already existing filter tensor is retrieved
_stored_filters = {}


def block_basisexpansion(basis: KernelBasis,
                         points: np.ndarray,
                         basis_filter: Callable[[dict], bool] = None,
                         recompute: bool = False
                         ) -> SingleBlockBasisExpansion:
    r"""


    Args:
        basis (KernelBasis): basis defining the space of kernels
        points (ndarray): points where the analytical basis should be sampled
        basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                           element's attributes and return whether to keep it or not.
        recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

    """

    # TODO: rather than recompute, could pass a custom key to allow for cutom model-based caching

    if basis_filter is not None:
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
    else:
        mask = np.ones(len(basis), dtype=bool)

    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        key = (basis, mask.tobytes(), points.tobytes())
        if key not in _stored_filters:
            _stored_filters[key] = SingleBlockBasisExpansion(basis, points, mask)

        return _stored_filters[key]
    
    else:
        return SingleBlockBasisExpansion(basis, points, mask)


def normalize_basis(basis: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    r"""

    Normalize the filters in the input tensor.
    The tensor of shape :math:`(B, O, I, ...)` is interpreted as a basis containing ``B`` filters/elements, each with
    ``I`` inputs and ``O`` outputs. The spatial dimensions ``...`` can be anything.

    .. notice ::
        Notice that the method changes the input tensor inplace

    Args:
        basis (torch.Tensor): tensor containing the basis to normalize
        sizes (torch.Tensor): original input size of the basis elements, without the padding and the change of basis

    Returns:
        the normalized basis (the operation is done inplace, so this is ust a reference to the input tensor)

    """
    
    b = basis.shape[0]
    assert len(basis.shape) > 2, basis.shape
    assert sizes.shape == (b,), (sizes.shape, b, basis.shape)
    
    # compute the norm of each basis vector
    norms = torch.einsum('bop...,bpq...->boq...', (basis, basis.transpose(1, 2)))
    
    # Removing the change of basis, these matrices should be multiples of the identity
    # where the scalar on the diagonal is the variance
    # in order to find this variance, we can compute the trace (which is invariant to the change of basis)
    # and divide by the number of elements in the diagonal ignoring the padding.
    # Therefore, we need to know the original size of each basis element.
    norms = torch.einsum("bii...->b", norms)
    # norms = norms.reshape(b, -1).sum(1)
    norms /= sizes

    norms[norms < 1e-15] = 0
    
    norms = torch.sqrt(norms)
    
    norms[norms < 1e-6] = 1
    norms[norms != norms] = 1
    
    norms = norms.view(b, *([1] * (len(basis.shape) - 1)))
    
    # divide by the norm
    basis /= norms

    return basis



