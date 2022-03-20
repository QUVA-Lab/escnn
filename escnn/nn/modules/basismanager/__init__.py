from escnn.group import Representation
from escnn.nn.modules import utils
from collections import defaultdict
from typing import List
import torch


def retrieve_indices(reprs: List[Representation]):
    fiber_position = 0
    _indices = defaultdict(list)
    _count = defaultdict(int)
    _contiguous = {}

    for repr in reprs:
        _indices[repr.name] += list(range(fiber_position, fiber_position + repr.size))
        fiber_position += repr.size
        _count[repr.name] += 1

    for name, indices in _indices.items():
        # _contiguous[o_name] = indices == list(range(indices[0], indices[0]+len(indices)))
        _contiguous[name] = utils.check_consecutive_numbers(indices)
        _indices[name] = torch.LongTensor(indices)

    return _count, _indices, _contiguous


########################################################################################################################


from .basismanager import BasisManager
from .basisexpansion_blocks import BlocksBasisExpansion
from .basisexpansion_singleblock import SingleBlockBasisExpansion
from .basissampler_blocks import BlocksBasisSampler
from .basissampler_singleblock import SingleBlockBasisSampler

__all__ = [
    "BasisManager",
    "BlocksBasisExpansion",
    "SingleBlockBasisExpansion",
    "BlocksBasisSampler",
    "SingleBlockBasisSampler",
]



