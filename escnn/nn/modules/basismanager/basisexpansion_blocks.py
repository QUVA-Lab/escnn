
from escnn.kernels import KernelBasis, EmptyBasisException
from escnn.group import Representation
from escnn.nn.modules import utils
from escnn.utils import unique_ever_seen

from .basismanager import BasisManager
from .basisexpansion_singleblock import block_basisexpansion

from . import retrieve_indices

from collections import defaultdict

from typing import Callable, List, Iterable, Dict

import torch
import numpy as np


__all__ = ["BlocksBasisExpansion"]


class BlocksBasisExpansion(torch.nn.Module, BasisManager):
    
    def __init__(self,
                 in_reprs: List[Representation],
                 out_reprs: List[Representation],
                 basis_generator: Callable[[Representation, Representation], KernelBasis],
                 points: np.ndarray,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 ):
        r"""
        
        Method that performs the expansion of a (already sampled) filter basis.
        
        Args:
            in_reprs (list): the input field type
            out_reprs (list): the output field type
            basis_generator (callable): method that generates the analytical filter basis
            points (~numpy.ndarray): points where the analytical basis should be sampled
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.
        
        Attributes:
            ~.S (int): number of points where the filters are sampled
            
        """

        super(BlocksBasisExpansion, self).__init__()
        self._in_reprs = in_reprs
        self._out_reprs = out_reprs
        self._input_size = sum(r.size for r in in_reprs)
        self._output_size = sum(r.size for r in out_reprs)
        self.points = points
        
        # int: number of points where the filters are sampled
        self.S = self.points.shape[0]

        # we group the basis vectors by their input and output representations
        _block_expansion_modules = {}
        
        # iterate through all different pairs of input/output representations
        # and, for each of them, build a basis
        for i_repr in unique_ever_seen(in_reprs):
            for o_repr in unique_ever_seen(out_reprs):
                reprs_names = (i_repr.name, o_repr.name)
                try:
                    
                    basis = basis_generator(i_repr, o_repr)
                    
                    block_expansion = block_basisexpansion(basis, points, basis_filter=basis_filter, recompute=recompute)
                    _block_expansion_modules[reprs_names] = block_expansion
                    
                    # register the block expansion as a submodule
                    self.add_module(f"block_expansion_{self._escape_pair(reprs_names)}", block_expansion)
                    
                except EmptyBasisException:
                    # print(f"Empty basis at {reprs_names}")
                    pass
        
        if len(_block_expansion_modules) == 0:
            print('WARNING! The basis for the block expansion of the filter is empty!')

        # the list of all pairs of input/output representations which don't have an empty basis
        self._representations_pairs = sorted(list(_block_expansion_modules.keys()))
        
        self._n_pairs = len(set(in_reprs)) * len(set(out_reprs))

        # retrieve for each representation in both input and output fields:
        # - the number of its occurrences,
        # - the indices where it occurs and
        # - whether its occurrences are contiguous or not
        self._in_count, _in_indices, _in_contiguous = retrieve_indices(in_reprs)
        self._out_count, _out_indices, _out_contiguous = retrieve_indices(out_reprs)
        
        self._weights_ranges = {}

        last_weight_position = 0

        self._contiguous = {}
        
        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        for io_pair in self._representations_pairs:
    
            self._contiguous[io_pair] = _in_contiguous[io_pair[0]] and _out_contiguous[io_pair[1]]
    
            # build the indices tensors
            if self._contiguous[io_pair]:
                # in_indices = torch.LongTensor([
                in_indices = [
                    _in_indices[io_pair[0]].min(),
                    _in_indices[io_pair[0]].max() + 1,
                    _in_indices[io_pair[0]].max() + 1 - _in_indices[io_pair[0]].min()
                ]# )
                # out_indices = torch.LongTensor([
                out_indices = [
                    _out_indices[io_pair[1]].min(),
                    _out_indices[io_pair[1]].max() + 1,
                    _out_indices[io_pair[1]].max() + 1 - _out_indices[io_pair[1]].min()
                ] #)
                
                setattr(self, 'in_indices_{}'.format(self._escape_pair(io_pair)), in_indices)
                setattr(self, 'out_indices_{}'.format(self._escape_pair(io_pair)), out_indices)

            else:
                out_indices, in_indices = torch.meshgrid([_out_indices[io_pair[1]], _in_indices[io_pair[0]]], indexing='ij')
                in_indices = in_indices.reshape(-1)
                out_indices = out_indices.reshape(-1)
                
                # register the indices tensors and the bases tensors as parameters of this module
                self.register_buffer('in_indices_{}'.format(self._escape_pair(io_pair)), in_indices, persistent=False)
                self.register_buffer('out_indices_{}'.format(self._escape_pair(io_pair)), out_indices, persistent=False)

            # number of occurrences of the input/output pair `io_pair`
            n_pairs = self._in_count[io_pair[0]] * self._out_count[io_pair[1]]
            
            # count the actual number of parameters
            total_weights = _block_expansion_modules[io_pair].dimension() * n_pairs

            # evaluate the indices in the global weights tensor to use for the basis belonging to this group
            self._weights_ranges[io_pair] = (last_weight_position, last_weight_position + total_weights)
    
            # increment the position counter
            last_weight_position += total_weights
        
        self._dim = last_weight_position

    def _escape_name(self, name: str):
        return name.replace('.', '^')

    def _escape_pair(self, pair):
        assert isinstance(pair, tuple), pair
        assert len(pair) == 2, len(pair)
        return (self._escape_name(pair[0]), self._escape_name(pair[1]))

    def get_element_info(self, idx: int) -> Dict:
        
        assert 0 <= idx < self._dim, idx
        
        reprs_names = None
        relative_idx = None
        for pair, idx_range in self._weights_ranges.items():
            if idx_range[0] <= idx < idx_range[1]:
                reprs_names = pair
                relative_idx = idx - idx_range[0]
                break
        assert reprs_names is not None and relative_idx is not None

        block_expansion = getattr(self, f"block_expansion_{self._escape_pair(reprs_names)}")
        block_idx = relative_idx // block_expansion.dimension()
        relative_idx = relative_idx % block_expansion.dimension()
        
        attr = block_expansion.get_element_info(relative_idx).copy()
        
        block_count = 0
        out_irreps_count = 0
        for o, o_repr in enumerate(self._out_reprs):
            in_irreps_count = 0
            for i, i_repr in enumerate(self._in_reprs):
            
                if reprs_names == (i_repr.name, o_repr.name):
                    
                    if block_count == block_idx:

                        # retrieve the attributes of each basis element and build a new list of
                        # attributes adding information specific to the current block
                        attr.update({
                            "in_irreps_position": in_irreps_count + attr["in_irrep_idx"],
                            "out_irreps_position": out_irreps_count + attr["out_irrep_idx"],
                            "in_repr": reprs_names[0],
                            "out_repr": reprs_names[1],
                            "in_field_position": i,
                            "out_field_position": o,
                        })
                    
                        # # build the ids of the basis vectors
                        # # add names and indices of the input and output fields
                        # id = '({}-{},{}-{})'.format(i_repr.name, i, o_repr.name, o)
                        # # add the original id in the block submodule
                        # id += "_" + attr["id"]
                        #
                        # # update with the new id
                        # attr["id"] = id
                        #
                        # attr["idx"] = idx
                        attr['block_id'] = attr['id']
                        attr['id'] = idx
                        
                        return attr
                        
                    block_count += 1
                
                in_irreps_count += len(i_repr.irreps)
            out_irreps_count += len(o_repr.irreps)
        
        raise ValueError(f"Parameter with index {idx} not found!")

    def get_basis_info(self) -> Iterable[Dict]:
        
        out_irreps_counts = [0]
        out_block_counts = defaultdict(list)
        for o, o_repr in enumerate(self._out_reprs):
            out_irreps_counts.append(out_irreps_counts[-1] + len(o_repr.irreps))
            out_block_counts[o_repr.name].append(o)
            
        in_irreps_counts = [0]
        in_block_counts = defaultdict(list)
        for i, i_repr in enumerate(self._in_reprs):
            in_irreps_counts.append(in_irreps_counts[-1] + len(i_repr.irreps))
            in_block_counts[i_repr.name].append(i)

        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        idx = 0
        for reprs_names in self._representations_pairs:

            block_expansion = getattr(self, f"block_expansion_{self._escape_pair(reprs_names)}")
            
            # since this method returns an iterable of attributes built on the fly, load all attributes first and then
            # iterate on this list
            attrs = list(block_expansion.get_basis_info())
            
            for o in out_block_counts[reprs_names[1]]:
                out_irreps_count = out_irreps_counts[o]
                for i in in_block_counts[reprs_names[0]]:
                    in_irreps_count = in_irreps_counts[i]
    
                    # retrieve the attributes of each basis element and build a new list of
                    # attributes adding information specific to the current block
                    for attr in attrs:
                        attr = attr.copy()
                        attr.update({
                            "in_irreps_position": in_irreps_count + attr["in_irrep_idx"],
                            "out_irreps_position": out_irreps_count + attr["out_irrep_idx"],
                            "in_repr": reprs_names[0],
                            "out_repr": reprs_names[1],
                            "in_field_position": i,
                            "out_field_position": o,
                        })
            
                        # build the ids of the basis vectors
                        # add names and indices of the input and output fields
                        # id = '({}-{},{}-{})'.format(reprs_names[0], i, reprs_names[1], o)
                        # # add the original id in the block submodule
                        # id += "_" + attr["id"]
                        #
                        # # update with the new id
                        # attr["id"] = id
                        #
                        # attr["idx"] = idx
                        
                        attr['block_id'] = attr['id']
                        attr['id'] = idx
                        
                        assert idx < self._dim
                        
                        idx += 1
                
                        yield attr

    def dimension(self) -> int:
        return self._dim

    def _expand_block(self, weights, io_pair):
        # retrieve the basis
        block_expansion = getattr(self, f"block_expansion_{self._escape_pair(io_pair)}")

        # retrieve the linear coefficients for the basis expansion
        coefficients = weights[self._weights_ranges[io_pair][0]:self._weights_ranges[io_pair][1]]
    
        # reshape coefficients for the batch matrix multiplication
        coefficients = coefficients.view(-1, block_expansion.dimension())
        
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        _filter = block_expansion(coefficients)
        k, o, i, p = _filter.shape
        
        _filter = _filter.view(
            self._out_count[io_pair[1]],
            self._in_count[io_pair[0]],
            o,
            i,
            self.S,
        )
        _filter = _filter.transpose(1, 2)
        return _filter
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the Module which expands the basis and returns the filter built

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters

        Returns:
            the filter built

        """
        assert weights.shape[0] == self.dimension()
        assert len(weights.shape) == 1
    
        if self._n_pairs == 1:
            # if there is only one block (i.e. one type of input field and one type of output field),
            #  we can return the expanded block immediately, instead of copying it inside a preallocated empty tensor
            io_pair = self._representations_pairs[0]
            in_indices = getattr(self, f"in_indices_{self._escape_pair(io_pair)}")
            out_indices = getattr(self, f"out_indices_{self._escape_pair(io_pair)}")
            _filter = self._expand_block(weights, io_pair).reshape(out_indices[2], in_indices[2], self.S)
            
        else:

            # to support Automatic Mixed Precision (AMP), we can not preallocate the output tensor with a specific dtype
            # Instead, we check the dtype of the first `expanded` block. For this reason, we postpose the allocation
            # of the full _filter tensor

            _filter = None

            # iterate through all input-output field representations pairs
            for io_pair in self._representations_pairs:
                
                # retrieve the indices
                in_indices = getattr(self, f"in_indices_{self._escape_pair(io_pair)}")
                out_indices = getattr(self, f"out_indices_{self._escape_pair(io_pair)}")
                
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter
                expanded = self._expand_block(weights, io_pair)

                if _filter is None:
                    # build the tensor which will contain the filter
                    # this lazy strategy allows us to use expanded.dtype which is dynamically chosen by PyTorch's AMP
                    _filter = torch.zeros(
                        self._output_size, self._input_size, self.S, device=weights.device, dtype=expanded.dtype
                    )

                if self._contiguous[io_pair]:
                    _filter[
                        out_indices[0]:out_indices[1],
                        in_indices[0]:in_indices[1],
                        :,
                    ] = expanded.reshape(out_indices[2], in_indices[2], self.S)
                else:
                    _filter[
                        out_indices,
                        in_indices,
                        :,
                    ] = expanded.reshape(-1, self.S)

            # just in case
            if _filter is None:
                # build the tensor which will contain the filter
                _filter = torch.zeros(
                    self._output_size, self._input_size, self.S, device=weights.device, dtype=weights.dtype
                )

        # return the new filter
        return _filter

    def __hash__(self):

        _hash = 0
        for io in self._representations_pairs:
            n_pairs = self._in_count[io[0]] * self._out_count[io[1]]
            _hash += hash(getattr(self, f"block_expansion_{self._escape_pair(io)}")) * n_pairs

        return _hash

    def __eq__(self, other):
        if not isinstance(other, BlocksBasisExpansion):
            return False

        if self._dim != other._dim:
            return False

        if self._representations_pairs != other._representations_pairs:
            return False

        for io in self._representations_pairs:
            if self._contiguous[io] != other._contiguous[io]:
                return False

            if self._weights_ranges[io] != other._weights_ranges[io]:
                return False

            io_escaped = self._escape_pair(io)

            if self._contiguous[io]:
                if getattr(self, f"in_indices_{io_escaped}") != getattr(other, f"in_indices_{io_escaped}"):
                    return False
                if getattr(self, f"out_indices_{io_escaped}") != getattr(other, f"out_indices_{io_escaped}"):
                    return False
            else:
                if torch.any(getattr(self, f"in_indices_{io_escaped}") != getattr(other, f"in_indices_{io_escaped}")):
                    return False
                if torch.any(getattr(self, f"out_indices_{io_escaped}") != getattr(other, f"out_indices_{io_escaped}")):
                    return False

            if getattr(self, f"block_expansion_{io_escaped}") != getattr(other, f"block_expansion_{io_escaped}"):
                return False

        return True

#
# def _retrieve_indices(reprs: List[Representation]):
#     fiber_position = 0
#     _indices = defaultdict(list)
#     _count = defaultdict(int)
#     _contiguous = {}
#
#     for repr in reprs:
#         _indices[repr.name] += list(range(fiber_position, fiber_position + repr.size))
#         fiber_position += repr.size
#         _count[repr.name] += 1
#
#     for name, indices in _indices.items():
#         # _contiguous[o_name] = indices == list(range(indices[0], indices[0]+len(indices)))
#         _contiguous[name] = utils.check_consecutive_numbers(indices)
#         _indices[name] = torch.LongTensor(indices)
#
#     return _count, _indices, _contiguous
#
