
from .basis import KernelBasis, EmptyBasisException
from .steerable_filters_basis import SteerableFiltersBasis

from escnn.group import Group
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation
from escnn.utils import unique_ever_seen

import torch

from typing import Type, Union, Tuple, Dict, List, Iterable, Callable, Set
from abc import ABC, abstractmethod
from collections import defaultdict


class IrrepBasis(KernelBasis):
    
    def __init__(self,
                 basis: SteerableFiltersBasis,
                 in_irrep: Union[IrreducibleRepresentation, Tuple],
                 out_irrep: Union[IrreducibleRepresentation, Tuple],
                 dim: int,
                 harmonics: List[Tuple] = None
                 ):
        r"""

        Abstract class for bases implementing the kernel constraint solutions associated to irreducible input and output
        representations.

        .. note ::
            The steerable *filter* ``basis`` is not necessarily associated with the same group as ``in_irrep`` and
            ``out_irrep``.
            For instance, :class:`~escnn.kernels.RestrictedWignerEckartBasis` uses a larger group to define ``basis``.
            The attribute ``IrrepBasis.group``, instead, refers to the equivariance group of this steerable *kernel*
            basis and is the same group of ``in_irrep`` and ``out_irrep``.
            The irreps in the list ``harmonics`` refer to the group in the steerable filter ``basis``,
            and not to ``IrrepBasis.group``.

        Args:
            basis (SteerableFiltersBasis): the steerable basis used to parameterize scalar filters and generate the kernel solutions
            in_irrep (IrreducibleRepresentation): the input irrep
            out_irrep (IrreducibleRepresentation): the output irrep
            dim (int): the number of elements in the basis
            harmonics (list, optional): optionally, use only a subset of the steerable filters in ``basis``. This list
                                        defines a subset of the group's irreps and is used to select only the steerable
                                        basis filters which transform according to these irreps.

        Attributes:
            ~.group (Group): the equivariance group
            ~.in_irrep (IrreducibleRepresentation): the input irrep
            ~.out_irrep (IrreducibleRepresentation): the output irrep
            ~.basis (SteerableFiltersBasis): the steerable basis used to parameterize scalar filters

        """

        super(IrrepBasis, self).__init__(dim, (out_irrep.size, in_irrep.size))

        assert in_irrep.group == out_irrep.group
        self.group: Group = in_irrep.group
        self.in_irrep: IrreducibleRepresentation = self.group.irrep(*self.group.get_irrep_id(in_irrep))
        self.out_irrep: IrreducibleRepresentation = self.group.irrep(*self.group.get_irrep_id(out_irrep))
        self.basis: SteerableFiltersBasis = basis

        self.js = []
        harmonics = set(harmonics)
        for j, _ in basis.js:
            if j not in self.js and (harmonics is None or j in harmonics):
                self.js.append(j)

        self._start_index = {}
        idx = 0
        for _j in self.js:
            self._start_index[_j] = idx
            idx += self.dim_harmonic(_j)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``points`` must be an array of shape `(N, d)`, where `N` is the number of points and `d` is the dimensionality
        of the Euclidean space where filters are defined.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
    
        assert len(points.shape) == 2
        S = points.shape[0]
    
        if out is None:
            out = torch.empty(S, self.dim, self.shape[0], self.shape[1], device=points.device, dtype=points.dtype)
    
        assert out.shape == (S, self.dim, self.shape[0], self.shape[1])

        steerable_basis = self.basis.sample_as_dict(points)

        B = 0
        outs = {}
        for b, j in enumerate(self.js):
            outs[j] = out[:, B:B + self.dim_harmonic(j), ...]
            B += self.dim_harmonic(j)

        self.sample_harmonics(steerable_basis, outs)
        return out

    @abstractmethod
    def sample_harmonics(self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[Tuple, torch.Tensor]:
        r"""

        Sample the continuous basis elements on the discrete set of points.
        Rather than using the points' coordinates, the method directly takes in input the steerable basis elements
        sampled on this points using the method :meth:`escnn.kernels.SteerableFilterBasis.sample_as_dict` of
        ``self.basis``.

        Similarly, rather than returning a single tensor containing all sampled basis elements, it groups basis elements
        by the ``G``-irrep acting on them.
        The method returns a dictionary mapping each irrep's ``id`` to a tensor of shape `(N, m, o, i)`, where
        `N` is the number of points,
        `m` is the multiplicity of the irrep (see :meth:`~escnn.kernels.SteerableKernelBasis.dim_harmonic`)
        and `o, i` is the number of input and output channels (see the ``shape`` attribute).

        Optionally, store the resulting tensors in ``out``, rather than allocating new memory.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        raise NotImplementedError()

    @abstractmethod
    def dim_harmonic(self, j: Tuple) -> int:
        r'''
        Number of kernel basis elements generated from elements of the steerable filter basis (``self.basis``) which
        transform according to the ``self.basis.group``-irrep identified by ``j``.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def attrs_j_iter(self, j: Tuple) -> Iterable:
        raise NotImplementedError()
    
    @abstractmethod
    def attrs_j(self, j: Tuple, idx) -> Dict:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _generator(cls,
                   basis: SteerableFiltersBasis,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   **kwargs
    ) -> 'IrrepBasis':
        raise NotImplementedError()


class SteerableKernelBasis(KernelBasis):
    
    def __init__(self,
                 basis: SteerableFiltersBasis,
                 in_repr: Representation,
                 out_repr: Representation,
                 irreps_basis: Type[IrrepBasis],
                 **kwargs):
        r"""
        
        Implements a general basis for the vector space of equivariant kernels over an Euclidean space :math:`X=\R^n`.
        A :math:`G`-equivariant kernel :math:`\kappa`, mapping between an input field, transforming under
        :math:`\rho_\text{in}` (``in_repr``), and an output field, transforming under  :math:`\rho_\text{out}`
        (``out_repr``), satisfies the following constraint:
        
        .. math ::
            
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1} \qquad \forall g \in G, \forall x \in X
        
        As the kernel constraint is a linear constraint, the space of equivariant kernels is a vector subspace of the
        space of all convolutional kernels. It follows that any equivariant kernel can be expressed in terms of a basis
        of this space.
        
        This class solves the kernel constraint for two arbitrary representations by combining the solutions of the
        kernel constraints associated to their :class:`~escnn.group.IrreducibleRepresentation` s.
        In order to do so, it relies on ``irreps_basis`` which solves individual irreps constraints. ``irreps_basis``
        must be a class (subclass of :class:`~escnn.kernels.IrrepsBasis`) which builds a basis for equivariant
        kernels associated with irreducible representations when instantiated.
        
        The groups :math:`G` which are currently implemented are origin-preserving isometries (what are called
        structure groups, or sometimes gauge groups, in the language of
        `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ ).
        The origin-preserving isometries of :math:`\R^d` are subgroups of :math:`O(d)`, i.e. reflections and rotations.
        Therefore, equivariance does not enforce any constraint on the radial component of the kernels.
        Hence, this class only implements a basis for the angular part of the kernels.
        
        .. warning ::
            
            Typically, the user does not need to manually instantiate this class.
            Instead, we suggest to use the interface provided in :doc:`escnn.gspaces`.
        
        Args:
            basis (SteerableFiltersBasis): a steerable basis for scalar filters over the base space
            in_repr (Representation): Representation associated with the input feature field
            out_repr (Representation): Representation associated with the output feature field
            irreps_basis (class): class defining the irreps basis. This class is instantiated for each pair of irreps to solve all irreps constraints.
            **kwargs: additional arguments used when instantiating ``irreps_basis``

        Attributes:
            ~.group (Group): the equivariance group ``G``.
            ~.in_repr (Representation): the input representation
            ~.out_repr (Representation): the output representation

        """
        
        assert in_repr.group == out_repr.group
        
        self.in_repr: Representation = in_repr
        self.out_repr: Representation = out_repr
        group = in_repr.group
        self.group: Group = group
        
        self._irrep_basis = irreps_basis
        self._irrep__basis_kwargs = kwargs

        ################

        # Dict[Tuple, IrrepsBasis]:
        self.irreps_bases = {}

        js = set()

        # loop over all input irreps
        for i_irrep_id in unique_ever_seen(in_repr.irreps):
            # loop over all output irreps
            for o_irrep_id in unique_ever_seen(out_repr.irreps):
                try:
                    # retrieve the irrep intertwiner basis
                    intertwiner_basis = irreps_basis._generator(basis, i_irrep_id, o_irrep_id, **kwargs)
                    assert intertwiner_basis.group == self.group
                    self.irreps_bases[(i_irrep_id, o_irrep_id)] = intertwiner_basis
                    # compute the set of all harmonics needed by all intertwiners bases
                    # in this way, we can precompute the embedding of the points with the harmonics and reuse the same
                    # embeddings for all irreps bases
                    js.update(intertwiner_basis.js)
                except EmptyBasisException:
                    # if the basis is empty, skip it
                    pass

        self._dim_harmonics = defaultdict(int)
        self.bases = [[None for _ in range(len(out_repr.irreps))] for _ in range(len(in_repr.irreps))]
        dim = 0
        # loop over all input irreps
        for ii, i_irrep_id in enumerate(in_repr.irreps):
            # loop over all output irreps
            for oo, o_irrep_id in enumerate(out_repr.irreps):
                if (i_irrep_id, o_irrep_id) in self.irreps_bases:
                    self.bases[ii][oo] = self.irreps_bases[(i_irrep_id, o_irrep_id)]
                    dim += self.irreps_bases[(i_irrep_id, o_irrep_id)].dim
                    for j in self.bases[ii][oo].js:
                        self._dim_harmonics[j] += self.irreps_bases[(i_irrep_id, o_irrep_id)].dim_harmonic(j)

        ################

        # before registering tensors as buffers and sub-modules, we need to call torch.nn.Module.__init__()
        super(SteerableKernelBasis, self).__init__(dim, (out_repr.size, in_repr.size))

        self.basis = basis

        for io_pair, intertwiner_basis in self.irreps_bases.items():
            self.add_module(f'basis_{io_pair}', intertwiner_basis)

        ################

        A_inv = torch.tensor(in_repr.change_of_basis_inv, dtype=torch.float32).clone()
        B = torch.tensor(out_repr.change_of_basis, dtype=torch.float32).clone()

        if not torch.allclose(A_inv, torch.eye(in_repr.size)):
            self.register_buffer('A_inv', A_inv)
        else:
            self.A_inv = None

        if not torch.allclose(B, torch.eye(out_repr.size)):
            self.register_buffer('B', B)
        else:
            self.B = None

        self.js = [j for j, m in self.basis.js if j in js]

        if self.basis.group.trivial_representation.id in self.js:
            # make sure that the harmonic corresponding to the trivial representation is the first in the list.
            self.js.remove(self.basis.group.trivial_representation.id)
            self.js = [self.basis.group.trivial_representation.id] + self.js
        
        self.in_sizes = []
        self.out_sizes = []
        # loop over all input irreps
        for ii, i_irrep_id in enumerate(in_repr.irreps):
            self.in_sizes.append(group.irrep(*i_irrep_id).size)
            
        # loop over all output irreps
        for oo, o_irrep_id in enumerate(out_repr.irreps):
            self.out_sizes.append(group.irrep(*o_irrep_id).size)

        self._slices = defaultdict(dict)
        basis_count = defaultdict(int)
        in_position = 0
        for ii, in_size in enumerate(self.in_sizes):
            out_position = 0
            for oo, out_size in enumerate(self.out_sizes):
                if self.bases[ii][oo] is not None:
                    for j in self.bases[ii][oo].js:
                        self._slices[(ii, oo)][j] = (
                            out_position,
                            out_position + out_size,
                            in_position,
                            in_position + in_size,
                            basis_count[j],
                            basis_count[j] + self.bases[ii][oo].dim_harmonic(j)
                        )
                        basis_count[j] += self.bases[ii][oo].dim_harmonic(j)
                out_position += out_size
            in_position += in_size

    def dim_harmonic(self, j: Tuple) -> int:
        r'''
            Number of kernel basis elements generated from elements of the steerable filter basis (``self.basis``) which
            transform according to the ``self.basis.group``-irrep identified by ``j``.
        '''
        return self._dim_harmonics[j]

    def compute_harmonics(self, points: torch.Tensor) -> Dict[Tuple, torch.Tensor]:
        r"""
        Pre-compute the sampled steerable filter basis over a set of point.
        This is an alias for ``self.basis.sample_as_dict(points)``.

        .. seealso ::
            :meth:`escnn.kernels.SteerableFiltersBasis.sample_as_dict`.

        """
        return self.basis.sample_as_dict(points)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.
        
        ``points`` must be an array of shape `(N, d)`, where `N` is the number of points.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis
            
        """
        assert len(points.shape) == 2
        S = points.shape[0]

        if out is None:
            out = torch.zeros((S, self.dim, self.shape[0], self.shape[1]), device=points.device, dtype=points.dtype)
        else:
            out[:] = 0.
            
        assert out.shape == (S, self.dim, self.shape[0], self.shape[1])

        outs = {}
        B = 0
        for j in self.js:
            outs[j] = out[:, B:B + self.dim_harmonic(j), ...]
            B += self.dim_harmonic(j)

        steerable_basis = self.compute_harmonics(points)
        self.sample_harmonics(steerable_basis, outs)

        return out

    def sample_harmonics(self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[Tuple, torch.Tensor]:
        r"""
        Sample the continuous basis elements on the discrete set of points.
        Rather than using the points' coordinates, the method directly takes in input the steerable basis elements
        sampled on this points using the method :meth:`escnn.kernels.SteerableKernelBasis.compute_harmonics`.

        Similarly, rather than returning a single tensor containing all sampled basis elements, it groups basis elements
        by the ``G``-irrep acting on them.
        The method returns a dictionary mapping each irrep's ``id`` to a tensor of shape `(N, m, o, i)`, where
        `N` is the number of points,
        `m` is the multiplicity of the irrep (see :meth:`~escnn.kernels.SteerableKernelBasis.dim_harmonic`)
        and `o, i` is the number of input and output channels (see the ``shape`` attribute).

        Optionally, store the resulting tensors in ``out``, rather than allocating new memory.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        if out is None:
            out = {
                j: torch.zeros(
                    (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]),
                    device=points[j].device, dtype=points[j].dtype
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1])

        if self.A_inv is None and self.B is None:
            out = self._sample_direct_sum(points, out=out)
        else:
            samples = self._sample_direct_sum(points)
            out = self._change_of_basis(samples, out=out)
    
        return out

    def _sample_direct_sum(self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[Tuple, torch.Tensor]:
    
        if out is None:
            out = {
                j: torch.zeros(
                    (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]),
                    device=points[j].device, dtype=points[j].dtype
                )
                for j in self.js
            }
        # else:
        #     for j in self.js:
        #         if j in out:
        #             out[j][:] = 0.
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1])

        for ii, in_size in enumerate(self.in_sizes):
            for oo, out_size in enumerate(self.out_sizes):
                if self.bases[ii][oo] is not None:
                    slices = self._slices[(ii, oo)]
                    
                    blocks = {
                        j: out[j][:, b_s:b_e, o_s:o_e, i_s:i_e]
                        for j, (o_s, o_e, i_s, i_e, b_s, b_e) in slices.items()
                    }
                    self.bases[ii][oo].sample_harmonics(points, out=blocks)

        return out
    
    def _change_of_basis(self, samples: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[Tuple, torch.Tensor]:
        # multiply by the change of basis matrices to transform the irreps basis in the full representations basis

        if out is None:
            out = {j: None for j in self.js}
            
        for j in samples.keys():
            if self.A_inv is not None and self.B is not None:
                out[j][:] = torch.einsum("no,pboi,ij->pbnj", self.B.to(samples[j].dtype), samples[j], self.A_inv.to(samples[j].dtype))
            elif self.A_inv is not None:
                out[j][:] = torch.einsum("pboi,ij->pboj", samples[j], self.A_inv.to(samples[j].dtype))
            elif self.B is not None:
                out[j][:] = torch.einsum("no,pboi->pbni", self.B.to(samples[j].dtype), samples[j])
            else:
                out[j][...] = samples[j]
        
        return out

    def __getitem__(self, idx):
        assert idx < self.dim
        
        j_idx = idx
        for j in self.js:
            dim = self.dim_harmonic(j)
            if j_idx >= dim:
                j_idx -= dim
            else:
                break
        
        assert j_idx < self.dim_harmonic(j), (j_idx, self.dim_harmonic(j))

        count = 0
        for ii in range(len(self.in_sizes)):
            for oo in range(len(self.out_sizes)):
                if self.bases[ii][oo] is not None:
                    dim = self.bases[ii][oo].dim_harmonic(j)

                    rel_idx = j_idx - count
                    if rel_idx >= 0 and rel_idx < dim:
                        
                        attr = dict(self.bases[ii][oo].attrs_j(j, rel_idx))
                        
                        attr["shape"] = self.bases[ii][oo].shape

                        attr["irreps_basis_idx"] = attr["idx"]
                        attr["idx"] = idx
                        attr["j"] = j
                        attr["j_idx"] = j_idx

                        attr["in_irrep"] = self.in_repr.irreps[ii]
                        attr["out_irrep"] = self.out_repr.irreps[oo]
                        
                        attr["in_irrep_idx"] = ii
                        attr["out_irrep_idx"] = oo
                        
                        return attr
                
                    count += dim

    def __iter__(self):
        idx = 0
        for j in self.js:
            j_idx = 0
            for ii in range(len(self.in_sizes)):
                for oo in range(len(self.out_sizes)):
                    basis = self.bases[ii][oo]
                    if basis is not None:
                        
                        for attr in basis.attrs_j_iter(j):
    
                            attr["shape"] = basis.shape
                            
                            attr["in_irrep"] = self.in_repr.irreps[ii]
                            attr["out_irrep"] = self.out_repr.irreps[oo]
                        
                            attr["in_irrep_idx"] = ii
                            attr["out_irrep_idx"] = oo

                            attr["j"] = j
                            attr["j_idx"] = j_idx
                            
                            attr["irreps_basis_idx"] = attr["idx"]
                            attr["idx"] = idx
                            
                            assert idx < self.dim

                            yield attr
                            
                            idx += 1
                            j_idx += 1

    def __eq__(self, other):
        if not isinstance(other, SteerableKernelBasis):
            return False
        elif self.basis != other.basis or self.in_repr != other.in_repr or self.out_repr != other.out_repr:
            return False
        else:
            sbk1 = sorted(self.irreps_bases.keys())
            sbk2 = sorted(other.irreps_bases.keys())
            if sbk1 != sbk2:
                return False
            
            for irreps, basis in self.irreps_bases.items():
                if basis != other.irreps_bases[irreps]:
                    return False
            
            return True

    def __hash__(self):
        h = hash(self.in_repr) + hash(self.out_repr) + hash(self.basis)
        for basis in self.irreps_bases.items():
            h += hash(basis)
        return h
        
        


