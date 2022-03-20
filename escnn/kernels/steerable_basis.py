
import numpy as np

from .basis import KernelBasis, EmptyBasisException
from .spaces import SpaceIsomorphism

from escnn.group import Group
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation

from typing import Type, Union, Tuple, Dict, List, Iterable, Callable, Set
from abc import ABC, abstractmethod
from collections import defaultdict


class IrrepBasis(KernelBasis):
    
    def __init__(self,
                 X: SpaceIsomorphism,
                 in_irrep: Union[IrreducibleRepresentation, Tuple],
                 out_irrep: Union[IrreducibleRepresentation, Tuple],
                 js: List[Tuple],
                 dim: int):
        r"""

        Abstract class for bases implementing the kernel constraint solutions associated to irreducible input and output
        representations.

        Args:
            in_irrep:
            out_irrep:
            dim:
        """
        self.X = X
        assert in_irrep.group == out_irrep.group
        self.group = in_irrep.group
        self.in_irrep = self.group.irrep(*self.group.get_irrep_id(in_irrep))
        self.out_irrep = self.group.irrep(*self.group.get_irrep_id(out_irrep))
        
        self.js = js

        super(IrrepBasis, self).__init__(dim, (out_irrep.size, in_irrep.size))

        self._start_index = {}
        idx = 0
        for _j in self.js:
            self._start_index[_j] = idx
            idx += self.dim_harmonic(_j)

    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``points`` must be an array of shape `(d, N)`, where `N` is the number of points.

        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
    
        assert len(points.shape) == 2
        S = points.shape[1]
    
        if out is None:
            out = np.empty((self.shape[0], self.shape[1], self.dim, S))
    
        assert out.shape == (self.shape[0], self.shape[1], self.dim, S)

        B = 0
        harmonics = {}
        outs = {}
        for b, j in enumerate(self.js):
    
            Ys = self.X.basis(points, j)
    
            harmonics[j] = Ys
            
            outs[j] = out[:, :, B:B + self.dim_harmonic(j), :]
            B += self.dim_harmonic(j)

        self.sample_harmonics(harmonics, outs)
        return out
    
    @abstractmethod
    def sample_harmonics(self, points: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[Tuple, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def dim_harmonic(self, j: Tuple) -> int:
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
                   X: SpaceIsomorphism,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   harmonics: List[Tuple] = None,
                   **kwargs
    ) -> 'IrrepBasis':
        raise NotImplementedError()


class SteerableKernelBasis(KernelBasis):
    
    def __init__(self,
                 X: SpaceIsomorphism,
                 in_repr: Representation,
                 out_repr: Representation,
                 irreps_basis: Type[IrrepBasis],
                 harmonics: Union[List, Set] = None, # optionally filter only some harmonics
                 **kwargs):
        r"""
        
        Implements a general basis for the vector space of equivariant kernels over the homogeneous space :math:`X`.
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
        
        In order to build a complete basis of kernels, you should combine this basis with a basis which defines the
        radial profile (such as :class:`~escnn.kernels.GaussianRadialProfile`) through
        :class:`~escnn.kernels.SphericalShellsBasis`.
        
        .. math::
            
            \mathcal{B} = \left\{ b_i (r) :=  \exp \left( \frac{ \left( r - r_i \right)^2}{2 \sigma_i^2} \right) \right\}_i
        
        .. warning ::
            
            Typically, the user does not need to manually instantiate this class.
            Instead, we suggest to use the interface provided in :doc:`escnn.gspaces`.
        
        Args:
            X (SpaceIsomorphism): the base space where the steerable kernel is defined
            in_repr (Representation): Representation associated with the input feature field
            out_repr (Representation): Representation associated with the output feature field
            irreps_basis (class): class defining the irreps basis. This class is instantiated for each pair of irreps to solve all irreps constraints.
            harmonics (optional): selects only a subset of the harmonics to use.
            **kwargs: additional arguments used when instantiating ``irreps_basis``
            
        """
        
        assert in_repr.group == out_repr.group
        
        self.X = X
        self.in_repr = in_repr
        self.out_repr = out_repr
        group = in_repr.group
        self.group = group
        
        self._irrep_basis = irreps_basis
        self._irrep__basis_kwargs = kwargs

        A_inv = np.array(in_repr.change_of_basis_inv, copy=True)
        B = np.array(out_repr.change_of_basis, copy=True)
        
        # A_inv = in_repr.change_of_basis_inv
        # B = out_repr.change_of_basis

        if not np.allclose(A_inv, np.eye(in_repr.size)):
            self.A_inv = A_inv
        else:
            self.A_inv = None
            
        if not np.allclose(B, np.eye(out_repr.size)):
            self.B = B
        else:
            self.B = None

        # Dict[Tuple, IrrepsBasis]:
        self.irreps_bases = {}
        
        js = set()
        
        # loop over all input irreps
        for i_irrep_id in set(in_repr.irreps):
            # loop over all output irreps
            for o_irrep_id in set(out_repr.irreps):
        
                try:
                    # retrieve the irrep intertwiner basis
                    basis = irreps_basis._generator(self.X, i_irrep_id, o_irrep_id, harmonics=harmonics, **kwargs)
                    assert basis.group == self.group

                    self.irreps_bases[(i_irrep_id, o_irrep_id)] = basis
                    
                    # compute the set of all harmonics needed by all irreps bases
                    # in this way, we can precompute the embedding of the points with the harmonics and reuse the same
                    # embeddings for all irreps bases
                    js.update(basis.js)

                except EmptyBasisException:
                    # if the basis is empty, skip it
                    pass
        
        if callable(harmonics):
            js = list(j for j in js if harmonics(j))
        elif isinstance(harmonics, set) or isinstance(harmonics, list):
            js = js.intersection(set(harmonics))
        
        self.js = sorted(list(js))
        if self.X.zero_harmonic in self.js:
            # make sure that the harmonic corresponding to the trivial representation (i.e. the harmonic spanning
            # the space of constant functions over the space) is the first in the list.
            self.js.remove(self.X.zero_harmonic)
            self.js = [self.X.zero_harmonic] + self.js
        
        self.bases = [[None for _ in range(len(out_repr.irreps))] for _ in range(len(in_repr.irreps))]
        
        self.in_sizes = []
        self.out_sizes = []
        # loop over all input irreps
        for ii, i_irrep_id in enumerate(in_repr.irreps):
            self.in_sizes.append(group.irrep(*i_irrep_id).size)
            
        # loop over all output irreps
        for oo, o_irrep_id in enumerate(out_repr.irreps):
            self.out_sizes.append(group.irrep(*o_irrep_id).size)

        self._dim_harmonics = defaultdict(int)
        dim = 0
        # loop over all input irreps
        for ii, i_irrep_id in enumerate(in_repr.irreps):
            # loop over all output irreps
            for oo, o_irrep_id in enumerate(out_repr.irreps):
                if (i_irrep_id, o_irrep_id) in self.irreps_bases:
                    self.bases[ii][oo] = self.irreps_bases[(i_irrep_id, o_irrep_id)]
                    dim += self.bases[ii][oo].dim
                    for j in self.bases[ii][oo].js:
                        self._dim_harmonics[j] += self.irreps_bases[(i_irrep_id, o_irrep_id)].dim_harmonic(j)
                        
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

        super(SteerableKernelBasis, self).__init__(dim, (out_repr.size, in_repr.size))
        
    def dim_harmonic(self, j: Tuple) -> int:
        return self._dim_harmonics[j]

    def compute_harmonics(self, points: np.ndarray) -> Dict[Tuple, np.ndarray]:
        harmonics = {}
        for j in self.js:
            Ys = self.X.basis(points, j)
            harmonics[j] = Ys

        return harmonics

    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.
        
        ``points`` must be an array of shape `(d, N)`, where `N` is the number of points.

        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis
            
        """
        assert len(points.shape) == 2

        if out is None:
            out = np.zeros((self.shape[0], self.shape[1], self.dim, points.shape[1]))
        else:
            out.fill(0.)
            
        assert out.shape == (self.shape[0], self.shape[1], self.dim, points.shape[1])

        harmonics = {}
        outs = {}
        B = 0
        for j in self.js:
            Ys = self.X.basis(points, j)
    
            harmonics[j] = Ys
    
            outs[j] = out[:, :, B:B + self.dim_harmonic(j), :]
            B += self.dim_harmonic(j)

        self.sample_harmonics(harmonics, outs)
        return out

    def sample_harmonics(self, points: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[Tuple, np.ndarray]:
        if out is None:
            out = {
                j: np.zeros((self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1]))
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1])
    
        if self.A_inv is None and self.B is None:
            out = self._sample_direct_sum(points, out=out)
        else:
            samples = self._sample_direct_sum(points)
            out = self._change_of_basis(samples, out=out)
    
        return out

    def _sample_direct_sum(self, points: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[Tuple, np.ndarray]:
    
        if out is None:
            out = {
                j: np.zeros((self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1]))
                for j in self.js
            }
        else:
            for j in self.js:
                if j in out:
                    out[j].fill(0)
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1])

        for ii, in_size in enumerate(self.in_sizes):
            for oo, out_size in enumerate(self.out_sizes):
                if self.bases[ii][oo] is not None:
                    slices = self._slices[(ii, oo)]
                    
                    blocks = {
                        j: out[j][o_s:o_e, i_s:i_e, b_s:b_e, ...]
                        for j, (o_s, o_e, i_s, i_e, b_s, b_e) in slices.items()
                    }
                    self.bases[ii][oo].sample_harmonics(points, out=blocks)

        return out
    
    def _change_of_basis(self, samples: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[Tuple, np.ndarray]:
        # multiply by the change of basis matrices to transform the irreps basis in the full representations basis

        if out is None:
            out = {j: None for j in self.js}
            
        for j in samples.keys():
            if self.A_inv is not None and self.B is not None:
                out[j] = np.einsum("no,oibp,ij->njbp", self.B, samples[j], self.A_inv, out=out[j])
            elif self.A_inv is not None:
                out[j] = np.einsum("oibp,ij->ojbp", samples[j], self.A_inv, out=out[j])
            elif self.B is not None:
                out[j] = np.einsum("no,oibp->nibp", self.B, samples[j], out=out[j])
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
        
        assert j_idx < self.dim_harmonic(j)

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
        elif self.X != other.X or self.in_repr != other.in_repr or self.out_repr != other.out_repr:
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
        h = hash(self.in_repr) + hash(self.out_repr) + hash(self.X)
        for basis in self.irreps_bases.items():
            h += hash(basis)
        return h
        
        


