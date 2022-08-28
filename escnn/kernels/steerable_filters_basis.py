
from .basis import KernelBasis, EmptyBasisException

from escnn.group import Group
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation

import torch
from escnn.kernels import utils

from typing import Type, Union, Tuple, Dict, List, Iterable, Callable, Set
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class SteerableFiltersBasis(KernelBasis):
    
    def __init__(self,
                 G: Group,
                 action: Representation,
                 js: List[Tuple],
        ):
        r"""

        Abstract class for bases implementing a :math:`G`-steerable basis for scalar filters

        """
        assert isinstance(G, Group)

        # Group: the group acting on the steerable basis
        self.group: Group = G

        # Representation: the representation of the group ``G`` defining its action on the Euclidean space
        assert action.group == self.group
        self.action = action

        assert isinstance(js, list)
        # List: list of irreps (and their multiplicity) describing how each invariant steerable subspace transform
        self.js = js

        self._js = {}
        dim = 0
        for j, m in js:
            assert isinstance(j, tuple)
            # check it corresponds to an irrep
            psi_j = self.group.irrep(*j)

            # the second entry represents the multiplicity of the irrep
            assert isinstance(m, int)
            self._js[j] = m

            dim += psi_j.size * m

        # This is a Filter basis, so it assumes 1 input and 1 output channels
        super(SteerableFiltersBasis, self).__init__(dim, (1, 1))

        self._start_index = {}
        idx = 0
        for _j, m in self.js:
            self._start_index[_j] = idx
            idx += self.dim_harmonic(_j)

    @property
    def dimensionality(self) -> int:
        """
            The dimensionality of the base space on which the scalar filters are defined.
        """
        return self.action.size

    @abstractmethod
    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``points`` must be an array of shape `(d, N)`, where `N` is the number of points and `d` is equal to
        :meth:`~escnn.kernels.SteerableFilterBasis.dimensionality`.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        raise NotImplementedError
        # assert len(points.shape) == 2
        # S = points.shape[1]
        # assert points.shape[0] == self.dimensionality, (points.shape, self.dimensionality)
        #
        # if out is None:
        #     out = torch.empty(1, 1, self.dim, S, device=points.device, dtype=points.dtype)
        #
        # assert out.shape == (1, 1, self.dim, S)
        #
        # B = 0
        # for b, j in enumerate(self.js):
        #     self.sample_j(points, j, out=out[:, :, B:B + self.dim_harmonic(j), :])
        #     B += self.dim_harmonic(j)
        #
        # return out

    def sample_as_dict(self, points: torch.Tensor, out: torch.Tensor = None) -> Dict[Tuple, torch.Tensor]:

        S = points.shape[1]

        if out is not None:
            assert out.shape == (self.dim, S), (out.shape, self.dim, S)
            out = out.view(1, 1, self.dim, S)

        out = self.sample(points, out)

        out = out.view(self.dim, S)

        basis = {}
        p = 0
        for j, m in self.js:
            psi = self.group.irrep(*j)
            dim = psi.size * m
            basis[j] = out[p : p+dim, :].view(m, psi.size, S)
            p += dim

        return basis

    def dim_harmonic(self, j: Tuple) -> int:
        psi = self.group.irrep(*j)
        if j in self._js:
            return psi.size * self._js[j]
        else:
            return 0

    def multiplicity(self, j: Tuple) -> int:
        if j in self._js:
            return self._js[j]
        else:
            return 0

    @abstractmethod
    def steerable_attrs_iter(self) -> Iterable:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        raise NotImplementedError()
    
    @abstractmethod
    def steerable_attrs(self, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        raise NotImplementedError()

    @abstractmethod
    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        raise NotImplementedError()

    @abstractmethod
    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        raise NotImplementedError()

    def check_equivariance(self):
        # Verify the steerability property of the basis

        S = 20
        points = torch.randn(self.dimensionality, S)

        basis = self.sample_as_dict(points)

        for _ in range(10):
            g = self.group.sample()

            points_g = torch.tensor(self.action(g), device=points.device, dtype=points.dtype) @ points
            basis_g = self.sample_as_dict(points_g)

            g_basis = {
                j: torch.einsum(
                    'ij,abmjp->abmip',
                    torch.tensor(self.group.irrep(*j)(g), device=points.device, dtype=points.dtype),
                    basis_j
                )
                for j, basis_j in basis.items()
            }

            for j, m in self.js:
                dim = self.group.irrep(*j).size
                assert basis_g[j].shape == (m, dim, S), (basis_g[j].shape, m, dim, S)
                assert g_basis[j].shape == (m, dim, S), (g_basis[j].shape, m, dim, S)
                assert torch.allclose(g_basis[j], basis_g[j], atol=1e-6, rtol=1e-4)


class PointBasis(SteerableFiltersBasis):

    def __init__(self, G: Group):
        super(PointBasis, self).__init__(G, G.trivial_representation, [(G.trivial_representation.id, 1)])

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:

        assert len(points.shape) == 2
        assert points.shape[0] == self.dimensionality, (points.shape[0], self.dimensionality)

        S = points.shape[1]

        if out is None:
            return torch.ones(1, 1, self.dim, S, device=points.device, dtype=points.dtype)
        else:
            assert out.shape == (1, 1, self.dim, S)
            out[:] = 1.
            return out

    def steerable_attrs_iter(self):
        yield self.steerable_attrs(0)

    def steerable_attrs(self, idx):
        assert idx == 0, idx
        attr = {}
        attr["idx"] = 0
        attr["j"] = self.group.trivial_representation.id
        attr["shape"] = (1, 1)
        return attr

    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        if j != self.group.trivial_representation.id:
            return
        yield self.steerable_attrs(0)

    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        if j != self.group.trivial_representation.id:
            return
        return self.steerable_attrs(idx)

    def __getitem__(self, idx):
        return self.steerable_attrs(idx)

    def __iter__(self):
        yield self.steerable_attrs(0)

    def __eq__(self, other):
        if isinstance(other, PointBasis):
            return self.group == other.group
        else:
            return False

    def __hash__(self):
        return hash(self.group.__class__) * 1000 + sum(hash(x) for x in self.group._keys.items())

