
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
                 js: List,
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

        self.sample(points, out)

        basis = {}
        p = 0
        for j, m in self.js:
            psi = self.group.irrep(*j)
            dim = psi.size * m
            basis[j] = out[:, :, p : p+dim, :].view(1, 1, m, psi.size, S)
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

        basis = {
            j: self.sample_j(points, j)[0, 0]
            for j, _ in self.js
        }

        for _ in range(10):
            g = self.group.sample()

            points_g = torch.tensor(self.action(g), device=points.device, dtype=points.dtype) @ points
            basis_g = {
                j: self.sample_j(points_g, j)[0, 0]
                for j, _ in self.js
            }

            g_basis = {
                j: torch.tensor(self.group.irrep(*j)(g), device=points.device, dtype=points.dtype) @ basis_j
                for j, basis_j in basis
            }

            for j, m in self.js:
                assert basis_g[j].shape == (self.dim_harmonic(j), S)
                assert g_basis[j].shape == (self.dim_harmonic(j), S)
                assert torch.allclose(g_basis[j], basis_g[j])
