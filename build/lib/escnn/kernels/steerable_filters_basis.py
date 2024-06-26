
from .basis import KernelBasis, EmptyBasisException

from escnn.group import Group
from escnn.group import IrreducibleRepresentation
from escnn.group import Representation

import torch

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

        Abstract class for bases implementing a :math:`G`-steerable basis for *scalar* filters over an Euclidean space
        :math:`\R^n`.

        The action of ``G`` on the Euclidean space is given by the :class:`~escnn.group.Representation` ``action``.
        The dimensionality of the Euclidean space is inferred by the size of the ``action``.

        A :math:`G`-steerable basis provides an irreps-decomposition of the action of :math:`G` on the vector space of
        square-integrable functions, i.e. :math:`L^2(\R^n)`.
        The input list ``js`` defines the order and the multiplicity of the ``G``-irreps in this decomposition.
        More precisely, ``js`` is a list of tuples ``(irrep_id, multiplicity)``, where ``irrep_id`` is the ``id`` of
        an :class:`~escnn.group.IrreducibleRepresentation`.

        The order of the basis elements sampled in :meth:`~escnn.kernels.SteerableFiltersBasis.sample` should be
        consistent with the order of the irreps in the list ``js``.

        Since this class only parameterizes scalar filters, ``SteerableFiltersBasis.shape = (1, 1)``.

        Args:
            G (Group): the symmetry group
            action (Representation): the representation of the action of ``G`` on the Euclidean space
            js (list): the multiplicity of each irrep in this basis.

        Attributes:
            ~.G (Group): the symmetry group
            ~.action (Representation): the representation of the action of ``G`` on the Euclidean space
            ~.js (list): the multiplicity of each irrep in this basis.

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
            The dimensionality of the Euclidean space on which the scalar filters are defined.
        """
        return self.action.size

    @abstractmethod
    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.
        Optionally, store the resulting multidimensional array in ``out``.

        ``points`` must be an array of shape `(N, d)`, where `N` is the number of points and `d` is equal to
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
        r"""

        Sample the continuous basis elements on the discrete set of points in ``points``.

        Rather than returning a single tensor containing all sampled basis elements, it groups basis elements by
        the ``G``-irrep acting on them.
        Then, the method returns a dictionary mapping each irrep's ``id`` to a tensor of shape `(N, m, d)`, where
        `m` is the multiplicity of the irrep (as in the list ``js``) and `d` is the size of the irrep.

        ``points`` must be an array of shape `(N, D)`, where `N` is the number of points and `D` is equal to
        :meth:`~escnn.kernels.SteerableFilterBasis.dimensionality`.

        Optionally, store the resulting multidimentional array in ``out``.

        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """

        S = points.shape[0]

        if out is not None:
            assert out.shape == (S, self.dim), (out.shape, self.dim, S)
            out = out.view(S, self.dim, 1, 1)

        out = self.sample(points, out)

        out = out.view(S, self.dim)

        basis = {}
        p = 0
        for j, m in self.js:
            psi = self.group.irrep(*j)
            dim = psi.size * m
            basis[j] = out[:, p : p+dim].view(S, m, psi.size)
            p += dim

        return basis

    def dim_harmonic(self, j: Tuple) -> int:
        r"""
            Number of basis elements associated with the ``G``-irrep with the id ``j``.
            This is equal to the multiplicity of this irrep times its size, i.e. the number of subspaces transforming
            according to this irrep times the dimensionality of these subspaces.
        """
        psi = self.group.irrep(*j)
        if j in self._js:
            return psi.size * self._js[j]
        else:
            return 0

    def multiplicity(self, j: Tuple) -> int:
        r"""
            The multiplicity of the ``G``-irrep with the id ``j`` as defined in the list ``js``.
        """
        if j in self._js:
            return self._js[j]
        else:
            return 0

    def _get_j_from_idx(self, idx: int):
        assert idx < self.dim
        p = 0
        for j, m in self.js:
            dim = self.dim_harmonic(j)
            if p + dim > idx:
                return j, idx - p
            p += dim
        raise ValueError()

    def _get_j_from_idx_steerable(self, idx: int):
        p = 0
        for j, m in self.js:
            if p + m > idx:
                return j, idx - p
            p += m
        raise ValueError()

    # def __getitem__(self, idx):
    #     j, rel_idx = self._get_j_from_idx(idx)
    #     return self.attrs_j(j, rel_idx)
    #
    # def __iter__(self):
    #     for j, _ in self.js:
    #         for attr in self.attrs_j_iter(j):
    #             yield attr

    def steerable_attrs_iter(self) -> Iterable:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        for j, _ in self.js:
            for attr in self.steerable_attrs_j_iter(j):
                yield attr

    def steerable_attrs(self, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis
        j, rel_idx = self._get_j_from_idx_steerable(idx)
        return self.steerable_attrs_j(j, rel_idx)

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
        points = torch.randn(S, self.dimensionality)

        basis = self.sample_as_dict(points)

        for _ in range(10):
            g = self.group.sample()

            points_g = points @ torch.tensor(self.action(g), device=points.device, dtype=points.dtype).T
            basis_g = self.sample_as_dict(points_g)

            g_basis = {}
            for j, basis_j in basis.items():
                rho_g = torch.tensor(self.group.irrep(*j)(g), device=points.device, dtype=points.dtype)
                g_basis[j] = torch.einsum(
                    'ij,pmj->pmi',
                    rho_g,
                    basis_j,
                )

            for j, m in self.js:
                dim = self.group.irrep(*j).size
                assert basis_g[j].shape == (S, m, dim), (basis_g[j].shape, m, dim, S)
                assert g_basis[j].shape == (S, m, dim), (g_basis[j].shape, m, dim, S)
                assert torch.allclose(g_basis[j], basis_g[j], atol=2e-6, rtol=5e-4), (g_basis[j]-basis_g[j]).max().item()


class PointBasis(SteerableFiltersBasis):

    def __init__(self, G: Group):
        r"""
            Basis for scalar functions over a "0-dimensional" Euclidean space, i.e. a space containing only a single
            point.
            This basis only contains the function associating the value :math:`1` to this single point.

            This basis is steerable for any group ``G``, which acts on this point by mapping it to itself.

            This class is useful to parameterize equivairant linear layers as a special case of steerable convolution
            layers in a neural network.

            .. warning::
                This class is implemented by using ``G.trivial_representation`` as ``action`` which, however, implies
                a 1-dimensional Euclidean space.

        """
        super(PointBasis, self).__init__(G, G.trivial_representation, [(G.trivial_representation.id, 1)])

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:

        assert len(points.shape) == 2
        assert points.shape[1] == self.dimensionality, (points.shape, self.dimensionality)

        S = points.shape[0]

        if out is None:
            return torch.ones(S, self.dim, 1, 1, device=points.device, dtype=points.dtype)
        else:
            assert out.shape == (S, self.dim, 1, 1)
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

