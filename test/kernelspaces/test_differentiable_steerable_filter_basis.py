import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *
import torch


class TestSolutionsEquivariance(TestCase):
    
    def test_circular(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = CircularShellsBasis(
            L = 5, radial=radial,
        )
        self._check(basis)

    def test_spherical(self):

        radial = GaussianRadialProfile(

            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = SphericalShellsBasis(
            L=5, radial=radial,
        )
        self._check(basis)

    def test_circular_filter(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = CircularShellsBasis(
            L = 5, radial=radial,
            filter=lambda attr: attr['irrep:frequency'] < 2*attr['radius']
        )
        self._check(basis)

    def test_spherical_filter(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = SphericalShellsBasis(
            L=5, radial=radial,
            filter=lambda attr: attr['irrep:frequency'] < 2 * attr['radius']
        )
        self._check(basis)

    def _check(self, basis: SteerableFiltersBasis):
        if basis is None:
            print("Empty KernelBasis!")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        B = 100

        points = 3 * torch.randn(B, basis.dimensionality).to(device=device)
        points[0, :] = 0.

        basis = basis.to(device)

        points.requires_grad_(True)

        from torch.optim import Adam

        optimizer = Adam([points], lr=1e-2)

        for i in range(10):
            optimizer.zero_grad()

            assert not torch.isnan(points).any(), i

            filters = basis.sample(points)

            loss = (filters ** 2 - .3 * filters + filters.abs()).mean()

            loss.backward()

            assert not torch.isnan(points.grad).any(), i
            # grad = torch.norm(points.grad, dim=1)
            # assert (grad > 0.).all(), (i, grad.reshape(-1).shape[0], (grad > 0.).sum().item(), grad.mean().item(), grad.std().item(), grad.max().item(), grad.min().item())
            # assert (grad[0] > 0.).all(), (i, grad[0].item())

            optimizer.step()


if __name__ == '__main__':
    unittest.main()
