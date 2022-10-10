import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *

import torch


class TestWEbasis(TestCase):
    
    def test_spherical_shells(self):
        G = o3_group(4)
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        irreps = [G.irrep(f, l) for f in range(2) for l in range(4)]
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep)
                
    def test_circular_shell(self):
        G = o2_group(8)
        
        axes = [0., np.pi/2, np.pi/3, np.pi/4] + (np.random.rand(5)*np.pi).tolist()
        for axis in axes:
            X = CircularShellsBasis(
                L=4,
                radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
                axis=axis
            )

            irreps = [G.irrep(0, 0)] + [G.irrep(1, l) for l in range(4)]
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep)

    def _check_irreps(self, X: SteerableFiltersBasis, in_rep: IrreducibleRepresentation,
                      out_rep: IrreducibleRepresentation):

        G = X.group

        try:
            basis = WignerEckartBasis(X, in_rep, out_rep)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        P = 20
        points = torch.randn(P, X.dimensionality, device=device)
        assert points.shape == (P, X.dimensionality)

        basis = basis.to(device)

        points.requires_grad_(True)

        from torch.optim import Adam

        optimizer = Adam([points], lr=1e-2)

        for i in range(10):
            optimizer.zero_grad()

            assert not torch.isnan(points).any(), i

            filters = basis.sample(points)

            loss = (filters ** 2 - filters + filters.abs()).mean()

            loss.backward()

            assert not torch.isnan(points.grad).any(), i
            grad = points.grad.abs()
            assert (grad > 0.).all(), (in_rep, out_rep, i, (grad > 0.).sum().item(), grad.mean().item(), grad.std().item(), grad.max().item(), grad.min().item())

            optimizer.step()

        for i in range(10):
            optimizer.zero_grad()

            assert not torch.isnan(points).any(), i

            harmonics = basis.basis.sample_as_dict(points)
            out = {
                j: torch.zeros(
                    (harmonics[j].shape[0], basis.dim_harmonic(j), basis.shape[0], basis.shape[1]),
                    device=harmonics[j].device, dtype=harmonics[j].dtype
                )
                for j in basis.js
            }

            filters = basis.sample_harmonics(harmonics, out)

            loss = sum(
                (filter ** 2 - filter + filter.abs()).mean()
                for filter in filters.values()
            )

            loss.backward()

            assert not torch.isnan(points.grad).any(), i
            grad = points.grad.abs()
            assert (grad > 0.).all(), (i, (grad > 0.).sum().item(), grad.mean().item(), grad.std().item(), grad.max().item(), grad.min().item())

            optimizer.step()


if __name__ == '__main__':
    unittest.main()
