import unittest
from unittest import TestCase

from escnn.kernels import HarmonicPolynomialR3Generator

import torch
import numpy as np


class TestHarmonicPolynomialsR3Generator(TestCase):
    
    def test_equivariance(self):
        hp = HarmonicPolynomialR3Generator(L=5)
        hp.check_equivariance()

    def test_compatibility_lielearn(self):

        try:
            # need to install lie_learn for this test
            from lie_learn.representations.SO3.spherical_harmonics import rsh
        except ImportError:
            print('You need to install lie_learn to run this test')
            raise

        def spherical_harmonics(points: torch.Tensor, L: int):
            r"""
                Compute the spherical harmonics up to frequency ``L``.
            """

            assert len(points.shape) == 2
            assert points.shape[1] == 3

            assert not points.requires_grad

            device = points.device
            dtype = points.dtype

            S = points.shape[0]

            radii = torch.norm(points, dim=1).detach().cpu().numpy()
            x, y, z = points.detach().cpu().numpy().T

            angles = np.empty((S, 2))
            angles[:, 0] = np.arccos(np.clip(z / radii, -1., 1.))
            angles[:, 1] = np.arctan2(y, x)

            Y = np.empty((S, (L + 1) ** 2))
            for l in range(L + 1):
                for m in range(-l, l + 1):
                    Y[:, l ** 2 + m + l] = rsh(l, m, np.pi - angles[:, 0], angles[:, 1])

                # the central column of the Wigner D Matrices is proportional to the corresponding Spherical Harmonic
                # we need to correct by this proportion factor
                Y[:, l ** 2:(l + 1) ** 2] *= np.sqrt(4 * np.pi / (2 * l + 1))
                if l % 2 == 1:
                    Y[:, l ** 2:(l + 1) ** 2] *= -1

            return torch.tensor(Y, device=device, dtype=dtype)

        with torch.no_grad():
            for L in [0, 1, 2, 3, 5]:
                hp = HarmonicPolynomialR3Generator(L=L)

                N = 100
                points = torch.randn(N, 3)
                points = torch.nn.functional.normalize(points, dim=1)

                hp.eval()

                sh_1 = hp(points)
                assert sh_1.shape == (N, (L+1)**2), (points.shape, sh_1.shape, N, L)
                sh_2 = spherical_harmonics(points, L)

                e_1 = torch.norm(sh_1[:, -2*L-1:], dim=-1)
                e_2 = torch.norm(sh_2[:, -2*L-1:], dim=-1)
                self.assertTrue(
                    torch.allclose(e_1, e_2, atol=1e-5, rtol=1e-4),
                    msg=f'{L} | Energy ratio: {(e_2 / e_1).mean().item()} +- {(e_2 / e_1).std().item()}'
                )

                assert torch.allclose(sh_1, sh_2, atol=1e-5, rtol=1e-4), f'{L} | Max error: {(sh_1 - sh_2).abs().max().item()}'
                self.assertTrue(
                    torch.allclose(sh_1, sh_2, atol=1e-5, rtol=1e-4),
                    msg=f'{L} | Max error: {(sh_1 - sh_2).abs().max().item()}'
                )

                del hp, points, sh_1, sh_2


if __name__ == '__main__':
    unittest.main()
