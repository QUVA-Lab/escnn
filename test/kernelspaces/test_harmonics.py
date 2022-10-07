import unittest
from unittest import TestCase

from escnn.group import *
from escnn.kernels import HarmonicPolynomialR3Generator
from escnn.kernels.polar_basis import spherical_harmonics

import torch
import numpy as np

import random

np.set_printoptions(precision=3, suppress=True, linewidth=100000, threshold=10000)


class TestHarmonicPolynomialsR3Generator(TestCase):
    
    def test_equivariance(self):
        hp = HarmonicPolynomialR3Generator(L=3)
        hp.check_equivariance()

    def test_compatibility_lielearn(self):
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
