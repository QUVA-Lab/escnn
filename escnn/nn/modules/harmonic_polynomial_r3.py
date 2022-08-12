
from escnn.gspaces import *
from escnn.group import *
from escnn.nn import GeometricTensor
from .equivariant_module import EquivariantModule

import torch
import numpy as np

from typing import Tuple


__all__ = ["HarmonicPolynomialR3"]



class HarmonicPolynomialR3(EquivariantModule):

    def __init__(self, L: int):
        r"""
            Module which computes the *harmonic polynomials* in :math:`\R^3` up to order `L`.

            This equivariant module takes a set of 3-dimensional points transforming according to the
            :meth:`~escnn.group.SO3.standard_representation` of :math:`SO(3)` and outputs :math:`(L+1)^2` dimensional
            feature vectors transforming like spherical harmonics according to
            :meth:`~escnn.group.SO3.bl_sphere_representation` with `L=L`.

            .. seealso ::

                Harmonic polynomial are related to the spherical harmonics.
                Check the
                `Wikipedia page <https://en.wikipedia.org/wiki/Spherical_harmonics#Harmonic_polynomial_representation>`_
                about them.

        """

        super(HarmonicPolynomialR3, self).__init__()

        self.G: SO3 = so3_group(L)

        self.gspace = no_base_space(self.G)

        self.L = L
        self.rho = self.G.bl_sphere_representation(L)
        self.d = self.rho.size

        self.in_type = self.gspace.type(self.G.standard_representation())
        self.out_type = self.gspace.type(self.rho)

        self.register_buffer(f'cob_1', torch.tensor(self.G.standard_representation().change_of_basis_inv, dtype=torch.float))

        for l in range(2, self.L+1):
            rho_l = self.G.irrep(l-1).tensor(self.G.irrep(1))
            d = 2*l+1
            cob = rho_l.change_of_basis_inv[-d:, :]

            p = cob @ cob.T
            assert np.allclose(p, np.eye(2*l+1), atol=1e-5, rtol=1e-5), l

            # to ensure normalization

            # this guarantees the column is normalized, i.e. corresponds to the central column of the Wigner D matrix
            cob *= np.sqrt((2*l-1)/l)

            self.register_buffer(f'cob_{l}', torch.tensor(cob, dtype=torch.float))

        sign_mask = torch.zeros(self.d, 2, dtype=torch.float)
        proj = torch.zeros(self.L+1, self.d, dtype=torch.float)
        for l in range(self.L+1):
            proj[l, l**2:(l+1)**2] = 1.

            if l % 2 == 0:
                sign_mask[l ** 2:(l + 1) ** 2, 0] = 1.
            else:
                sign_mask[l ** 2:(l + 1) ** 2, 1] = -1.

    def forward(self, points: GeometricTensor):
        assert points.type == self.in_type

        points = points.tensor

        feature_1 = points @ getattr(self, 'cob_1').T

        features = [
            torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype),
            feature_1,
        ]

        for l in range(2, self.L+1):
            cob = getattr(self, f'cob_{l}')
            f = torch.einsum('bi,bj->bij', features[-1], feature_1).reshape(points.shape[0], -1)
            assert f.shape[1] == 3*(2*l-1), (f.shape, l)
            f = f @ cob.T
            assert f.shape[1] == 2*l+1, (f.shape, l)
            features.append(f)

        features = torch.cat(features, dim=1)

        return self.out_type(features)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2
        assert input_shape[1] == self.in_type.size

        return (input_shape[0], self.out_type.size)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-3):

        device = self.cob_1.device

        N = 40
        points = torch.randn(N, 3, device=device)
        # radii = torch.norm(points, dim=-1).view(-1, 1)
        # points = points / radii
        # points[radii.view(-1) < 1e-3, :] = 0.

        points = self.in_type(points)

        sh = self(points)
        for _ in range(10):
            g = self.G.sample()
            sh_rot = self(g @ points)
            rot_sh = g @ sh
            assert torch.allclose(rot_sh.tensor, sh_rot.tensor, atol=atol, rtol=rtol), (rot_sh - sh_rot).abs().max().item()

