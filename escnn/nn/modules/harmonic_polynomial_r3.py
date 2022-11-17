
from escnn.gspaces import *
from escnn.group import *
from escnn.nn import GeometricTensor
from .equivariant_module import EquivariantModule

from escnn.kernels import HarmonicPolynomialR3Generator

import torch
import numpy as np

from typing import Tuple, Union


__all__ = ["HarmonicPolynomialR3"]


class HarmonicPolynomialR3(EquivariantModule):

    def __init__(self, L: int, group: Union[Group, str] = 'so3'):
        r"""
            Module which computes the *harmonic polynomials* in :math:`\R^3` up to order `L`.

            The argument `group` can be a string (`"so3"` or `"o3"`) or a group (instance of
            :class:`~escnn.group.SO3` or :class:`~escnn.group.O3` ).

            This equivariant module takes a set of 3-dimensional points transforming according to
            the :meth:`~escnn.group.SO3.standard_representation` of :math:`SO(3)`
            (or the :meth:`~escnn.group.O3.standard_representation` of :math:`O(3)`)
            and outputs :math:`(L+1)^2` dimensional
            feature vectors transforming like spherical harmonics according to
            :meth:`~escnn.group.SO3.bl_sphere_representation` of :math:`SO(3)`
            (or :meth:`~escnn.group.O3.bl_sphere_representation` of :math:`O(3)`)
            with `L=L`.

            .. seealso ::

                Harmonic polynomial are related to the spherical harmonics.
                Check the
                `Wikipedia page <https://en.wikipedia.org/wiki/Spherical_harmonics#Harmonic_polynomial_representation>`_
                about them.

        """

        super(HarmonicPolynomialR3, self).__init__()

        if group == 'so3':
            group = so3_group(L)
        elif group == 'o3':
            group = o3_group(L)

        assert isinstance(group, SO3) or isinstance(group, O3), group

        self.G: Union[SO3, O3] = group

        self.gspace = no_base_space(self.G)

        self.L = L

        self.harmonics_generator = HarmonicPolynomialR3Generator(self.L)
        self.rho = self.G.bl_sphere_representation(L)

        self.in_type = self.gspace.type(self.G.standard_representation())
        self.out_type = self.gspace.type(self.rho)

    def forward(self, points: GeometricTensor):
        assert points.type == self.in_type
        features = self.harmonics_generator(points.tensor)
        return self.out_type(features, coords=points.coords)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2
        assert input_shape[1] == self.in_type.size

        return (input_shape[0], self.out_type.size)

    def check_equivariance(self, atol: float = 1e-4, rtol: float = 1e-3):

        device = self.harmonics_generator.cob_1.device

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
            assert torch.allclose(rot_sh.tensor, sh_rot.tensor, atol=atol, rtol=rtol), (rot_sh - sh_rot).tensor.abs().max().item()

