import unittest
from unittest import TestCase

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *
from escnn.nn.modules.pointconv.rd_point_convolution import _RdPointConv

import torch


class TestMixedPrecision(TestCase):

    def test_r3conv_non_uniform(self):
        gs = flipRot3dOnR3(maximum_frequency=4)

        G: O3 = gs.fibergroup
        G.bl_sphere_representation(3)
        G.bl_sphere_representation(2)
        G.bl_regular_representation(3)
        G.bl_regular_representation(2)
        G.bl_regular_representation(1)
        G.bl_regular_representation(0)

        channels = 3
        in_type = out_type = gs.type(
            *list(gs.representations.values()) * channels
        )

        m = R3Conv(in_type, out_type, kernel_size=3, padding=1)

        self.check_amp(m)

    def test_r2conv_non_uniform(self):
        gs = flipRot2dOnR2(N=-1, maximum_frequency=4)

        channels = 12
        L = 4

        # in_type = gs.type(gs.trivial_repr)
        # out_type = gs.type(
        in_type = out_type = gs.type(
            # *[gs.irreps[0] for _ in range(L + 1)] * channels,
            *[gs.irreps[i] for i in range(L + 1)] * channels,
            # *list(gs.representations.values()) * channels
            gs.fibergroup.bl_regular_representation(L)
        )

        m = R2Conv(in_type, out_type, kernel_size=5, padding=1)

        self.check_amp(m)

    def test_r2conv_uniform(self):
        gs = flipRot2dOnR2(N=-1)

        channels = 12
        L = 4

        in_type = gs.type(gs.trivial_repr)
        out_type = gs.type(
            *[gs.irreps[0] for _ in range(L + 1)] * channels,
            *[directsum([gs.irreps[i] for i in range(L + 1)])] * channels
        )

        m = R2Conv(in_type, out_type, kernel_size=5, padding=1)

        self.check_amp(m)

    def test_r3pointconv_non_uniform(self):
        gs = flipRot3dOnR3(maximum_frequency=4)

        G: O3 = gs.fibergroup
        G.bl_sphere_representation(3)
        G.bl_sphere_representation(2)
        G.bl_regular_representation(3)
        G.bl_regular_representation(2)
        G.bl_regular_representation(1)
        G.bl_regular_representation(0)

        channels = 3
        in_type = out_type = gs.type(
            *list(gs.representations.values()) * channels
        )

        m = R3PointConv(in_type, out_type, width=2., n_rings=3, frequencies_cutoff=3.)

        self.check_amp_pointconv(m)

    def test_fourier(self):
        gs = no_base_space(so2_group(10))

        F = 2
        N = 11
        grid = {
            'type': 'regular',
            'N': N
        }
        m = FourierELU(gs, 3, gs.fibergroup.bl_irreps(F), out_irreps=gs.fibergroup.bl_irreps(0), **grid)
        self.check_amp(m)

    def test_fieldnorm(self):
        gs = flipRot2dOnR2(N=-1)

        channels = 12
        L = 4

        in_type = gs.type(
            *[gs.irreps[0] for _ in range(L + 1)] * channels,
            *[directsum([gs.irreps[i] for i in range(L + 1)])] * channels
        )

        m = FieldNorm(in_type)
        m.train()

        self.check_amp(m)

        m.eval()

        self.check_amp(m)

    def test_gnorm(self):
        gs = flipRot2dOnR2(N=-1)

        channels = 12
        L = 4

        in_type = gs.type(
            *[gs.irreps[0] for _ in range(L + 1)] * channels,
            *[directsum([gs.irreps[i] for i in range(L + 1)])] * channels
        )

        m = GNormBatchNorm(in_type)
        m.train()

        self.check_amp(m)

        m.eval()

        self.check_amp(m)

    def check_amp(self, m: EquivariantModule):

        device = "cuda"

        intype = m.in_type

        m = m.to(device)
        x = intype(torch.randn(10, intype.size, *[5]*intype.gspace.dimensionality)).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            m(x)

    def check_amp_pointconv(self, m: _RdPointConv):

        device = "cuda"

        intype = m.in_type

        B = 30

        coords = torch.randn(B, intype.gspace.dimensionality, device=device) * 3.
        edge_index = torch.randint(B, (2, 3*B), dtype=torch.long, device=device)

        m = m.to(device)
        x = intype(torch.randn(B, intype.size), coords=coords).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            m(x, edge_index=edge_index)


if __name__ == '__main__':
    unittest.main()
