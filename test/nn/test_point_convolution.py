import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.gspaces import *

import numpy as np
import math

import torch


class TestConvolution(TestCase):

    def test_cyclic(self):
        N = 8
        g = rot2dOnR2(N)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])

        sigma = None
        fco = None

        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)
        cl.bias.data = 20*torch.randn_like(cl.bias.data)

        for _ in range(3):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

        cl.train()
        for _ in range(3):
            cl.check_equivariance()

        cl.eval()

        for _ in range(3):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_so2(self):
        N = 7
        g = rot2dOnR2(-1, 2*N)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(N)] + [g.fibergroup.bl_regular_representation(N)]
        r1 = r2 = g.type(*reprs)

        sigma = None
        fco = None
        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = flipRot2dOnR2(N, axis=np.pi/3)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
        # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
        # r2 = FieldType(g, [g.regular_repr])

        sigma = None
        fco = None
        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 4
        g = flipRot2dOnR2(-1, 2*N)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(N)] + [g.fibergroup.bl_regular_representation(N)]
        r1 = r2 = g.type(*reprs)

        sigma = None
        fco = None
        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_flip(self):
        # g = flip2dOnR2(axis=np.pi/3)
        g = flip2dOnR2(axis=np.pi/2)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()

        sigma = None
        fco = None
        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        for _ in range(32):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_so3(self):
        g = rot3dOnR3(3)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = r2 = g.type(*reprs)

        sigma = None
        fco = None
        cl = R3PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)
        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(4):
            # init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance()

    def test_octa(self):
        g = octaOnR3()

        r1 = FieldType(g, g.irreps)
        r2 = FieldType(g, g.irreps)

        sigma = None
        fco = None
        cl = R3PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)
        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            # cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance()

    def test_ico(self):
        g = icoOnR3()

        r1 = FieldType(g, g.irreps)
        r2 = FieldType(g, g.irreps)

        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R3PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            # cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance()

    def test_train(self):
        N = 7
        g = flipRot2dOnR2(-1, N)

        r1 = FieldType(g, [g.trivial_repr, g.irrep(1, 1)])
        r2 = FieldType(g, [g.trivial_repr, g.irrep(1, 2)])

        sigma = None
        fco = None
        cl = R2PointConv(r1, r2,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        init.generalized_he_init(cl.weights.data, cl.basissampler)
        cl.eval()

        optimizer = torch.optim.Adam(cl.parameters(), lr=1e-2)
        cl.train()

        B = 20

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cl.to(device)

        for i in range(10):
            optimizer.zero_grad()

            x = torch.randn(B, cl.in_type.size, device=device)
            t = torch.randn(B, cl.out_type.size, device=device)

            pos = torch.randn(B, 2, device=device)
            distance = torch.norm(pos.unsqueeze(1) - pos, dim=2, keepdim=False)
            thr = sorted(distance.view(-1).tolist())[
                int(B ** 2 // 8)
            ]
            edge_index = torch.nonzero(distance < thr).T.contiguous()

            row, cols = edge_index
            edge_delta = pos[row] - pos[cols]

            x = cl.in_type(x, coords=pos)

            y = cl(x, edge_index, edge_delta)

            loss = ((y.tensor - t)**2).mean()
            loss.backward()

            optimizer.step()

            del loss

        cl.eval()
        cl.cpu()
        cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
