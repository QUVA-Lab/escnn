import unittest
from unittest import TestCase

import matplotlib.pyplot as plt

import escnn.nn.init as init
from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *

import numpy as np

import torch


class TestFourier(TestCase):
    
    def test_so2(self):
        g = no_base_space(so2_group(10))
        
        for F, N in zip(range(1, 6), [6, 11, 15, 19, 24]):
            grid = {
                'type': 'regular',
                'N': N
            }
            print(F, grid['N'])

            cl = FourierELU(g, 3, [(l,) for l in range(F+1)], **grid)
            cl.check_equivariance(rtol=3e-2)

    def test_o2(self):
        g = no_base_space(o2_group(10))

        for F, N in zip(range(1, 6), [6, 11, 15, 19, 24]):
            grid = {
                'type': 'regular',
                'N': N*2
            }
            print(F, grid['N'])

            irreps = [(0, 0)] + [(1, l) for l in range(F+1)]
            
            cl = FourierELU(g, 3, irreps, **grid)
            cl.check_equivariance()

    def test_so3(self):
        g = no_base_space(so3_group(1))

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(1), type='thomson_cube', N=1)
        cl.check_equivariance(rtol=1e-1)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), type='thomson_cube', N=5)
        cl.check_equivariance(rtol=1e-1)

        for F, N in zip(range(1, 4), [30, 120, 350]):
            d = sum((2*l+1)**2 for l in range(F+1))
            grid = {
                'type': 'thomson',
                'N': N
            }
            print(F, grid['N'])
        
            cl = FourierELU(g, 3, [(l,) for l in range(F + 1)], **grid)
            cl.check_equivariance(rtol=1e-1)

    def test_o3(self):
        g = no_base_space(o3_group(1))

        for F, N in zip(range(1, 4), [30, 120, 350]):
            grid = {
                'type': 'thomson',
                'N': N*2
            }
            print(F, grid['N'])
        
            cl = FourierELU(g, 3, [(k, l) for k in range(2) for l in range(F + 1)], **grid)
            cl.check_equivariance(rtol=1e-1)

    def test_so2_quot_cn(self):
        g = no_base_space(so2_group(10))
        
        Ns = {
            1: 6,
            2: 11,
            3: 15,
            4: 19,
            5: 24
        }

        for n in range(1, 6):
            F = 5
            f = int(F / n)
            
            N = Ns[f]
            grid = [
                g.fibergroup.element(i*2*np.pi/(n*N), 'radians')
                for i in range(N)
            ]
            
            print(n, F, len(grid))
        
            cl = QuotientFourierELU(g, n, 3, [(l,) for l in range(F + 1)], grid=grid)
            cl.check_equivariance(rtol=3e-2)

    def test_so3_sphere(self):
        g = no_base_space(so3_group(1))

        grid = g.fibergroup.sphere_grid(type='ico')
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), grid=grid)
        cl.check_equivariance(rtol=1e-1)

        grid = g.fibergroup.sphere_grid(type='thomson_cube', N=1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), grid=grid)
        cl.check_equivariance(rtol=1e-1)

        for F, N in zip(range(1, 5), [8, 17, 33, 52]):

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)

            print(F, len(grid))
            cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance(rtol=1e-1)

    def test_o3_sphere(self):
        g = no_base_space(o3_group(1))

        for F, N in zip(range(1, 5), [8, 17, 33, 52]):

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)

            print(F, len(grid))

            cl = QuotientFourierELU(g, ('cone', -1), 3, g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance(rtol=1e-1)

    def test_so3_octa(self):
        g = no_base_space(so3_group(1))

        atol=1e-5
        rtol=1e-6
        for n, F in zip([1, 3, 7], [1, 2, 3]):
            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), type='thomson_cube', N=n)
            # cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), type='cube')
            # cl = QuotientFourierELU(g, (False,-1), 3, g.fibergroup.bl_irreps(F), grid=g.fibergroup.sphere_grid(type='thomson_cube', N=1))

            with torch.no_grad():
                x = torch.randn(30, cl.in_type.size)
                x = cl.in_type(x)

                for i, el in enumerate(g.fibergroup.grid('cube')):
                    # print(i)

                    out1 = cl(x).transform(el).tensor.detach().numpy()
                    out2 = cl(x.transform(el)).tensor.detach().numpy()

                    errs = np.abs(out1 - out2)

                    esum = np.maximum(np.abs(out1), np.abs(out2))
                    esum[esum <1e-7] = 1

                    tol = rtol * esum + atol

                    self.assertTrue(
                        np.all(errs < tol),
                        'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'
                        .format(el, errs.max(), errs.mean(), errs.var())
                    )

    def test_so3_cube_conv(self):
        g = rot3dOnR3()

        # cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), type='thomson_cube', N=1)
        # cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), type='cube', N=1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2),
                                grid=g.fibergroup.sphere_grid(type='thomson_cube', N=1))
        # cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), type='thomson_cube', N=1)

        t = g.type(g.trivial_repr)

        conv1 = R3PointConv(t, cl.in_type, bias=False, width=2, n_rings=3)
        conv2 = R3PointConv(cl.out_type, t, bias=False, width=2, n_rings=3)

        P = 30
        atol = 1e-4
        rtol = 1e-5

        for _ in range(8):

            conv1.weights.data.normal_()
            # init.generalized_he_init(conv1.weights.data, conv1.basissampler)
            conv2.weights.data.normal_()
            # init.generalized_he_init(conv2.weights.data, conv2.basissampler)
            conv1.eval()
            conv2.eval()

            pos = torch.randn(P, 3)
            x = torch.randn(P, t.size)
            x = GeometricTensor(x, t, pos)

            distance = torch.norm(pos.unsqueeze(1) - pos, dim=2, keepdim=False)
            thr = sorted(distance.view(-1).tolist())[int(P ** 2 // 16)]
            edge_index = torch.nonzero(distance < thr).T.contiguous()

            for i, el in enumerate(g.fibergroup.grid('cube')):

                out1 = (conv2(cl(conv1(x, edge_index)), edge_index)).transform(el).tensor.detach().numpy()
                out2 = (conv2(cl(conv1(x.transform(el), edge_index)), edge_index)).tensor.detach().numpy()

                out1 = out1.mean(0)
                out2 = out2.mean(0)

                errs = np.abs(out1 - out2)

                esum = np.maximum(np.abs(out1), np.abs(out2))
                esum[esum <1e-7] = 1

                tol = rtol * esum + atol

                self.assertTrue(
                    np.all(errs < tol),
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'
                    .format(el, errs.max(), errs.mean(), errs.var())
                )


    ########################################################################################
    # test with different input/output frequencies
    ########################################################################################

    def test_so2_different_out(self):
        g = no_base_space(so2_group(10))

        for F, N in zip(range(1, 5), [6, 11, 15, 19]):
            grid = {
                'type': 'regular',
                'N': N
            }

            print(F, N)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), **grid)
            cl.check_equivariance()

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F-2)), **grid)
            cl.check_equivariance()

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(max(0, F-2)), out_irreps=g.fibergroup.bl_irreps(F), **grid)
            cl.check_equivariance()

    def test_o2_different_out(self):
        g = no_base_space(o2_group(10))

        for F, N in zip(range(1, 5), [6, 11, 15, 19]):
            grid = {
                'type': 'regular',
                'N': N * 2
            }
            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), **grid)
            cl.check_equivariance()

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), **grid)
            cl.check_equivariance()

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), **grid)
            cl.check_equivariance()

    def test_so3_different_out(self):
        g = no_base_space(so3_group(1))

        grid = g.fibergroup.sphere_grid(type='ico')
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=5e-2)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), grid=grid)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=5e-2)

        grid = g.fibergroup.sphere_grid(type='thomson_cube', N=1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=5e-2)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), grid=grid)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=5e-2)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(1), out_irreps=g.fibergroup.bl_irreps(0), type='thomson_cube', N=1)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=6e-2)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(0), type='thomson_cube', N=3)
        # cl.check_equivariance(rtol=1e-1)
        cl.check_equivariance(rtol=7e-2)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), type='thomson_cube', N=4)
        cl.check_equivariance(rtol=1e-1)

        for F, N in zip(range(1, 3), [24, 120]):
            grid = {
                'type': 'thomson',
                'N': N
            }
            print(F, N)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(max(0, F - 1)), out_irreps=g.fibergroup.bl_irreps(F), **grid)
            # cl.check_equivariance(rtol=1e-1)
            cl.check_equivariance(rtol=5e-2)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), **grid)
            cl.check_equivariance(rtol=1e-1)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 1)), **grid)
            cl.check_equivariance(rtol=1e-1)

    def test_o3_different_out(self):
        g = no_base_space(o3_group(1))

        for F, N in zip(range(1, 3), [30, 120]):
            grid = {
                'type': 'thomson',
                'N': N * 2
            }

            print(F, len(grid))

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), **grid)
            cl.check_equivariance(rtol=1e-1)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), **grid)
            cl.check_equivariance(rtol=1e-1)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), **grid)
            cl.check_equivariance(rtol=1e-1)

    def test_so2_quot_cn_different_out(self):
        g = no_base_space(so2_group(10))

        Ns = {
            1: 6,
            2: 11,
            3: 15,
            4: 19,
            5: 24
        }

        for n in range(1, 6):
            F = 5
            f = int(F / n)

            N = Ns[f]
            grid = [
                g.fibergroup.element(i * 2 * np.pi / (n * N), 'radians')
                for i in range(N)
            ]

            cl = QuotientFourierELU(g, n, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
            cl.check_equivariance(rtol=3e-2)

            cl = QuotientFourierELU(g, n, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), grid=grid)
            cl.check_equivariance(rtol=3e-2)

            cl = QuotientFourierELU(g, n, 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance(rtol=3e-2)

    def test_so3_sphere_different_out(self):
        g = no_base_space(so3_group(1))

        for F, N in zip(range(1, 5), [8, 17, 33, 52]):
            grid = g.fibergroup.sphere_grid(type='thomson', N=N)

            print(F, len(grid))

            cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
            cl.check_equivariance(rtol=1e-1, atol=1e-2)

            cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), grid=grid)
            cl.check_equivariance(rtol=1e-1)

            cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance(rtol=1e-1)

    def test_o3_sphere_different_out(self):
        g = no_base_space(o3_group(1))

        for F, N in zip(range(1, 5), [8, 17, 33, 52]):
            grid = g.fibergroup.sphere_grid(type='thomson', N=N)

            print(F, len(grid))

            cl = QuotientFourierELU(g, ('cone', -1), 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
            # cl.check_equivariance(rtol=1e-1, atol=1e-2)
            cl.check_equivariance(rtol=5e-2, atol=1e-2)

            cl = QuotientFourierELU(g, ('cone', -1), 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), grid=grid)
            cl.check_equivariance(rtol=1e-1)

            cl = QuotientFourierELU(g, ('cone', -1), 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance(rtol=1e-1)

    ########################################################################################

    def test_with_spatial_dimns(self):
        g = rot2dOnR2(-1)
    
        for F, N in zip(range(1, 6), [6, 11, 15, 19, 24]):
            grid = {
                'type': 'regular',
                'N': N
            }
            print(F, grid['N'])
        
            cl = FourierELU(g, 3, [(l,) for l in range(F + 1)], **grid)
            cl.check_equivariance(atol=1e-1)
            
        g = rot3dOnR3()

        for F, N in zip(range(1, 4), [30, 120, 350]):
            d = sum((2 * l + 1) ** 2 for l in range(F + 1))
            grid = {
                'type': 'thomson',
                'N': N
            }
            print(F, grid['N'])
    
            cl = FourierELU(g, 3, [(l,) for l in range(F + 1)], **grid)
            cl.check_equivariance(rtol=1e-1)

    def test_norm_preserved_relu(self):

        # Check if the norm of the features is approximately preserved

        # To do so, we try to apply ReLU over strictly positive features
        # or apply ReLU over roughly centered features (expecting the norm to reduce by approx sqrt(2))

        # for these tests to work, we should not apply normalization of the Fourier matrices, otherwise the scale
        # of the output features depends on the number of Fourier components recovered

        g = no_base_space(so3_group(1))

        for F, N in zip(range(1, 4), [24, 120, 300]):
            print(F, N)

            cl = FourierPointwise(g, 1, g.fibergroup.bl_irreps(F), normalize=False, function='p_relu', type='thomson', N=N)

            c = cl.in_type.size
            B = 256

            # make sure the average value of the feature is very high so the ReLU acts as an identity
            # then, check that the norm of the input and output features are similar
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 150.
                p += irr.size

            in_norms = torch.linalg.norm(x, dim=2).reshape(-1).cpu().detach().numpy()
            x = x.view(B, cl.in_type.size)
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            y = y.view(B, len(cl.out_type), cl.rho_out.size)
            out_norms = torch.linalg.norm(y, dim=2).reshape(-1).cpu().detach().numpy()
            self.assertTrue(np.allclose(in_norms, out_norms))

            # now, make sure features are centered around zero such that, on average, half of the entries are set to 0
            # by ReLU. We expect the output to have a norm roughly sqrt(2) smaller than the input
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 0.
                p += irr.size

            in_norms = torch.linalg.norm(x, dim=2).reshape(-1).cpu().detach().numpy()
            x = x.view(B, cl.in_type.size)
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            y = y.view(B, len(cl.out_type), cl.rho_out.size)
            out_norms = torch.linalg.norm(y, dim=2).reshape(-1).cpu().detach().numpy()
            norms_ratio = out_norms / in_norms
            print(norms_ratio.max(), norms_ratio.min(), norms_ratio.mean(), norms_ratio.std())
            self.assertTrue(np.allclose(norms_ratio, 1./np.sqrt(2), atol=1e-1, rtol=1e-1), msg=f"{np.fabs(norms_ratio-1./np.sqrt(2)).max()}")

        for F, N in zip(range(1, 4), [16, 34, 66]):
            print(F, N)

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)
            cl = QuotientFourierPointwise(g, (False, -1), 40, g.fibergroup.bl_irreps(F), normalize=False, function='p_relu', grid=grid)

            c = cl.in_type.size
            B = 32

            # make sure the average value of the feature is very high so the ReLU acts as an identity
            # then, check that the norm of the input and output features are similar
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 150.
                p += irr.size
            x = x.view(B, cl.in_type.size)

            in_norms = torch.linalg.norm(x, dim=1).reshape(-1).cpu().detach().numpy()
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            out_norms = torch.linalg.norm(y, dim=1).reshape(-1).cpu().detach().numpy()
            self.assertTrue(np.allclose(in_norms, out_norms))

            # now, make sure features are centered around zero such that, on average, half of the entries are set to 0
            # by ReLU. We expect the output to have a norm roughly sqrt(2) smaller than the input
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 0.
                p += irr.size

            x = x.view(B, cl.in_type.size)
            in_norms = torch.linalg.norm(x, dim=1).reshape(-1).cpu().detach().numpy()
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            out_norms = torch.linalg.norm(y, dim=1).reshape(-1).cpu().detach().numpy()
            norms_ratio = out_norms / in_norms
            # print(norms_ratio.max(), norms_ratio.min(), norms_ratio.mean(), norms_ratio.std())
            self.assertTrue(np.allclose(norms_ratio, 1. / np.sqrt(2), atol=1e-1, rtol=1e-1), msg=f"{np.fabs(norms_ratio-1./np.sqrt(2)).max()}")

        max_f = 5
        for F, N in zip(range(1, max_f+1), [200]*max_f):
            print(F, N)

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)
            cl = QuotientFourierPointwise(g, (False, -1), 30, g.fibergroup.bl_irreps(1), out_irreps=g.fibergroup.bl_irreps(F), normalize=False, function='p_relu', grid=grid)

            c = cl.in_type.size
            B = 32

            # make sure the average value of the feature is very high so the ReLU acts as an identity
            # then, check that the norm of the input and output features are similar
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 150.
                p += irr.size
            x = x.view(B, cl.in_type.size)

            in_norms = torch.linalg.norm(x, dim=1).reshape(-1).cpu().detach().numpy()
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            out_norms = torch.linalg.norm(y, dim=1).reshape(-1).cpu().detach().numpy()
            self.assertTrue(np.allclose(in_norms, out_norms))

            # now, make sure features are centered around zero such that, on average, half of the entries are set to 0
            # by ReLU. We expect the output to have a norm roughly sqrt(2) smaller than the input
            x = torch.randn(B, c)
            x = x.view(B, len(cl.in_type), cl.rho.size)
            p = 0
            for irr in cl.rho.irreps:
                irr = g.irrep(*irr)
                if irr.is_trivial():
                    x[:, :, p] = 0.
                p += irr.size

            x = x.view(B, cl.in_type.size)
            in_norms = torch.linalg.norm(x, dim=1).reshape(-1).cpu().detach().numpy()
            x = cl.in_type(x).transform_fibers(g.fibergroup.sample())
            y = cl(x).tensor
            out_norms = torch.linalg.norm(y, dim=1).reshape(-1).cpu().detach().numpy()
            norms_ratio = out_norms / in_norms
            self.assertTrue(np.allclose(norms_ratio, 1. / np.sqrt(2), atol=1e-1, rtol=1e-1), msg=f"{np.fabs(norms_ratio-1./np.sqrt(2)).max()}")

    def plot_errors_so3(self):
        g = no_base_space(so3_group(1))

        Fs = list(range(1, 4))
        Ns = list(range(20, 230, 10))

        errors_mean = np.zeros((len(Fs), len(Ns)))
        errors_std = np.zeros((len(Fs), len(Ns)))

        for i, F in enumerate(Fs):
            print(F)
            for j, N in enumerate(Ns):
                cl = FourierPointwise(g, 1, g.fibergroup.bl_irreps(F), normalize=False, function='p_relu', type='thomson', N=N)
                errs = cl.check_equivariance(rtol=1e-1, assert_raise=False)

                errors_mean[i, j] = errs.mean()
                errors_std[i, j] = errs.std()

        fig, ax = plt.subplots()
        for i, F in enumerate(Fs):
            plt.plot(Ns, errors_mean[i, :], label=f'{F}')
            plt.fill_between(Ns, errors_mean[i, :] - errors_std[i, :], errors_mean[i, :] + errors_std[i, :], alpha=0.3)

        plt.ylim([0., 0.3])
        plt.legend()
        plt.show()

    def plot_errors_sphere(self):
        g = no_base_space(so3_group(1))

        Fs = list(range(1, 5))
        Ns = list(range(8, 88, 8))

        errors_mean = np.zeros((len(Fs), len(Ns)))
        errors_std = np.zeros((len(Fs), len(Ns)))

        for i, F in enumerate(Fs):
            print(F)
            for j, N in enumerate(Ns):
                cl = QuotientFourierPointwise(g, (False, -1), 1, g.fibergroup.bl_irreps(F), normalize=False, function='p_relu', grid=g.fibergroup.sphere_grid(type='thomson', N=N))
                errs = cl.check_equivariance(rtol=1e-1, assert_raise=False)

                errors_mean[i, j] = errs.mean()
                errors_std[i, j] = errs.std()

        fig, ax = plt.subplots()
        for i, F in enumerate(Fs):
            plt.plot(Ns, errors_mean[i, :], label=f'{F}')
            plt.fill_between(Ns, errors_mean[i, :] - errors_std[i, :], errors_mean[i, :] + errors_std[i, :], alpha=0.3)

        plt.ylim([0., 0.2])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
