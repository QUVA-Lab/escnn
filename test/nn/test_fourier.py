import unittest
from unittest import TestCase

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
            cl.check_equivariance()

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
            cl.check_equivariance()

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
        cl.check_equivariance(rtol=1e-1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), grid=grid)
        cl.check_equivariance(rtol=1e-1)

        grid = g.fibergroup.sphere_grid(type='thomson_cube', N=1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(0), grid=grid)
        cl.check_equivariance(rtol=1e-1)
        cl = QuotientFourierELU(g, (False, -1), 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), grid=grid)
        cl.check_equivariance(rtol=1e-1)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(1), out_irreps=g.fibergroup.bl_irreps(0), type='thomson_cube', N=1)
        cl.check_equivariance(rtol=1e-1)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(0), type='thomson_cube', N=3)
        cl.check_equivariance(rtol=1e-1)

        cl = FourierELU(g, 3, g.fibergroup.bl_irreps(2), out_irreps=g.fibergroup.bl_irreps(1), type='thomson_cube', N=3)
        cl.check_equivariance(rtol=1e-1)

        for F, N in zip(range(1, 3), [24, 120]):
            grid = {
                'type': 'thomson',
                'N': N
            }
            print(F, N)

            cl = FourierELU(g, 3, g.fibergroup.bl_irreps(max(0, F - 1)), out_irreps=g.fibergroup.bl_irreps(F), **grid)
            cl.check_equivariance(rtol=1e-1)

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
            cl.check_equivariance()

            cl = QuotientFourierELU(g, n, 3, g.fibergroup.bl_irreps(F), out_irreps=g.fibergroup.bl_irreps(max(0, F - 2)), grid=grid)
            cl.check_equivariance()

            cl = QuotientFourierELU(g, n, 3, g.fibergroup.bl_irreps(max(0, F - 2)), out_irreps=g.fibergroup.bl_irreps(F), grid=grid)
            cl.check_equivariance()

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
            cl.check_equivariance(rtol=1e-1, atol=1e-2)

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

        g = no_base_space(so3_group(1))

        for F, N in zip(range(1, 4), [24, 120, 300]):

            cl = FourierPointwise(g, 1, g.fibergroup.bl_irreps(F), function='p_relu', type='thomson', N=N)

            c = cl.in_type.size
            B = 128

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
            norms_ratio = in_norms / out_norms
            self.assertTrue(np.allclose(norms_ratio, np.sqrt(2), atol=3e-1, rtol=1e-1), msg=f"{np.fabs(norms_ratio-np.sqrt(2)).max()}")

        for F, N in zip(range(1, 4), [8, 17, 33]):

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)
            cl = QuotientFourierPointwise(g, (False, -1), 1, g.fibergroup.bl_irreps(F), function='p_relu', grid=grid)

            c = cl.in_type.size
            B = 128

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
            norms_ratio = in_norms / out_norms
            self.assertTrue(np.allclose(norms_ratio, np.sqrt(2), atol=4e-1, rtol=1e-1), msg=f"{np.fabs(norms_ratio-np.sqrt(2)).max()}")


if __name__ == '__main__':
    unittest.main()
