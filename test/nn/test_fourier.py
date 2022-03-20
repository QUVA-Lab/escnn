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
    
        for F, N in zip(range(1, 5), [7, 18, 36, 60]):
            
            grid = g.fibergroup.sphere_grid(type='thomson', N=N)
            
            print(F, len(grid))
        
            cl = QuotientFourierELU(g, (False, -1), 3, [(l,) for l in range(F + 1)], grid=grid)
            cl.check_equivariance(rtol=1e-1)

    def test_o3_sphere(self):
        g = no_base_space(o3_group(1))

        for F, N in zip(range(1, 5), [7, 18, 36, 60]):

            grid = g.fibergroup.sphere_grid(type='thomson', N=N)
    
            print(F, len(grid))
    
            cl = QuotientFourierELU(g, ('cone', -1), 3, [(k, l) for k in range(2) for l in range(F + 1)], grid=grid)
            cl.check_equivariance(rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
