import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.group import *
from escnn.gspaces import *

import numpy as np

import torch


class TestConvolution(TestCase):
    
    def test_cyclic(self):
        N = 8
        g = rot2dOnR2(N)
        
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        s = 7
        sigma = None
        # fco = lambda r: 1. * r * np.pi
        fco = None
        
        cl = R2ConvTransposed(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        cl.bias.data = 20*torch.randn_like(cl.bias.data)

        for _ in range(1):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()
        
        cl.train()
        for _ in range(1):
            cl.check_equivariance()
        
        cl.eval()
        
        for _ in range(5):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            filter = cl.filter.clone()
            cl.check_equivariance()
            self.assertTrue(torch.allclose(filter, cl.filter))

    def test_so2(self):
        N = 7
        g = rot2dOnR2(-1, N)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
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
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 7
        g = flipRot2dOnR2(-1, N)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_flip(self):
        # g = flip2dOnR2(axis=np.pi/3)
        g = flip2dOnR2(axis=np.pi/2)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(32):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o2_3d(self):
        N = 7
        g = conicalOnR3()

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        o2_group(5)

        s = 5
        sigma = None
        fco = None
        cl = R3ConvTransposed(r1, r2, s,
                              sigma=sigma,
                              frequencies_cutoff=fco,
                              bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_so2_3d(self):
        N = 7
        g = rot2dOnR3()

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        so2_group(5)

        s = 7
        sigma = None
        fco = None
        cl = R3ConvTransposed(r1, r2, s,
                              sigma=sigma,
                              frequencies_cutoff=fco,
                              bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_so3(self):
        N = 7
        g = rot3dOnR3()

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        s = 7
        sigma = None
        fco = None
        cl = R3ConvTransposed(r1, r2, s,
                              sigma=sigma,
                              frequencies_cutoff=fco,
                              bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o3(self):
        N = 7
        g = rot3dOnR3()

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        s = 5
        sigma = None
        fco = None
        cl = R3ConvTransposed(r1, r2, s,
                              sigma=sigma,
                              frequencies_cutoff=fco,
                              bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
