import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *

import numpy as np

import torch


class TestConvolution(TestCase):
    
    def test_cyclic(self):
        N = 8
        g = no_base_space(cyclic_group(N))
        
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        cl = Linear(r1, r2, bias=True)
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
            matrix = cl.matrix.clone()
            cl.check_equivariance()
            self.assertTrue(torch.allclose(matrix, cl.matrix))

    def test_so2(self):
        N = 7
        g = no_base_space(so2_group(N))

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = g.type(*reprs)
        r2 = g.type(*reprs)

        cl = Linear(r1, r2, bias=True)
        
        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = no_base_space(dihedral_group(N))

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
        # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
        # r2 = FieldType(g, [g.regular_repr])
    
        cl = Linear(r1, r2, bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 7
        g = no_base_space(o2_group(N))

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(4)]
        r1 = g.type(*reprs)
        r2 = g.type(*reprs)

        cl = Linear(r1, r2, bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_so3(self):
        g = no_base_space(so3_group(1))

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = g.type(*reprs)
        r2 = g.type(*reprs)

        cl = Linear(r1, r2, bias=True)
        
        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            # cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
