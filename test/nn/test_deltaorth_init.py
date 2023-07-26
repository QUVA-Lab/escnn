import unittest
from unittest import TestCase

import numpy as np
from escnn.nn import *
from escnn.nn import init
from escnn.gspaces import *
from escnn.group import *

import torch

from random import shuffle


class TestDeltaOrth(TestCase):
    
    def test_one_block(self):
        # gspace = flipRot2dOnR2(6)
        gspace = rot2dOnR2(8)
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(5)], name="irrepssum")
        # t1 = FieldType(gspace, [gspace.regular_repr]*1)
        # t2 = FieldType(gspace, [gspace.regular_repr]*1)
        t1 = FieldType(gspace, [irreps])
        t2 = FieldType(gspace, [irreps])
        self.check(t1, t2)
        
    def test_many_block_discontinuous(self):
        gspace = rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 7)
        t2 = FieldType(gspace, list(gspace.representations.values()) * 7)
        self.check(t1, t2)
        
    def test_many_block_sorted(self):
        gspace = rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        self.check(t1, t2)

    def test_3d(self):
        N = 3
        gspace = rot3dOnR3(maximum_frequency=N)
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(N)], name="irrepssum")
        t1 = FieldType(gspace, [irreps]*2)
        t2 = FieldType(gspace, [irreps]*3)
        # t1 = FieldType(gspace, [gspace.regular_repr]*2)
        # t2 = FieldType(gspace, [gspace.regular_repr]*3)
        self.check(t1, t2, 3)

    def test_0d(self):
        N = 3
        gspace = no_base_space(so3_group(maximum_frequency=N))
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(N)], name="irrepssum")
        t1 = FieldType(gspace, [irreps]*2)
        t2 = FieldType(gspace, [irreps]*3)
        # t1 = FieldType(gspace, [gspace.regular_repr]*2)
        # t2 = FieldType(gspace, [gspace.regular_repr]*3)
        self.check(t1, t2, 0)

    def check(self, r1: FieldType, r2: FieldType, D: int = 2):
        
        np.set_printoptions(precision=7, threshold=60000, suppress=True)
        
        assert r1.gspace == r2.gspace
        
        assert r2.size >= r1.size
    
        s = 7
        
        c = int(s//2)

        if D == 2:
            cl = R2Conv(r1, r2, s,
                        # sigma=[0.01] + [0.6]*int(s//2),
                        frequencies_cutoff=3.)
        elif D == 3:
            cl = R3Conv(r1, r2, s,
                        # sigma=[0.01] + [0.6]*int(s//2),
                        frequencies_cutoff=3.)
        elif D == 0:
            cl = Linear(r1, r2)
        else:
            raise ValueError()

        for _ in range(20):
            init.deltaorthonormal_init(cl.weights.data, cl.basisexpansion)
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            
            filter, _ = cl.expand_parameters()
            
            center = filter[(...,) + (c,)*D]
            
            # print(center.detach().numpy())

            id = torch.einsum("ab,bc->ac", center.t(), center)
            # print(id.detach().numpy())
            
            # we actually check that the matrix is a "multiple" of an orthonormal matrix because some energy might
            # be lost on surrounding cells
            id /= id.max()

            self.assertTrue(torch.allclose(id, torch.eye(r1.size), atol=5e-7))
            
            # filter /= (filter[..., c, c]**2 / filter.shape[1]).sum().sqrt()[..., None, None]

            filter[(...,) + (c,)*D].fill_(0)

            # max, _ = filter.reshape(-1, s, s,).abs().max(0)
            # print(max.detach().numpy())
            # print("\n")
            
            self.assertTrue(torch.allclose(filter, torch.zeros_like(filter), atol=1e-7))


if __name__ == '__main__':
    unittest.main()
