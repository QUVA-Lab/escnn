import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import numpy as np
import torch


class TestGroupedConv(TestCase):
    
    def test_cyclic(self):
        N = 4
        g = rot2dOnR2(N)

        groups = 5
        
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        s = 7
        sigma = None
        # fco = lambda r: 1. * r * np.pi
        fco = None

        cl = R2PointConv(r1, r2, groups=groups,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_so2(self):
        N = 5
        g = rot2dOnR2(-1, N)
        groups = 5

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = FieldType(g, reprs * groups)
        r2 = FieldType(g, reprs * groups)
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2PointConv(r1, r2, groups=groups,
                        sigma=sigma,
                        width=2.,
                        n_rings=3,
                        frequencies_cutoff=fco,
                        bias=True)

        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = flipRot2dOnR2(N, axis=np.pi/3)

        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
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
        cl = R2PointConv(r1, r2, groups=groups,
                    sigma=sigma,
                    width=2.,
                    n_rings=3,
                    frequencies_cutoff=fco,
                    bias=True)

        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 8
        g = flipRot2dOnR2(-1, N)
        groups = 5
        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = FieldType(g, reprs * groups)
        r2 = FieldType(g, reprs * groups)

        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2PointConv(r1, r2, groups=groups,
                    sigma=sigma,
                    width=2.,
                    n_rings=3,
                    frequencies_cutoff=fco,
                    bias=True)

        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_flip(self):
        # g = flip2dOnR2(axis=np.pi/3)
        g = flip2dOnR2(axis=np.pi/2)
        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
    
        s = 9
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2PointConv(r1, r2, groups=groups,
                    sigma=sigma,
                    width=2.,
                    n_rings=3,
                    frequencies_cutoff=fco,
                    bias=True)

        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(32):
            init.generalized_he_init(cl.weights.data, cl.basissampler)
            cl.eval()
            cl.check_equivariance()

    def test_so3(self):
        g = rot3dOnR3(3)

        groups = 2
        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        r1 = FieldType(g, reprs * groups)
        r2 = FieldType(g, reprs * groups)

        sigma = None
        fco = None
        cl = R3PointConv(r1, r2, groups=groups,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)
        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler, cache=True)
            cl.eval()
            cl.check_equivariance()

    def test_octa(self):
        g = octaOnR3()

        groups = 5
        reprs = g.irreps
        r1 = FieldType(g, reprs * groups)
        r2 = FieldType(g, reprs * groups)

        sigma = None
        fco = None
        cl = R3PointConv(r1, r2, groups=groups,
                         sigma=sigma,
                         width=2.,
                         n_rings=3,
                         frequencies_cutoff=fco,
                         bias=True)
        cl = cl.to('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(4):
            init.generalized_he_init(cl.weights.data, cl.basissampler, cache=True)
            # cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
