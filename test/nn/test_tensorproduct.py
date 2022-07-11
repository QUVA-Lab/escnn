import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *

import numpy as np

import torch


class TestTensor(TestCase):

    def test_so2(self):
        g = no_base_space(so2_group(10))
        G = g.fibergroup

        for f1 in range(1, 4):
            for f2 in range(1, 4):
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(atol=1e-5, rtol=1e-3)

    def test_o2(self):
        g = no_base_space(o2_group(10))

        G = g.fibergroup

        for f1 in range(1, 4):
            for f2 in range(1, 4):
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(atol=1e-5, rtol=1e-3)

    def test_so3(self):
        g = no_base_space(so3_group(3))
        G = g.fibergroup

        for f1 in range(1, 3):
            for f2 in range(1, 3):
                print(f1, f2)
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(atol=1e-5, rtol=1e-3)

    def test_o3(self):
        g = no_base_space(o3_group(3))
        G = g.fibergroup

        for f1 in range(1, 3):
            for f2 in range(1, 3):
                print(f1, f2)
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(atol=1e-5, rtol=1e-3)

    def test_with_spatial_dimns(self):
        g = rot2dOnR2(-1)

        G = g.fibergroup

        for f1 in range(1, 4):
            for f2 in range(1, 4):
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(rtol=1e-1)

        g = rot3dOnR3()

        G = g.fibergroup

        for f1 in range(1, 3):
            for f2 in range(1, 3):
                cl = TensorProductModule.construct(g, 3, G.bl_regular_representation(f1), G.bl_regular_representation(f2))
                cl.check_equivariance(rtol=1e-1)


if __name__ == '__main__':
    unittest.main()

